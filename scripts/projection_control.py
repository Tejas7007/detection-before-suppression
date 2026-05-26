"""
Feature-specific intervention: remove ONLY the duplication direction
(probe weight vector) instead of replacing the entire residual stream.

If projecting out the probe direction removes S-bias but projecting out
a random direction of the SAME (small) magnitude does not, the effect
is feature-specific, not just S2 disruption.
"""
import os; os.environ["HF_TOKEN"] = ""
import torch, json, numpy as np, sys, random
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES, CANDIDATE_NAMES

DEVICE = "cuda"
MODEL_NAME = "EleutherAI/pythia-160m-deduped"
SEED = 42

def get_s2_positions(tokens, s_ids):
    s2_positions = []
    for i in range(tokens.shape[0]):
        s_tok = s_ids[i].item(); cnt = 0; pos = -1
        for j in range(1, tokens.shape[1]):
            if tokens[i, j].item() == s_tok:
                cnt += 1
                if cnt == 2: pos = j; break
        s2_positions.append(pos)
    return s2_positions

print("=" * 60)
print("  PROJECTION-BASED INTERVENTION")
print("  Remove ONLY the duplication direction vs random direction")
print("=" * 60)

hf = AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision="step2000")
model = HookedTransformer.from_pretrained(MODEL_NAME, hf_model=hf, device=DEVICE,
    center_writing_weights=True, center_unembed=True, fold_ln=True)
del hf

random.seed(SEED)
single_names = [model.tokenizer.encode(" "+n, add_special_tokens=False)[0]
                for n in CANDIDATE_NAMES if len(model.tokenizer.encode(" "+n, add_special_tokens=False))==1]

# Step 1: Collect activations and train probe to get duplication direction
print("\n1. Training duplication probe to get direction...")
probe_acts = []; probe_labels = []
template_data = []

for tmpl in ALL_TEMPLATES[:10]:
    ds = IOIDataset(model=model, n_prompts=30, templates=[tmpl], symmetric=True, seed=SEED)
    tokens = model.to_tokens(ds.prompts).to(DEVICE)
    io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
    s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)
    s2_pos = get_s2_positions(tokens, s_ids)
    n = tokens.shape[0]

    ctrl = tokens.clone()
    for i in range(n):
        if s2_pos[i] > 0:
            cands = [t for t in single_names if t != s_ids[i].item() and t != io_ids[i].item()]
            if cands: ctrl[i, s2_pos[i]] = random.choice(cands)

    template_data.append((tokens, ctrl, io_ids, s_ids, s2_pos, n))

    # Get activations at layer 5 (best probe layer during dip)
    for probe_layer in [5]:
        _, cache_ioi = model.run_with_cache(tokens, remove_batch_dim=False)
        _, cache_ctrl = model.run_with_cache(ctrl, remove_batch_dim=False)
        for i in range(n):
            if s2_pos[i] > 0:
                probe_acts.append(cache_ioi["blocks.%d.hook_resid_post" % probe_layer][i, s2_pos[i], :].detach().cpu().float().numpy())
                probe_labels.append(1)
                probe_acts.append(cache_ctrl["blocks.%d.hook_resid_post" % probe_layer][i, s2_pos[i], :].detach().cpu().float().numpy())
                probe_labels.append(0)
        del cache_ioi, cache_ctrl; torch.cuda.empty_cache()

X = np.array(probe_acts)
y = np.array(probe_labels)
clf = LogisticRegression(max_iter=2000, random_state=SEED, C=1.0)
clf.fit(X, y)
probe_acc = clf.score(X, y)
print("  Probe accuracy: %.1f%%" % (probe_acc * 100))

# The duplication direction is the probe's weight vector
dup_direction = torch.tensor(clf.coef_[0], dtype=torch.float32, device=DEVICE)
dup_direction = dup_direction / dup_direction.norm()  # normalize
print("  Duplication direction norm: %.4f" % dup_direction.norm().item())

# Shuffled probe direction
y_shuffled = y.copy()
np.random.seed(SEED + 1)
np.random.shuffle(y_shuffled)
clf_shuffled = LogisticRegression(max_iter=2000, random_state=SEED, C=1.0)
clf_shuffled.fit(X, y_shuffled)
shuffled_dir = torch.tensor(clf_shuffled.coef_[0], dtype=torch.float32, device=DEVICE)
shuffled_dir = shuffled_dir / shuffled_dir.norm()

# Random directions (5 for averaging)
random_dirs = []
for _ in range(5):
    rd = torch.randn(768, device=DEVICE)
    rd = rd / rd.norm()
    random_dirs.append(rd)

# Orthogonal random direction (orthogonal to dup_direction)
ortho_dir = torch.randn(768, device=DEVICE)
ortho_dir = ortho_dir - (ortho_dir @ dup_direction) * dup_direction
ortho_dir = ortho_dir / ortho_dir.norm()

print("  Dot(dup, ortho): %.6f (should be ~0)" % (dup_direction @ ortho_dir).item())

# Step 2: Run interventions
print("\n2. Running projection-based interventions...")

conditions = {
    "baseline": None,
    "remove_dup_direction": dup_direction,
    "remove_ortho_direction": ortho_dir,
    "remove_shuffled_direction": shuffled_dir,
}
for i, rd in enumerate(random_dirs):
    conditions["remove_random_%d" % i] = rd

# Also test different strengths (dose response)
strengths = [0.5, 1.0, 2.0, 4.0]

all_results = {}

for cond_name, direction in conditions.items():
    for strength in (strengths if direction is not None else [1.0]):
        key = "%s_%.1fx" % (cond_name, strength) if direction is not None else "baseline"
        ld_values = []

        for tokens, ctrl, io_ids, s_ids, s2_pos, n in template_data:
            if direction is None:
                logits = model(tokens)[:, -1, :]
            else:
                def make_proj_hook(d=direction, s=strength, sp=s2_pos, nn=n):
                    def hook_fn(value, hook):
                        for i in range(nn):
                            if sp[i] > 0:
                                proj = (value[i, sp[i], :] @ d) * d
                                value[i, sp[i], :] -= s * proj
                        return value
                    return hook_fn
                hook = ("blocks.5.hook_resid_post", make_proj_hook())
                logits = model.run_with_hooks(tokens, fwd_hooks=[hook])[:, -1, :]

            for i in range(n):
                ld_values.append(logits[i, io_ids[i]].item() - logits[i, s_ids[i]].item())

        mean_ld = float(np.mean(ld_values))
        acc = float((np.array(ld_values) > 0).mean())
        all_results[key] = {"ld": round(mean_ld, 4), "acc": round(acc, 4)}

        if direction is not None and strength == 1.0:
            bl = all_results["baseline"]["ld"]
            print("  %s: LD=%.4f (change=%+.4f), acc=%.1f%%" % (
                cond_name, mean_ld, mean_ld - bl, acc * 100))

# Print dose response
bl = all_results["baseline"]["ld"]
print("\n3. Dose response (strength multiplier):")
print("  %-8s  %-12s  %-12s  %-12s" % ("Strength", "Dup dir", "Ortho dir", "Random dir"))
for s in strengths:
    dup_v = all_results.get("remove_dup_direction_%.1fx" % s, {}).get("ld", 0)
    ort_v = all_results.get("remove_ortho_direction_%.1fx" % s, {}).get("ld", 0)
    rnd_v = all_results.get("remove_random_0_%.1fx" % s, {}).get("ld", 0)
    print("  %-8s  %+.4f (%+.4f)  %+.4f (%+.4f)  %+.4f (%+.4f)" % (
        "%.1fx" % s,
        dup_v, dup_v - bl,
        ort_v, ort_v - bl,
        rnd_v, rnd_v - bl))

# Summary
print("\n4. Summary at 1x strength:")
print("  Baseline:           LD=%.4f" % all_results["baseline"]["ld"])
print("  Remove dup dir:     LD=%.4f (change=%+.4f)" % (
    all_results["remove_dup_direction_1.0x"]["ld"],
    all_results["remove_dup_direction_1.0x"]["ld"] - bl))
print("  Remove ortho dir:   LD=%.4f (change=%+.4f)" % (
    all_results["remove_ortho_direction_1.0x"]["ld"],
    all_results["remove_ortho_direction_1.0x"]["ld"] - bl))
print("  Remove shuffled:    LD=%.4f (change=%+.4f)" % (
    all_results["remove_shuffled_direction_1.0x"]["ld"],
    all_results["remove_shuffled_direction_1.0x"]["ld"] - bl))
rand_changes = [all_results["remove_random_%d_1.0x" % i]["ld"] - bl for i in range(5)]
print("  Remove random (avg): change=%+.4f (±%.4f)" % (np.mean(rand_changes), np.std(rand_changes)))
print("\n  If dup >> ortho and dup >> random: FEATURE SPECIFIC")
print("  If dup ~ random: NOT feature specific (S2 disruption)")

with open("results/projection_controls.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved: results/projection_controls.json")

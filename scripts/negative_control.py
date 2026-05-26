import os; os.environ["HF_TOKEN"] = ""
import torch, json, numpy as np, sys, random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES, CANDIDATE_NAMES

DEVICE = "cuda"
MODEL_NAME = "EleutherAI/pythia-160m-deduped"

def get_s2_positions(tokens, s_ids):
    s2_positions = []
    for i in range(tokens.shape[0]):
        s_tok = s_ids[i].item(); s_count = 0; s2_pos = -1
        for j in range(1, tokens.shape[1]):
            if tokens[i, j].item() == s_tok:
                s_count += 1
                if s_count == 2: s2_pos = j; break
        s2_positions.append(s2_pos)
    return s2_positions

print("=" * 60)
print("  NEGATIVE CONTROLS FOR CAUSAL INTERVENTION")
print("=" * 60)

# Load step 2000 (strongest effect)
hf = AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision="step2000")
model = HookedTransformer.from_pretrained(MODEL_NAME, hf_model=hf, device=DEVICE,
    center_writing_weights=True, center_unembed=True, fold_ln=True)
del hf

random.seed(42)
single_names = [model.tokenizer.encode(" "+n, add_special_tokens=False)[0]
                for n in CANDIDATE_NAMES if len(model.tokenizer.encode(" "+n, add_special_tokens=False))==1]

all_baseline_ld = []
all_random_ld = []
all_wrong_pos_ld = []
all_real_ld = []

for tmpl in ALL_TEMPLATES[:10]:
    ds = IOIDataset(model=model, n_prompts=30, templates=[tmpl], symmetric=True, seed=42)
    tokens = model.to_tokens(ds.prompts).to(DEVICE)
    io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
    s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)
    s2_pos = get_s2_positions(tokens, s_ids)
    n = tokens.shape[0]

    # Control tokens
    ctrl = tokens.clone()
    for i in range(n):
        if s2_pos[i] > 0:
            cands = [t for t in single_names if t != s_ids[i].item() and t != io_ids[i].item()]
            if cands: ctrl[i, s2_pos[i]] = random.choice(cands)

    # Cache
    _, cache_ioi = model.run_with_cache(tokens, remove_batch_dim=False)
    _, cache_ctrl = model.run_with_cache(ctrl, remove_batch_dim=False)

    # Baseline
    logits = model(tokens)[:, -1, :]
    for i in range(n):
        all_baseline_ld.append(logits[i, io_ids[i]].item() - logits[i, s_ids[i]].item())

    # REAL intervention: patch control's S2 into IOI (layers 3-5)
    def make_real_hook(li):
        def hook_fn(value, hook):
            for i in range(n):
                if s2_pos[i] > 0:
                    value[i, s2_pos[i], :] = cache_ctrl["blocks.%d.hook_resid_post" % li][i, s2_pos[i], :]
            return value
        return hook_fn
    hooks = [("blocks.%d.hook_resid_post" % l, make_real_hook(l)) for l in range(3, 6)]
    logits_real = model.run_with_hooks(tokens, fwd_hooks=hooks)[:, -1, :]
    for i in range(n):
        all_real_ld.append(logits_real[i, io_ids[i]].item() - logits_real[i, s_ids[i]].item())

    # CONTROL 1: Random direction at S2 with same norm as real intervention
    def make_random_hook(li):
        def hook_fn(value, hook):
            for i in range(n):
                if s2_pos[i] > 0:
                    real_diff = cache_ctrl["blocks.%d.hook_resid_post" % li][i, s2_pos[i], :] - \
                                cache_ioi["blocks.%d.hook_resid_post" % li][i, s2_pos[i], :]
                    norm = real_diff.norm()
                    rand_dir = torch.randn_like(real_diff)
                    rand_dir = rand_dir / rand_dir.norm() * norm
                    value[i, s2_pos[i], :] += rand_dir
            return value
        return hook_fn
    hooks_rand = [("blocks.%d.hook_resid_post" % l, make_random_hook(l)) for l in range(3, 6)]
    logits_rand = model.run_with_hooks(tokens, fwd_hooks=hooks_rand)[:, -1, :]
    for i in range(n):
        all_random_ld.append(logits_rand[i, io_ids[i]].item() - logits_rand[i, s_ids[i]].item())

    # CONTROL 2: Real intervention but at WRONG position (position 1 instead of S2)
    def make_wrong_hook(li):
        def hook_fn(value, hook):
            for i in range(n):
                value[i, 1, :] = cache_ctrl["blocks.%d.hook_resid_post" % li][i, 1, :]
            return value
        return hook_fn
    hooks_wrong = [("blocks.%d.hook_resid_post" % l, make_wrong_hook(l)) for l in range(3, 6)]
    logits_wrong = model.run_with_hooks(tokens, fwd_hooks=hooks_wrong)[:, -1, :]
    for i in range(n):
        all_wrong_pos_ld.append(logits_wrong[i, io_ids[i]].item() - logits_wrong[i, s_ids[i]].item())

    del cache_ioi, cache_ctrl; torch.cuda.empty_cache()

bl = float(np.mean(all_baseline_ld))
print("\nStep 2000 Results:")
print("  Baseline LD:              %.4f (acc=%.1f%%)" % (bl, (np.array(all_baseline_ld)>0).mean()*100))
print("  Real intervention (S2):   %.4f (change=%+.4f)" % (np.mean(all_real_ld), np.mean(all_real_ld)-bl))
print("  Random direction (S2):    %.4f (change=%+.4f)" % (np.mean(all_random_ld), np.mean(all_random_ld)-bl))
print("  Wrong position (pos 1):   %.4f (change=%+.4f)" % (np.mean(all_wrong_pos_ld), np.mean(all_wrong_pos_ld)-bl))
print("\n  If real >> random and real >> wrong_pos: effect is specific to duplication at S2")

results = {
    "step": 2000,
    "baseline_ld": round(bl, 4),
    "real_intervention": round(float(np.mean(all_real_ld)), 4),
    "random_direction": round(float(np.mean(all_random_ld)), 4),
    "wrong_position": round(float(np.mean(all_wrong_pos_ld)), 4),
    "real_change": round(float(np.mean(all_real_ld))-bl, 4),
    "random_change": round(float(np.mean(all_random_ld))-bl, 4),
    "wrong_pos_change": round(float(np.mean(all_wrong_pos_ld))-bl, 4),
}
with open("results/negative_controls.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: results/negative_controls.json")

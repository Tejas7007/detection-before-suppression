"""
PolyPythias IOI Analysis - Fixed loading
"""
import torch, json, os, time, shutil
import numpy as np
from collections import Counter
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES

DEVICE = "cuda"
TEMPLATES = ALL_TEMPLATES[:15]
PPT, SEED = 20, 42
RESULTS_FILE = "results/polypythias_ioi.json"

MODELS = [
    ("EleutherAI/pythia-160m-seed1", "seed1"),
    ("EleutherAI/pythia-160m-seed3", "seed3"),
    ("EleutherAI/pythia-160m-seed5", "seed5"),
    ("EleutherAI/pythia-160m-data-seed1", "data-seed1"),
    ("EleutherAI/pythia-160m-data-seed2", "data-seed2"),
    ("EleutherAI/pythia-160m-data-seed3", "data-seed3"),
    ("EleutherAI/pythia-160m-weight-seed1", "weight-seed1"),
    ("EleutherAI/pythia-160m-weight-seed2", "weight-seed2"),
    ("EleutherAI/pythia-160m-weight-seed3", "weight-seed3"),
]

CHECKPOINTS = [0, 512, 1000, 2000, 3000, 5000, 8000, 10000, 33000, 143000]

def clear_cache():
    for d in ['/workspace/.hf_home/hub', os.path.expanduser('~/.cache/huggingface/hub')]:
        if os.path.exists(d):
            for f in os.listdir(d):
                if f.startswith('models--'):
                    shutil.rmtree(os.path.join(d, f), ignore_errors=True)

def save_results(results):
    os.makedirs("results", exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)

def load_model(model_name, step):
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, revision="step%d" % step,
    )
    model = HookedTransformer.from_pretrained(
        "EleutherAI/pythia-160m-deduped",
        hf_model=hf_model,
        device=DEVICE,
        center_writing_weights=True,
        center_unembed=True,
        fold_ln=True,
    )
    del hf_model
    return model

def main():
    print("=" * 60)
    print("  POLYPYTHIAS IOI ANALYSIS")
    print("  %d models x %d checkpoints" % (len(MODELS), len(CHECKPOINTS)))
    print("  Started: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    t0 = time.time()

    for model_name, label in MODELS:
        if label not in results:
            results[label] = {"model": model_name, "checkpoints": {}}

        print("\n" + "=" * 40)
        print("  %s" % label)
        print("=" * 40)

        for step in CHECKPOINTS:
            step_key = "step_%d" % step

            if step_key in results[label].get("checkpoints", {}):
                print("  Step %d done, skip" % step)
                continue

            print("\n  --- Step %d ---" % step)
            try:
                clear_cache()
                model = load_model(model_name, step)
            except Exception as e:
                print("    FAILED: %s" % str(e)[:100])
                continue

            all_lds = []
            total = 0

            for tmpl in TEMPLATES:
                try:
                    ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl],
                                    symmetric=True, seed=SEED)
                    tokens = model.to_tokens(ds.prompts).to(DEVICE)
                    io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
                    s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)
                    logits = model(tokens)
                    last = logits[:, -1, :].float()
                    for i in range(len(io_ids)):
                        total += 1
                        ld = last[i, io_ids[i]].item() - last[i, s_ids[i]].item()
                        all_lds.append(ld)
                except:
                    continue

            if total == 0:
                del model
                torch.cuda.empty_cache()
                continue

            lds = np.array(all_lds)
            accuracy = float((lds > 0).mean())

            results[label]["checkpoints"][step_key] = {
                "accuracy": round(accuracy, 4),
                "mean_ld": round(float(lds.mean()), 4),
                "pct_s_preferred": round(float((lds < 0).mean()), 4),
                "n_examples": total,
            }
            save_results(results)

            print("    Acc=%.3f, LD=%.4f, pct_S=%.1f%%" % (
                accuracy, lds.mean(), (lds < 0).mean() * 100))

            del model
            torch.cuda.empty_cache()

    # Summary table
    print("\n" + "=" * 60)
    print("  SUMMARY: IOI Accuracy at Each Step")
    print("=" * 60)
    header = "%10s" % "Step"
    for _, label in MODELS:
        header += " %11s" % label
    print(header)

    for step in CHECKPOINTS:
        sk = "step_%d" % step
        row = "%10d" % step
        for _, label in MODELS:
            if label in results and sk in results[label].get("checkpoints", {}):
                acc = results[label]["checkpoints"][sk]["accuracy"]
                row += " %10.1f%%" % (acc * 100)
            else:
                row += " %11s" % "---"
        print(row)

    elapsed = time.time() - t0
    print("\n  Time: %.0fs (%.1f hours)" % (elapsed, elapsed / 3600))
    print("  DONE.")

if __name__ == "__main__":
    main()

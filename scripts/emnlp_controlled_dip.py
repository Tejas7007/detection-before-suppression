"""
Controlled Dip: predictive control of the S-preference dip in a real LM
========================================================================
The ceiling-raiser. We train real GPT-2-style LMs that differ ONLY in
the rate of injected IOI-task examples, and show the dip depth/recovery
is a smooth, predictable function of that rate.

Standardized (Wang et al. 2022) generalization protocol:
  - Templates are SHARED between injection and eval (the IOI task IS
    defined by these structures). Generalization is tested over NAMES:
    injection names and eval names are DISJOINT. The model must apply
    IOI to name pairs it never saw in IOI context.
  - Two eval sets each checkpoint:
      * held-out  : eval names  (the generalization claim)
      * in-dist   : inject names (diagnostic: did it learn IOI at all)
    This lets us always separate "didn't learn" from "didn't generalize."

Design (rigor-first):
  - Base corpus: WikiText-103 (established "deep dip, no recovery").
  - Task signal: synthetic IOI sentences whose answer is the IO. Injected
    at the SEQUENCE level with prob = rate, from a pre-built shared pool.
  - Conditions: rate in {0.0, 0.05, 0.15, 0.30}.
  - ISOLATION: identical init (same seed) and identical WikiText stream
    across conditions; injection draws are seeded and nested.
  - Continuous + discrete metrics (accuracy AND logit difference) to
    preempt the metric-artifact (Schaeffer) objection.

Order: [0.0, 0.30, 0.15, 0.05] so baseline + strongest injection first.
Output: results/emnlp_controlled_dip.json
"""

import os, gc, json, time, sys, random, math
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

DEVICE = "cuda"
SEED = 42
CTX = 256
D_MODEL = 512
N_LAYER = 8
N_HEAD = 8
BATCH = 32
LR = 3e-4
WARMUP = 1000
TOTAL_STEPS = 40000
EVAL_EVERY = 2000
INJECTION_RATES = [0.0, 0.30, 0.15, 0.05]  # baseline + strongest first
RESULTS_PATH = "results/emnlp_controlled_dip.json"
CKPT_DIR = "results/controlled_dip_ckpts"

# SHARED IOI templates (used for BOTH injection and eval). Answer = IO.
TEMPLATE_POOL = [
    "When {IO} and {S} went to the {place}, {S} gave a {obj} to {IO}",
    "Then {IO} and {S} arrived at the {place}, and {S} handed a {obj} to {IO}",
    "After {IO} and {S} left the {place}, {S} passed a {obj} to {IO}",
    "While {IO} and {S} sat in the {place}, {S} offered a {obj} to {IO}",
    "Because {IO} and {S} visited the {place}, {S} sold a {obj} to {IO}",
    "Once {IO} and {S} entered the {place}, {S} brought a {obj} to {IO}",
    "Although {IO} and {S} stayed at the {place}, {S} lent a {obj} to {IO}",
    "Before {IO} and {S} reached the {place}, {S} showed a {obj} to {IO}",
]
PLACES = ["store", "park", "school", "office", "garden", "market", "station",
          "library", "museum", "harbor", "theater", "stadium"]
OBJECTS = ["book", "ring", "drink", "ball", "letter", "coin", "pen", "key",
           "card", "rose", "watch", "map"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_single_token_names(tokenizer):
    candidates = [
        "John","Mary","Tom","James","Robert","Michael","William","David","Richard",
        "Joseph","Charles","Thomas","Daniel","Paul","Mark","Donald","George","Kenneth",
        "Steven","Edward","Brian","Ronald","Anthony","Kevin","Jason","Matthew","Gary",
        "Timothy","Jose","Larry","Jeffrey","Frank","Scott","Eric","Stephen","Andrew",
        "Raymond","Gregory","Joshua","Jerry","Dennis","Walter","Patrick","Peter","Harold",
        "Henry","Carl","Arthur","Ryan","Roger","Joe","Juan","Jack","Albert","Jonathan",
        "Justin","Terry","Gerald","Keith","Samuel","Willie","Ralph","Lawrence","Nicholas",
        "Roy","Benjamin","Bruce","Brandon","Adam","Harry","Fred","Wayne","Billy","Steve",
        "Louis","Jeremy","Aaron","Randy","Howard","Eugene","Carlos","Russell","Bobby",
        "Victor","Martin","Ernest","Phillip","Todd","Jesse","Craig","Alan","Shawn",
        "Clarence","Sean","Philip","Chris","Johnny","Earl","Jimmy","Antonio","Danny",
        "Bryan","Tony","Luis","Mike","Stanley","Leonard","Nathan","Dale","Manuel",
        "Rodney","Curtis","Norman","Allen","Marvin","Vincent","Glenn","Jeffery","Travis",
        "Jeff","Chad","Jacob","Lee","Melvin","Alfred","Kyle","Francis","Bradley",
        "Jesus","Herbert","Frederick","Ray","Joel","Edwin","Don","Eddie","Ricky",
        "Troy","Randall","Barry","Alexander","Bernard","Mario","Leroy","Francisco",
        "Marcus","Micheal","Theodore","Clifford","Miguel","Oscar","Jay","Jim",
        "Calvin","Alex","Jon","Ronnie","Bill","Lloyd","Tommy","Leon","Derek","Warren",
        "Darrell","Jerome","Floyd","Leo","Alvin","Tim","Wesley","Gordon","Dean","Greg",
        "Jorge","Dustin","Pedro","Derrick","Dan","Lewis","Zachary","Corey","Herman",
        "Maurice","Vernon","Roberto","Clyde","Glen","Hector","Shane","Ricardo","Sam",
    ]
    single, seen = [], set()
    for nm in candidates:
        if nm in seen: continue
        seen.add(nm)
        ids = tokenizer.encode(" " + nm)
        if len(ids) == 1:
            single.append((nm, ids[0]))
    return single


def make_ioi_text(name_pool, rng):
    io, s = rng.sample([n[0] for n in name_pool], 2)
    tmpl = rng.choice(TEMPLATE_POOL)
    return tmpl.format(IO=io, S=s, place=rng.choice(PLACES), obj=rng.choice(OBJECTS))


def make_injected_sequence(tokenizer, name_pool, rng):
    parts, total = [], 0
    while total < CTX + 8:
        t = make_ioi_text(name_pool, rng) + "."
        parts.append(t)
        total += len(tokenizer.encode(" " + t))
    ids = tokenizer.encode(" ".join(parts))[:CTX]
    if len(ids) < CTX:
        ids += [tokenizer.eos_token_id] * (CTX - len(ids))
    return ids


def build_injection_pool(tokenizer, inject_names, n_pool=4000, seed=SEED + 999):
    rng = random.Random(seed)
    return torch.tensor([make_injected_sequence(tokenizer, inject_names, rng)
                         for _ in range(n_pool)], dtype=torch.long)


def build_eval_set(tokenizer, name_pool, n=300, seed=SEED):
    rng = random.Random(seed)
    name_ids = {nm: tid for nm, tid in name_pool}
    names_only = [n[0] for n in name_pool]
    prompts, io_ids, s_ids = [], [], []
    for _ in range(n):
        io, s = rng.sample(names_only, 2)
        tmpl = rng.choice(TEMPLATE_POOL)
        full = tmpl.format(IO=io, S=s, place=rng.choice(PLACES), obj=rng.choice(OBJECTS))
        suffix = " " + io
        assert full.endswith(suffix)
        prompts.append(full[:-len(suffix)])
        io_ids.append(name_ids[io]); s_ids.append(name_ids[s])
    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    return enc["input_ids"], enc["attention_mask"], torch.tensor(io_ids), torch.tensor(s_ids)


@torch.no_grad()
def evaluate_ioi(model, eval_set):
    eval_ids, eval_mask, io_ids, s_ids = eval_set
    model.eval()
    ids = eval_ids.to(DEVICE); mask = eval_mask.to(DEVICE)
    io = io_ids.to(DEVICE); s = s_ids.to(DEVICE)
    logits = model(input_ids=ids, attention_mask=mask).logits
    last_pos = mask.sum(1) - 1
    bidx = torch.arange(ids.shape[0], device=DEVICE)
    last_logits = logits[bidx, last_pos]
    ld = (last_logits[bidx, io] - last_logits[bidx, s])
    model.train()
    return float((ld > 0).float().mean().item()), float(ld.mean().item())


def load_wikitext_chunks(tokenizer):
    from datasets import load_dataset
    log("  loading WikiText-103...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    log("  tokenizing...")
    chunks, buf = [], []
    for ex in ds:
        t = ex["text"].strip()
        if not t: continue
        buf.extend(tokenizer.encode(t))
        while len(buf) >= CTX:
            chunks.append(buf[:CTX]); buf = buf[CTX:]
        if len(chunks) >= 400000:
            break
    log(f"  {len(chunks)} WikiText chunks of {CTX} tokens")
    return torch.tensor(chunks, dtype=torch.long)


def train_condition(rate, wikitext, inject_pool, eval_heldout, eval_indist, vocab_size):
    torch.manual_seed(SEED); np.random.seed(SEED)
    cfg = GPT2Config(vocab_size=vocab_size, n_positions=CTX, n_embd=D_MODEL,
                     n_layer=N_LAYER, n_head=N_HEAD)
    model = GPT2LMHeadModel(cfg).to(DEVICE); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    def lr_lambda(step):
        if step < WARMUP: return step / WARMUP
        prog = (step - WARMUP) / max(1, TOTAL_STEPS - WARMUP)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    data_rng = np.random.default_rng(SEED)
    inject_rng = np.random.default_rng(SEED + 777)
    pool_rng = np.random.default_rng(SEED + 1234)

    trajectory = []
    n_chunks, n_pool = wikitext.shape[0], inject_pool.shape[0]
    t0 = time.time()
    for step in range(TOTAL_STEPS + 1):
        if step % EVAL_EVERY == 0:
            ho_acc, ho_ld = evaluate_ioi(model, eval_heldout)
            id_acc, id_ld = evaluate_ioi(model, eval_indist)
            elapsed = (time.time() - t0) / 60
            trajectory.append({"step": step, "heldout_acc": ho_acc, "heldout_ld": ho_ld,
                               "indist_acc": id_acc, "indist_ld": id_ld})
            log(f"    [rate={rate:.2f}] step {step:>6}: heldout={ho_acc*100:5.1f}%"
                f"(LD{ho_ld:+.2f})  indist={id_acc*100:5.1f}%(LD{id_ld:+.2f})  ({elapsed:.1f}m)")
            _save_traj(rate, trajectory)
            if step in (10000, 20000, TOTAL_STEPS):
                os.makedirs(CKPT_DIR, exist_ok=True)
                torch.save(model.state_dict(), f"{CKPT_DIR}/rate{int(rate*100)}_step{step}.pt")
        if step == TOTAL_STEPS: break

        chunk_idx = data_rng.integers(0, n_chunks, size=BATCH)
        draws = inject_rng.random(BATCH)
        pool_idx = pool_rng.integers(0, n_pool, size=BATCH)
        seq_list = [inject_pool[pool_idx[b]] if draws[b] < rate else wikitext[chunk_idx[b]]
                    for b in range(BATCH)]
        batch = torch.stack(seq_list).to(DEVICE)
        loss = model(input_ids=batch, labels=batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()

    del model, opt; torch.cuda.empty_cache(); gc.collect()
    return trajectory


def _save_traj(rate, trajectory):
    results = {}
    if os.path.exists(RESULTS_PATH):
        try: results = json.load(open(RESULTS_PATH))
        except: pass
    results[f"rate_{rate}"] = {"rate": rate, "trajectory": trajectory,
        "config": {"d_model": D_MODEL, "n_layer": N_LAYER, "n_head": N_HEAD,
                   "ctx": CTX, "batch": BATCH, "lr": LR, "total_steps": TOTAL_STEPS,
                   "protocol": "wang_heldout_names_shared_templates"}}
    json.dump(results, open(RESULTS_PATH, "w"), indent=2)


def main():
    os.makedirs("results", exist_ok=True)
    # Fresh start under the new protocol (preserve old baseline separately).
    if os.path.exists(RESULTS_PATH):
        os.rename(RESULTS_PATH, RESULTS_PATH.replace(".json", "_strict_eval_OLD.json"))
        log("  archived previous (strict-eval) results")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    all_names = get_single_token_names(tokenizer)
    log(f"{len(all_names)} single-token names")
    k = int(len(all_names) * 0.6)
    inject_names, eval_names = all_names[:k], all_names[k:]
    log(f"  {len(inject_names)} inject names, {len(eval_names)} eval names (disjoint)")
    log(f"  templates SHARED ({len(TEMPLATE_POOL)}); generalization tested over names (Wang)")

    eval_heldout = build_eval_set(tokenizer, eval_names, n=300, seed=SEED)
    eval_indist = build_eval_set(tokenizer, inject_names, n=300, seed=SEED + 5)
    log(f"  eval sets built (held-out + in-distribution)")

    log("  building injection pool...")
    inject_pool = build_injection_pool(tokenizer, inject_names, n_pool=4000)
    log(f"  injection pool: {inject_pool.shape}")

    wikitext = load_wikitext_chunks(tokenizer)

    for rate in INJECTION_RATES:
        log("=" * 60)
        log(f"TRAINING condition: injection rate = {rate}")
        log("=" * 60)
        train_condition(rate, wikitext, inject_pool, eval_heldout, eval_indist, vocab_size)

    log("\n" + "=" * 60)
    log("CONTROLLED DIP SUMMARY (held-out names, shared templates)")
    log("=" * 60)
    results = json.load(open(RESULTS_PATH))
    log(f"  {'rate':>6} {'ho_floor':>9} {'ho_final':>9} {'id_final':>9} {'recovered?':>11}")
    for rate in sorted([float(k.split('_')[1]) for k in results]):
        traj = results[f"rate_{rate}"]["trajectory"]
        ho = [(t["step"], t["heldout_acc"]) for t in traj]
        fs, fl = min(ho, key=lambda x: x[1])
        ho_final = ho[-1][1]
        id_final = traj[-1]["indist_acc"]
        rec = "YES" if ho_final > 0.6 else ("partial" if ho_final > 0.45 else "no")
        log(f"  {rate:>6.2f} {fl*100:>8.1f}% {ho_final*100:>8.1f}% {id_final*100:>8.1f}% {rec:>11}")
    log("\n  Prediction: dip floor rises + recovery strengthens with rate.")


if __name__ == "__main__":
    main()

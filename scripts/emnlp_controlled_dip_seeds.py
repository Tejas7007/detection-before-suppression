"""
Controlled Dip — seed replication
==================================
Hardens the controlled-dip result (which was n=1 per condition) by adding
training seeds {43, 44} on the key conditions {0.0, 0.15, 0.30}. Combined
with the existing seed-42 run (emnlp_controlled_dip.json), this gives n=3.

Only TRAINING randomness varies across seeds (model init, WikiText order,
injection draws). The eval sets and the injection-pool CONTENT are held
FIXED (seed 42 / 999) so every seed is measured on the identical eval and
sees the identical pool of IOI sentences — clean seed comparison.

Full 40K steps, Wang-standard eval (held-out names, shared templates),
dual eval (held-out + in-distribution). Order: seed 43 (all conditions)
first so n=2 lands within ~8 hrs; seed 44 completes during writing.

Output: results/emnlp_controlled_dip_seeds.json
"""

import os, gc, json, time, math, random
import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

DEVICE = "cuda"
EVAL_SEED = 42        # fixed eval set
POOL_SEED = 999       # fixed injection-pool content
CTX = 256
D_MODEL = 512
N_LAYER = 8
N_HEAD = 8
BATCH = 32
LR = 3e-4
WARMUP = 1000
TOTAL_STEPS = 40000
EVAL_EVERY = 2000
TRAIN_SEEDS = [43, 44]
CONDITIONS = [0.0, 0.15, 0.30]
RESULTS_PATH = "results/emnlp_controlled_dip_seeds.json"

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


def build_injection_pool(tokenizer, inject_names, n_pool=4000, seed=POOL_SEED):
    rng = random.Random(seed)
    return torch.tensor([make_injected_sequence(tokenizer, inject_names, rng)
                         for _ in range(n_pool)], dtype=torch.long)


def build_eval_set(tokenizer, name_pool, n=300, seed=EVAL_SEED):
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
    log(f"  {len(chunks)} WikiText chunks")
    return torch.tensor(chunks, dtype=torch.long)


def train(rate, train_seed, wikitext, inject_pool, eval_ho, eval_id, vocab_size):
    torch.manual_seed(train_seed); np.random.seed(train_seed)
    cfg = GPT2Config(vocab_size=vocab_size, n_positions=CTX, n_embd=D_MODEL,
                     n_layer=N_LAYER, n_head=N_HEAD)
    model = GPT2LMHeadModel(cfg).to(DEVICE); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))
    def lr_lambda(step):
        if step < WARMUP: return step / WARMUP
        prog = (step - WARMUP) / max(1, TOTAL_STEPS - WARMUP)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    data_rng = np.random.default_rng(train_seed)
    inject_rng = np.random.default_rng(train_seed + 777)
    pool_rng = np.random.default_rng(train_seed + 1234)

    traj = []
    n_chunks, n_pool = wikitext.shape[0], inject_pool.shape[0]
    t0 = time.time()
    for step in range(TOTAL_STEPS + 1):
        if step % EVAL_EVERY == 0:
            ho_acc, ho_ld = evaluate_ioi(model, eval_ho)
            id_acc, id_ld = evaluate_ioi(model, eval_id)
            traj.append({"step": step, "heldout_acc": ho_acc, "heldout_ld": ho_ld,
                         "indist_acc": id_acc, "indist_ld": id_ld})
            log(f"    [r={rate:.2f} s={train_seed}] step {step:>6}: "
                f"heldout={ho_acc*100:5.1f}%(LD{ho_ld:+.2f}) indist={id_acc*100:5.1f}% "
                f"({(time.time()-t0)/60:.1f}m)")
            _save(rate, train_seed, traj)
        if step == TOTAL_STEPS: break
        chunk_idx = data_rng.integers(0, n_chunks, size=BATCH)
        draws = inject_rng.random(BATCH)
        pool_idx = pool_rng.integers(0, n_pool, size=BATCH)
        seq = [inject_pool[pool_idx[b]] if draws[b] < rate else wikitext[chunk_idx[b]]
               for b in range(BATCH)]
        batch = torch.stack(seq).to(DEVICE)
        loss = model(input_ids=batch, labels=batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
    del model, opt; torch.cuda.empty_cache(); gc.collect()


def _save(rate, seed, traj):
    res = {}
    if os.path.exists(RESULTS_PATH):
        try: res = json.load(open(RESULTS_PATH))
        except: pass
    res[f"rate_{rate}_seed_{seed}"] = {"rate": rate, "seed": seed, "trajectory": traj}
    json.dump(res, open(RESULTS_PATH, "w"), indent=2)


def main():
    os.makedirs("results", exist_ok=True)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    vocab = len(tok)
    names = get_single_token_names(tok)
    k = int(len(names) * 0.6)
    inject_names, eval_names = names[:k], names[k:]
    log(f"{len(names)} names; {len(inject_names)} inject / {len(eval_names)} eval (disjoint)")
    eval_ho = build_eval_set(tok, eval_names, seed=EVAL_SEED)
    eval_id = build_eval_set(tok, inject_names, seed=EVAL_SEED + 5)
    pool = build_injection_pool(tok, inject_names)
    log(f"  fixed eval sets + pool ({pool.shape}) built")
    wikitext = load_wikitext_chunks(tok)

    done = {}
    if os.path.exists(RESULTS_PATH):
        try: done = json.load(open(RESULTS_PATH))
        except: pass

    # seed 43 (all conditions) first, then seed 44
    for seed in TRAIN_SEEDS:
        for rate in CONDITIONS:
            key = f"rate_{rate}_seed_{seed}"
            if key in done and len(done[key].get("trajectory", [])) >= (TOTAL_STEPS // EVAL_EVERY):
                log(f"SKIP {key}"); continue
            log("=" * 60); log(f"rate={rate}  seed={seed}"); log("=" * 60)
            train(rate, seed, wikitext, pool, eval_ho, eval_id, vocab)

    # Combined summary (seed 42 from main file + new seeds).
    log("\n" + "=" * 60); log("SEED-REPLICATION SUMMARY"); log("=" * 60)
    main_res = {}
    if os.path.exists("results/emnlp_controlled_dip.json"):
        main_res = json.load(open("results/emnlp_controlled_dip.json"))
    seed_res = json.load(open(RESULTS_PATH))
    for rate in CONDITIONS:
        floors, finals = [], []
        # seed 42
        mk = f"rate_{rate}"
        if mk in main_res:
            t = main_res[mk]["trajectory"]
            floors.append(min(x["heldout_acc"] for x in t)); finals.append(t[-1]["heldout_acc"])
        for seed in TRAIN_SEEDS:
            k2 = f"rate_{rate}_seed_{seed}"
            if k2 in seed_res:
                t = seed_res[k2]["trajectory"]
                floors.append(min(x["heldout_acc"] for x in t)); finals.append(t[-1]["heldout_acc"])
        if floors:
            log(f"  rate={rate:.2f}: n={len(floors)}  floor={np.mean(floors)*100:.1f}"
                f"±{np.std(floors)*100:.1f}%  final={np.mean(finals)*100:.1f}±{np.std(finals)*100:.1f}%")


if __name__ == "__main__":
    main()

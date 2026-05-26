"""
Train GPT-2-style model on OpenWebText (streaming)
====================================================

The WikiText-trained model showed the IOI dip (19% accuracy) but never
recovered. WikiText-103 has only 100M tokens of Wikipedia — not enough
diverse name-tracking text.

OpenWebText has ~8B tokens of web text (forums, blogs, stories) with
far more name-tracking patterns. The model should encounter enough
"When X and Y..., X gave to Y" patterns to develop the S-inhibition
circuit and RECOVER from the dip.

Architecture: 8 layers, d=512, 8 heads (~50M params)
Training: 200K steps, batch=32, ctx=256
Data: OpenWebText streamed (no full download needed)
Checkpoints: every 1K (first 30K), then every 5K

After training, run emnlp_trained_model_ioi.py for IOI analysis.

Output: trained_model_ckpts/openwebtext_seed{seed}/
"""

import os, gc, json, time, math, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

DEVICE = "cuda"
CKPT_DIR = "trained_model_ckpts"
LOG_PATH = "results/emnlp_train_openwebtext_log.txt"

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


class StreamingTokenBuffer:
    """Accumulates tokens from a streaming dataset and yields fixed-length chunks."""

    def __init__(self, dataset_iter, tokenizer, ctx_length):
        self.ds_iter = dataset_iter
        self.tokenizer = tokenizer
        self.ctx = ctx_length
        self.buffer = []
        self.docs_processed = 0
        self.tokens_processed = 0

    def _fill_buffer(self, min_tokens):
        """Fill buffer until it has at least min_tokens."""
        while len(self.buffer) < min_tokens:
            try:
                doc = next(self.ds_iter)
                text = doc.get("text", "")
                if len(text.strip()) < 50:
                    continue
                tokens = self.tokenizer.encode(text)
                # Add EOS between documents.
                self.buffer.extend(tokens)
                self.buffer.append(self.tokenizer.eos_token_id)
                self.docs_processed += 1
            except StopIteration:
                log(f"    Stream exhausted at {self.docs_processed} docs, "
                    f"{self.tokens_processed:,} tokens. Restarting...")
                from datasets import load_dataset
                ds = load_dataset("openwebtext", split="train", streaming=True)
                self.ds_iter = iter(ds.shuffle(seed=self.docs_processed))

    def get_batch(self, batch_size):
        """Get a batch of (input, target) pairs of shape [batch_size, ctx]."""
        needed = batch_size * (self.ctx + 1)
        self._fill_buffer(needed)

        chunks_x, chunks_y = [], []
        for _ in range(batch_size):
            chunk = self.buffer[:self.ctx + 1]
            self.buffer = self.buffer[self.ctx:]  # overlap by 1 for targets
            self.tokens_processed += self.ctx
            chunks_x.append(chunk[:-1])
            chunks_y.append(chunk[1:])

        return (torch.tensor(chunks_x, dtype=torch.long, device=DEVICE),
                torch.tensor(chunks_y, dtype=torch.long, device=DEVICE))


def train(seed=42, n_steps=200_000, batch_size=32, ctx=256,
          n_layers=8, d_model=512, n_heads=8, lr=6e-4):

    log(f"{'='*60}")
    log(f"Training on OpenWebText (seed={seed})")
    log(f"{'='*60}")

    # Model.
    config = GPT2Config(
        vocab_size=50257, n_positions=ctx,
        n_embd=d_model, n_layer=n_layers, n_head=n_heads,
        n_inner=d_model * 4,
        activation_function="gelu_new",
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
    )
    torch.manual_seed(seed)
    model = GPT2LMHeadModel(config).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Model: {n_layers}L, d={d_model}, h={n_heads}, params={n_params/1e6:.1f}M")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Data stream.
    log("  Loading OpenWebText stream...")
    from datasets import load_dataset
    ds = load_dataset("openwebtext", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10000)
    token_buffer = StreamingTokenBuffer(iter(ds), tokenizer, ctx)

    # Optimizer.
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                            betas=(0.9, 0.95))
    warmup = 2000

    # Checkpoint schedule.
    save_early = set(range(0, 30001, 1000))
    save_late = set(range(35000, n_steps + 1, 5000))
    save_steps = save_early | save_late | {n_steps}

    ckpt_base = os.path.join(CKPT_DIR, f"openwebtext_seed{seed}")
    os.makedirs(ckpt_base, exist_ok=True)
    config.save_pretrained(ckpt_base)
    tokenizer.save_pretrained(ckpt_base)

    log(f"  Training for {n_steps:,} steps (batch={batch_size}, ctx={ctx})")
    model.train()
    losses = []
    t0 = time.time()

    for step in range(n_steps + 1):
        # LR schedule: linear warmup + cosine decay.
        if step < warmup:
            current_lr = lr * (step + 1) / warmup
        else:
            progress = (step - warmup) / (n_steps - warmup)
            current_lr = lr * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in opt.param_groups:
            pg["lr"] = current_lr

        # Save checkpoint.
        if step in save_steps:
            ckpt_path = os.path.join(ckpt_base, f"step_{step}")
            if not os.path.exists(ckpt_path):
                model.save_pretrained(ckpt_path)
            avg_loss = np.mean(losses[-200:]) if losses else float("nan")
            elapsed = (time.time() - t0) / 3600
            tokens = token_buffer.tokens_processed
            log(f"  step={step:>6}  loss={avg_loss:.4f}  lr={current_lr:.2e}  "
                f"docs={token_buffer.docs_processed:,}  "
                f"tokens={tokens:,}  elapsed={elapsed:.1f}h  [saved]")

        elif step % 10000 == 0 and step > 0:
            avg_loss = np.mean(losses[-200:])
            elapsed = (time.time() - t0) / 3600
            log(f"  step={step:>6}  loss={avg_loss:.4f}  lr={current_lr:.2e}  "
                f"elapsed={elapsed:.1f}h")

        if step >= n_steps:
            break

        # Training step.
        batch_x, batch_y = token_buffer.get_batch(batch_size)
        out = model(batch_x, labels=batch_y)
        loss = out.loss
        losses.append(float(loss.item()))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    elapsed = (time.time() - t0) / 3600
    log(f"\n  Training complete. {elapsed:.1f} hours, "
        f"{token_buffer.tokens_processed:,} tokens, "
        f"{token_buffer.docs_processed:,} documents")

    del model, opt; torch.cuda.empty_cache(); gc.collect()
    return {
        "seed": seed, "n_params": n_params, "n_steps": n_steps,
        "tokens_processed": token_buffer.tokens_processed,
        "docs_processed": token_buffer.docs_processed,
        "final_loss": float(np.mean(losses[-200:])),
        "elapsed_hours": elapsed,
    }


def main():
    os.makedirs("results", exist_ok=True)

    # Clear log.
    with open(LOG_PATH, "w") as f:
        f.write("")

    results = {}

    # Train seed 42.
    log("Starting OpenWebText training")
    r1 = train(seed=42, n_steps=200_000)
    results["seed42"] = r1
    with open("results/emnlp_train_openwebtext.json", "w") as f:
        json.dump(results, f, indent=2)

    # Train seed 123.
    r2 = train(seed=123, n_steps=200_000)
    results["seed123"] = r2
    with open("results/emnlp_train_openwebtext.json", "w") as f:
        json.dump(results, f, indent=2)

    log("\nAll training complete.")
    log("Run IOI analysis with:")
    log("  python3 scripts/emnlp_trained_model_ioi.py")
    log("(Update CKPT_DIR to 'trained_model_ckpts' and look for openwebtext_seed* dirs)")


if __name__ == "__main__":
    main()

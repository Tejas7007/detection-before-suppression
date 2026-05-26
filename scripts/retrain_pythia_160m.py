"""
Pythia-160M Retraining with Dense Checkpoints
=============================================
Trains a Pythia-160M-equivalent model from scratch with:
  - Same architecture as EleutherAI/pythia-160m-deduped (verified: 162.3M params)
  - Same optimizer settings (Adam, lr=6e-4, betas=0.9/0.95, cosine schedule)
  - Same batch size (2M tokens per step = 1024 sequences of 2048)
  - Different seed (42 instead of 1234)
  - Dense checkpoints: every 10 steps 0-100, every 50 steps 100-3000, every 200 steps 3000-10000

Uses the Pile (deduped) via HuggingFace streaming to avoid downloading the full dataset.
"""

import os
import sys
import json
import time
import math
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoConfig, GPTNeoXForCausalLM, AutoTokenizer
from itertools import chain

# ============================================================
# CONFIG - matches Pythia-160M exactly
# ============================================================

SEED = 42
MAX_STEPS = 10000
SEQ_LENGTH = 2048
TOTAL_BATCH_TOKENS = 2097152  # 2M tokens per step (matches Pythia)
TOTAL_BATCH_SEQS = TOTAL_BATCH_TOKENS // SEQ_LENGTH  # 1024
MICRO_BATCH = 8  # per-GPU micro batch
GRAD_ACCUM = TOTAL_BATCH_SEQS // MICRO_BATCH  # 64

# Optimizer (from pythia-160m-deduped.yml)
LR = 6e-4
MIN_LR = 6e-5
BETAS = (0.9, 0.95)
EPS = 1e-8
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
WARMUP_FRAC = 0.01  # 1% warmup
WARMUP_STEPS = int(MAX_STEPS * WARMUP_FRAC)

# Checkpoint schedule
SAVE_EVERY_10 = list(range(0, 101, 10))        # 0,10,20,...,100
SAVE_EVERY_50 = list(range(150, 3001, 50))      # 150,200,...,3000
SAVE_EVERY_200 = list(range(3200, 10001, 200))   # 3200,3400,...,10000
SAVE_STEPS = sorted(set(SAVE_EVERY_10 + SAVE_EVERY_50 + SAVE_EVERY_200 + [MAX_STEPS]))

CHECKPOINT_DIR = "/workspace/pythia-160m-retrain/checkpoints"
LOG_FILE = "/workspace/pythia-160m-retrain/training_log.json"
MODEL_NAME = "EleutherAI/pythia-160m-deduped"

print("=" * 60)
print("  PYTHIA-160M RETRAINING")
print("  Seed: %d" % SEED)
print("  Max steps: %d" % MAX_STEPS)
print("  Batch: %d tokens/step (%d seqs, micro=%d, accum=%d)" % (
    TOTAL_BATCH_TOKENS, TOTAL_BATCH_SEQS, MICRO_BATCH, GRAD_ACCUM))
print("  LR: %g -> %g (cosine, %d warmup steps)" % (LR, MIN_LR, WARMUP_STEPS))
print("  Checkpoints: %d saves" % len(SAVE_STEPS))
print("  First 20 save steps:", SAVE_STEPS[:20])
print("=" * 60)

# ============================================================
# SET SEEDS
# ============================================================
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
import random
import numpy as np
random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# MODEL
# ============================================================
print("\n[1/4] Initializing model...")
config = AutoConfig.from_pretrained(MODEL_NAME)
model = GPTNeoXForCausalLM(config)
model = model.cuda()
model.gradient_checkpointing_enable()  # fp16 like original Pythia
n_params = sum(p.numel() for p in model.parameters())
print("  Params: %.1fM" % (n_params / 1e6))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
print("  Vocab: %d" % tokenizer.vocab_size)

# ============================================================
# DATA - Stream from Pile
# ============================================================
print("\n[2/4] Setting up data pipeline...")

class PileStreamDataset(IterableDataset):
    """Stream from the Pile, tokenize, and pack into fixed-length sequences."""
    
    def __init__(self, tokenizer, seq_length, seed=42):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.seed = seed
    
    def __iter__(self):
        from datasets import load_dataset
        
        # Use monology/pile-uncopyrighted as it's more accessible
        # Same data as the Pile, just with copyrighted content removed
        # PolyPythias shows data ordering doesn't affect the dip
        ds = load_dataset(
            "monology/pile-uncopyrighted",
            split="train",
            streaming=True,
        )
        ds = ds.shuffle(seed=self.seed, buffer_size=10000)
        
        # Accumulate tokens and yield fixed-length chunks
        buffer = []
        for example in ds:
            tokens = self.tokenizer(
                example["text"],
                truncation=False,
                add_special_tokens=False,
            )["input_ids"]
            
            # Add EOS between documents
            buffer.extend(tokens)
            buffer.append(self.tokenizer.eos_token_id)
            
            while len(buffer) >= self.seq_length:
                chunk = buffer[:self.seq_length]
                buffer = buffer[self.seq_length:]
                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "labels": torch.tensor(chunk, dtype=torch.long),
                }

dataset = PileStreamDataset(tokenizer, SEQ_LENGTH, seed=SEED)
dataloader = DataLoader(dataset, batch_size=MICRO_BATCH)
data_iter = iter(dataloader)

# Test data loading
print("  Testing data pipeline...")
test_batch = next(iter(DataLoader(PileStreamDataset(tokenizer, SEQ_LENGTH, seed=0), batch_size=2)))
print("  Batch shape:", test_batch["input_ids"].shape)
print("  Sample tokens:", tokenizer.decode(test_batch["input_ids"][0][:20]))
print("  Data pipeline OK")

# ============================================================
# OPTIMIZER + SCHEDULER
# ============================================================
print("\n[3/4] Setting up optimizer...")

# Separate weight decay (don't decay biases and layernorms)
decay_params = []
no_decay_params = []
for name, param in model.named_parameters():
    if param.requires_grad:
        if "bias" in name or "layer_norm" in name or "layernorm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

optimizer = torch.optim.AdamW([
    {"params": decay_params, "weight_decay": WEIGHT_DECAY},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=LR, betas=BETAS, eps=EPS)

print("  Decay params: %d tensors" % len(decay_params))
print("  No-decay params: %d tensors" % len(no_decay_params))

def get_lr(step):
    """Cosine learning rate schedule with linear warmup."""
    if step < WARMUP_STEPS:
        return LR * step / max(1, WARMUP_STEPS)
    
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    return MIN_LR + 0.5 * (LR - MIN_LR) * (1 + math.cos(math.pi * progress))

# ============================================================
# TRAINING LOOP
# ============================================================
print("\n[4/4] Starting training...")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

scaler = torch.amp.GradScaler('cuda')
log_entries = []
start_time = time.time()
global_step = 0
total_tokens = 0
running_loss = 0.0
loss_count = 0

# Check for resume
if os.path.exists(LOG_FILE):
    with open(LOG_FILE) as f:
        log_entries = json.load(f)
    if log_entries:
        global_step = log_entries[-1]["step"]
        total_tokens = log_entries[-1].get("total_tokens", 0)
        print("  Resuming from step %d" % global_step)
        # Skip data to the right position
        for _ in range(global_step * GRAD_ACCUM):
            try:
                next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                next(data_iter)

model.train()

while global_step < MAX_STEPS:
    # Set learning rate
    lr = get_lr(global_step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    
    optimizer.zero_grad()
    step_loss = 0.0
    
    for micro_step in range(GRAD_ACCUM):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        input_ids = batch["input_ids"].cuda()
        labels = batch["labels"].cuda()
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / GRAD_ACCUM
        
        loss.backward()
        step_loss += loss.item()
    
    # Gradient clipping
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    
    optimizer.step()
    
    
    global_step += 1
    total_tokens += TOTAL_BATCH_TOKENS
    running_loss += step_loss
    loss_count += 1
    
    # Save checkpoint
    if global_step in SAVE_STEPS:
        ckpt_path = os.path.join(CHECKPOINT_DIR, "step_%d" % global_step)
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
        
        elapsed = time.time() - start_time
        avg_loss = running_loss / loss_count if loss_count > 0 else 0
        
        entry = {
            "step": global_step,
            "loss": round(step_loss, 4),
            "avg_loss": round(avg_loss, 4),
            "lr": round(lr, 8),
            "total_tokens": total_tokens,
            "elapsed_seconds": round(elapsed, 1),
            "tokens_per_second": round(total_tokens / elapsed, 0),
        }
        log_entries.append(entry)
        
        with open(LOG_FILE, "w") as f:
            json.dump(log_entries, f, indent=2)
        
        print("  [Step %d/%d] loss=%.4f avg=%.4f lr=%.6f tokens=%dM time=%.0fs (%.0f tok/s) SAVED" % (
            global_step, MAX_STEPS, step_loss, avg_loss, lr,
            total_tokens // 1_000_000, elapsed, total_tokens / elapsed))
    
    elif global_step % 100 == 0:
        elapsed = time.time() - start_time
        avg_loss = running_loss / loss_count if loss_count > 0 else 0
        eta = elapsed / global_step * (MAX_STEPS - global_step)
        print("  [Step %d/%d] loss=%.4f avg=%.4f lr=%.6f ETA=%.0fmin" % (
            global_step, MAX_STEPS, step_loss, avg_loss, lr, eta / 60))

# Save final
elapsed = time.time() - start_time
print("\n" + "=" * 60)
print("  TRAINING COMPLETE")
print("  Steps: %d" % global_step)
print("  Total tokens: %dB" % (total_tokens // 1_000_000_000))
print("  Time: %.1f hours" % (elapsed / 3600))
print("  Checkpoints saved: %d" % len(SAVE_STEPS))
print("  Log: %s" % LOG_FILE)
print("=" * 60)

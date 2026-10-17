import hashlib
import math
import os
import time

import torch
from datasets import load_dataset
from tokenizers import Tokenizer

import config
from model.gpt import GPT

# ---------------------------------------------------------------
# Run settings
# ---------------------------------------------------------------
DATASET = "noanabeshima/TinyStoriesV2"   # cleaned, GPT-4-only version
NUM_TRAIN_STORIES = 1_000_000
NUM_VAL_STORIES = 5_000
MAX_STEPS = 20_000
WARMUP_STEPS = 500
MIN_LR_RATIO = 0.1
WEIGHT_DECAY = 0.1
EVAL_EVERY = 500
EVAL_BATCHES = 50
LOG_EVERY = 100
USE_COMPILE = False        # try True later; first compile takes a few minutes
SAMPLE_PROMPT = "Once upon a time"

# Everything saved here survives Colab disconnects (Google Drive)
SAVE_DIR = "/content/drive/MyDrive/gpt_run"
os.makedirs(SAVE_DIR, exist_ok=True)


# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------
def tokenizer_id():
    with open(config.TOKENIZER_PATH, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:8]


def build_stream(split, n, tokenizer, eos_id):
    # cache name includes a tokenizer hash, so a new tokenizer never reuses old IDs
    name = DATASET.split("/")[-1]
    cache = os.path.join(SAVE_DIR, f"cache_{name}_{split}_{n}_{tokenizer_id()}.pt")
    if os.path.exists(cache):
        return torch.load(cache)

    print(f"Tokenizing {n} stories from '{split}' (one time only)...")
    ds = load_dataset(DATASET, split=f"{split}[:{n}]")
    chunks = []
    for i in range(0, len(ds), 1000):
        texts = ds[i:i + 1000]["text"]
        ids = []
        for enc in tokenizer.encode_batch(texts):
            ids.extend(enc.ids)
            ids.append(eos_id)
        chunks.append(torch.tensor(ids, dtype=torch.int16))   # vocab < 32768, saves RAM
    data = torch.cat(chunks)
    torch.save(data, cache)
    return data


def get_batch(data, batch_size, seq_len):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix]).long()
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix]).long()
    return x, y


def get_lr(step):
    peak = config.LEARNING_RATE
    if step < WARMUP_STEPS:
        return peak * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return peak * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * cosine)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    assert torch.cuda.is_available(), (
        "No GPU found. In Colab: Runtime > Change runtime type > T4 GPU"
    )
    device = torch.device("cuda")
    print(f"Training on: {torch.cuda.get_device_name(0)}")

    # bf16 needs Ampere or newer; the free T4 uses fp16 with a grad scaler
    use_bf16 = torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)
    print(f"Mixed precision: {'bf16' if use_bf16 else 'fp16 + GradScaler'}")

    tokenizer = Tokenizer.from_file(config.TOKENIZER_PATH)
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    assert tokenizer.get_vocab_size() == config.VOCAB_SIZE, (
        f"Tokenizer vocab {tokenizer.get_vocab_size()} != config.VOCAB_SIZE {config.VOCAB_SIZE}"
    )

    train_data = build_stream("train", NUM_TRAIN_STORIES, tokenizer, eos_id)
    val_data = build_stream("validation", NUM_VAL_STORIES, tokenizer, eos_id)
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

    model = GPT(
        vocab_size=config.VOCAB_SIZE,
        embed_size=config.EMBED_SIZE,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        max_seq_len=config.MAX_SEQ_LEN,
        dropout=config.DROPOUT,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Expected starting loss: about {math.log(config.VOCAB_SIZE):.2f}")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=config.LEARNING_RATE,
        betas=(0.9, 0.95),
    )

    # Resume if a checkpoint exists (just re-run the script after a disconnect)
    ckpt_path = os.path.join(SAVE_DIR, "ckpt.pt")
    start_step, best_val = 0, float("inf")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start_step = ck["step"] + 1
        best_val = ck["best_val"]
        print(f"Resumed from step {start_step} (best val {best_val:.4f})")

    fwd = torch.compile(model) if USE_COMPILE else model

    def autocast():
        return torch.autocast(device_type="cuda", dtype=amp_dtype)

    @torch.no_grad()
    def estimate_loss():
        model.eval()
        losses = []
        for _ in range(EVAL_BATCHES):
            x, y = get_batch(val_data, config.BATCH_SIZE, config.MAX_SEQ_LEN)
            with autocast():
                _, loss = fwd(x.to(device), targets=y.to(device))
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    def sample():
        model.eval()
        prompt = torch.tensor([tokenizer.encode(SAMPLE_PROMPT).ids], device=device)
        with autocast():
            out = model.generate(prompt, max_new_tokens=100, temperature=0.8, top_k=40)
        model.train()
        return tokenizer.decode(out[0].tolist())

    model.train()
    start = time.time()

    for step in range(start_step, MAX_STEPS):
        lr = get_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, config.BATCH_SIZE, config.MAX_SEQ_LEN)
        x, y = x.to(device), y.to(device)

        with autocast():
            _, loss = fwd(x, targets=y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % LOG_EVERY == 0:
            print(f"step {step:6d}/{MAX_STEPS} | loss {loss.item():.4f} | "
                  f"lr {lr:.2e} | {time.time() - start:.0f}s")

        if (step + 1) % EVAL_EVERY == 0 or step == MAX_STEPS - 1:
            val_loss = estimate_loss()
            print(f"  >>> val loss {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                # plain weights only: works with generate.py on your laptop
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, "gpt_best.pt"))
                print("  >>> saved gpt_best.pt")

            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "best_val": best_val,
            }, ckpt_path)

            print("  >>> sample:", sample().replace("\n", " "))

    print("Done.")


if __name__ == "__main__":
    main()
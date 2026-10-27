import math
import os
import shutil
import time

import numpy as np
import torch
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

import config
from model.gpt import GPT

# ---------------------------------------------------------------
# Data: general-purpose mix from HuggingFaceTB/smollm-corpus
#   FineWeb-Edu (educational web pages) + Cosmopedia v2 (synthetic
#   textbooks, blog posts and stories), roughly 5:1 by tokens.
# ---------------------------------------------------------------
CORPUS = "HuggingFaceTB/smollm-corpus"
SOURCES = [
    {"tag": "fineweb_edu", "subset": "fineweb-edu-dedup", "train_tokens": 580_000_000, "tok_docs": 40_000},
    {"tag": "cosmopedia", "subset": "cosmopedia-v2", "train_tokens": 120_000_000, "tok_docs": 15_000},
]
VAL_TOKENS = 1_500_000      # held out per source (first documents of each stream)

# ---------------------------------------------------------------
# Run settings
# ---------------------------------------------------------------
MAX_STEPS = 42_000    # about one pass over ~700M tokens at 16,384 tokens per step
USE_COMPILE = False   # torch.compile can speed up training but takes minutes to start
WARMUP_STEPS = 1_000
MIN_LR_RATIO = 0.1
WEIGHT_DECAY = 0.1
EVAL_EVERY = 2_000          # validation + checkpoint (checkpoint is what lets you resume)
EVAL_BATCHES = 20           # per validation set
LOG_EVERY = 200

# Everything saved here survives Colab disconnects (Google Drive)
SAVE_DIR = "/content/drive/MyDrive/gpt_run_50m_general"
LOCAL_DIR = "/content/data"   # fast local copy of the token files
os.makedirs(SAVE_DIR, exist_ok=True)


# ---------------------------------------------------------------
# Tokenizer + data preparation (each runs once, then is cached)
# ---------------------------------------------------------------
def stream_docs(subset):
    ds = load_dataset(CORPUS, subset, split="train", streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=10_000)
    for row in ds:
        yield row["text"]


def get_tokenizer():
    path = os.path.join(SAVE_DIR, "tokenizer.json")
    if os.path.exists(path):
        return Tokenizer.from_file(path)

    print(f"Training a {config.VOCAB_SIZE}-token tokenizer (one time only)...")

    def texts():
        for src in SOURCES:
            for i, text in enumerate(stream_docs(src["subset"])):
                if i >= src["tok_docs"]:
                    break
                yield text

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=config.VOCAB_SIZE,
        min_frequency=2,
        special_tokens=["[UNK]", "<|endoftext|>", "<|user|>", "<|assistant|>"],   # last two reserved for chat fine-tuning later
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(texts(), trainer=trainer)
    tok.save(path)
    return tok


def build_data(tokenizer, eos_id):
    train_path = os.path.join(SAVE_DIR, "train.bin")
    val_paths = {s["tag"]: os.path.join(SAVE_DIR, f"val_{s['tag']}.bin") for s in SOURCES}
    if os.path.exists(train_path) and all(os.path.exists(p) for p in val_paths.values()):
        return train_path, val_paths

    train_chunks = []
    for src in SOURCES:
        print(f"Tokenizing {src['subset']} (one time only)...")
        val_chunks, v, t, batch = [], 0, 0, []
        next_report, t0 = 50_000_000, time.time()
        for text in stream_docs(src["subset"]):
            batch.append(text)
            if len(batch) < 500:
                continue
            for enc in tokenizer.encode_batch(batch):
                arr = np.array(enc.ids + [eos_id], dtype=np.uint16)
                if v < VAL_TOKENS:
                    val_chunks.append(arr)
                    v += len(arr)
                else:
                    train_chunks.append(arr)
                    t += len(arr)
            batch = []
            if t >= next_report:
                print(f"  {src['tag']}: {t:,} tokens ({time.time() - t0:.0f}s)")
                next_report += 50_000_000
            if t >= src["train_tokens"]:
                break
        np.concatenate(val_chunks).tofile(val_paths[src["tag"]])
        print(f"  {src['tag']}: {t:,} train tokens, {v:,} val tokens")

    np.concatenate(train_chunks).tofile(train_path)
    return train_path, val_paths


def open_tokens(path):
    # copy from Drive to local disk once; reading from Drive during training is slow
    os.makedirs(LOCAL_DIR, exist_ok=True)
    dst = os.path.join(LOCAL_DIR, os.path.basename(path))
    if not os.path.exists(dst):
        shutil.copy(path, dst)
    return np.memmap(dst, dtype=np.uint16, mode="r")


def get_batch(data, batch_size, seq_len):
    ix = np.random.randint(0, len(data) - seq_len - 1, size=batch_size)
    x = np.stack([data[i:i + seq_len] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + seq_len + 1] for i in ix]).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


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

    tokenizer = get_tokenizer()
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    assert tokenizer.get_vocab_size() == config.VOCAB_SIZE, (
        f"Tokenizer vocab {tokenizer.get_vocab_size()} != config.VOCAB_SIZE {config.VOCAB_SIZE}"
    )

    train_path, val_paths = build_data(tokenizer, eos_id)
    train_data = open_tokens(train_path)
    val_sets = [(tag, open_tokens(p)) for tag, p in val_paths.items()]
    print(f"Train tokens: {len(train_data):,}")
    for tag, v in val_sets:
        print(f"  val {tag}: {len(v):,} tokens")

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
    def estimate_loss(data):
        model.eval()
        losses = []
        for _ in range(EVAL_BATCHES):
            x, y = get_batch(data, config.BATCH_SIZE, config.MAX_SEQ_LEN)
            with autocast():
                _, loss = fwd(x.to(device), targets=y.to(device))
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

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
            per_set = {tag: estimate_loss(v) for tag, v in val_sets}
            val_loss = sum(per_set.values()) / len(per_set)
            detail = " | ".join(f"{t} {v:.4f}" for t, v in per_set.items())
            print(f"  >>> val loss {detail} | mean {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                # plain weights only: works with generate.py on your laptop
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, "gpt_best.pt"))

            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "best_val": best_val,
            }, ckpt_path)

    print(f"Done. Files are in {SAVE_DIR} (gpt_best.pt and tokenizer.json)")


if __name__ == "__main__":
    main()
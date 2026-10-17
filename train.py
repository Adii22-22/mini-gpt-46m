import math
import os
import time

import torch
import torch_directml
from datasets import load_dataset
from tokenizers import Tokenizer

import config
from model.gpt import GPT

# ---------------------------------------------------------------
# Run settings (kept here so config.py doesn't need to change)
# ---------------------------------------------------------------
SMOKE_TEST = False          # True = 20-step check. Set False for the real run.

NUM_TRAIN_STORIES = 50_000
NUM_VAL_STORIES = 1_000
MAX_STEPS = 4_000
WARMUP_STEPS = 200
MIN_LR_RATIO = 0.1         # LR decays down to 10% of peak
WEIGHT_DECAY = 0.1
EVAL_EVERY = 250
EVAL_BATCHES = 20
LOG_EVERY = 50
SAMPLE_PROMPT = "Once upon a time"

if SMOKE_TEST:
    NUM_TRAIN_STORIES, NUM_VAL_STORIES = 500, 100
    MAX_STEPS, WARMUP_STEPS = 20, 5
    EVAL_EVERY, EVAL_BATCHES, LOG_EVERY = 10, 2, 1


# ---------------------------------------------------------------
# Data: tokenize once, pack everything into one long token stream
# ---------------------------------------------------------------
def build_stream(split, n, tokenizer, eos_id):
    cache = f"cache_{split}_{n}.pt"
    if os.path.exists(cache):
        return torch.load(cache)

    print(f"Tokenizing {n} stories from '{split}' (one time only)...")
    ds = load_dataset("roneneldan/TinyStories", split=f"{split}[:{n}]")
    chunks = []   # one small tensor per 1000 stories (keeps RAM low)
    for i in range(0, len(ds), 1000):
        texts = ds[i:i + 1000]["text"]
        ids = []
        for enc in tokenizer.encode_batch(texts):
            ids.extend(enc.ids)
            ids.append(eos_id)          # marks the end of each story
        chunks.append(torch.tensor(ids, dtype=torch.long))
    data = torch.cat(chunks)
    torch.save(data, cache)
    return data


def get_batch(data, batch_size, seq_len):
    # random windows; y is x shifted by one token
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
    return x, y


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def get_lr(step):
    peak = config.LEARNING_RATE
    if step < WARMUP_STEPS:
        return peak * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return peak * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * cosine)


@torch.no_grad()
def estimate_loss(model, data, device):
    model.eval()
    losses = []
    for _ in range(EVAL_BATCHES):
        x, y = get_batch(data, config.BATCH_SIZE, config.MAX_SEQ_LEN)
        _, loss = model(x.to(device), targets=y.to(device))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def make_model():
    return GPT(
        vocab_size=config.VOCAB_SIZE,
        embed_size=config.EMBED_SIZE,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        max_seq_len=config.MAX_SEQ_LEN,
        dropout=config.DROPOUT,
    )


def cpu_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def sample_story(model, tokenizer):
    # Generate on CPU: topk/multinomial can be flaky on DirectML
    cpu_model = make_model()
    cpu_model.load_state_dict(cpu_state_dict(model))
    cpu_model.eval()
    prompt = torch.tensor([tokenizer.encode(SAMPLE_PROMPT).ids])
    out = cpu_model.generate(prompt, max_new_tokens=60, temperature=0.8, top_k=40)
    return tokenizer.decode(out[0].tolist())


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    device = torch_directml.device()
    print(f"Training on: {torch_directml.device_name(0)}")
    if SMOKE_TEST:
        print(">>> SMOKE TEST MODE (20 steps). Set SMOKE_TEST = False for the real run.")

    tokenizer = Tokenizer.from_file(config.TOKENIZER_PATH)
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_id is None:
        eos_id = 0
    assert tokenizer.get_vocab_size() <= config.VOCAB_SIZE, (
        f"Tokenizer vocab ({tokenizer.get_vocab_size()}) is larger than "
        f"config.VOCAB_SIZE ({config.VOCAB_SIZE})"
    )

    train_data = build_stream("train", NUM_TRAIN_STORIES, tokenizer, eos_id)
    val_data = build_stream("validation", NUM_VAL_STORIES, tokenizer, eos_id)
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

    model = make_model().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Expected starting loss: about {math.log(config.VOCAB_SIZE):.2f}")

    # Weight decay on matrices only (not LayerNorm / bias)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=config.LEARNING_RATE,
        betas=(0.9, 0.95),
    )

    best_val = float("inf")
    model.train()
    start = time.time()

    for step in range(MAX_STEPS):
        lr = get_lr(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_data, config.BATCH_SIZE, config.MAX_SEQ_LEN)
        x, y = x.to(device), y.to(device)

        _, loss = model(x, targets=y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % LOG_EVERY == 0:
            elapsed = time.time() - start
            print(f"step {step:5d}/{MAX_STEPS} | loss {loss.item():.4f} | "
                  f"lr {lr:.2e} | {elapsed:.0f}s")

        if (step + 1) % EVAL_EVERY == 0 or step == MAX_STEPS - 1:
            val_loss = estimate_loss(model, val_data, device)
            print(f"  >>> val loss {val_loss:.4f}")

            torch.save(cpu_state_dict(model), "gpt_last.pt")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(cpu_state_dict(model), "gpt_best.pt")
                print("  >>> saved gpt_best.pt")

            print("  >>> sample:", sample_story(model, tokenizer).replace("\n", " "))

    print("Done.")


if __name__ == "__main__":
    main()
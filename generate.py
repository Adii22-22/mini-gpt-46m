import argparse

import torch
from tokenizers import Tokenizer

import config
from model.gpt import GPT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="gpt_best.pt")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    tokenizer = Tokenizer.from_file(config.TOKENIZER_PATH)
    assert tokenizer.get_vocab_size() == config.VOCAB_SIZE, ("Tokenizer vocabulary does not match config.VOCAB_SIZE")
    eos_id = tokenizer.token_to_id("<|endoftext|>")

    # CPU is fine for generation and avoids DirectML issues
    model = GPT(
        vocab_size=config.VOCAB_SIZE,
        embed_size=config.EMBED_SIZE,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        max_seq_len=config.MAX_SEQ_LEN,
        dropout=config.DROPOUT,
    )
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu",weights_only=True))
    model.eval()
    print(f"Loaded {args.ckpt} | temp={args.temperature} top_k={args.top_k}\n")

    prompt_ids = tokenizer.encode(args.prompt).ids

    for n in range(args.samples):
        idx = torch.tensor([prompt_ids])
        out = model.generate(
            idx,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )[0].tolist()

        # Stop at the end-of-story token if the model produced one
        if eos_id in out[len(prompt_ids):]:
            out = out[: len(prompt_ids) + out[len(prompt_ids):].index(eos_id)]

        print(f"--- Sample {n + 1} ---")
        print(tokenizer.decode(out))
        print()


if __name__ == "__main__":
    main()
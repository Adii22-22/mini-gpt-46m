import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.embedding import GPTEmbedding
from model.transformer_block import TransformerBlock


class GPT(nn.Module):
    def __init__(self, vocab_size=350, embed_size=256, num_heads=4,
                 num_layers=4, max_seq_len=128, dropout=0.1):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        # 1. Embeddings (tokens + positions)
        self.embeddings = GPTEmbedding(vocab_size, embed_size, max_seq_len)

        # 2. Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # 3. Final layer norm
        self.ln_f = nn.LayerNorm(embed_size)

        # 4. Output head (embed_size -> vocab_size)
        self.lm_head = nn.Linear(embed_size, vocab_size, bias=False)

        # Initialize weights first, then tie (so both share one tensor)
        self.apply(self._init_weights)

        # Weight tying: output layer shares weights with token embedding
        self.lm_head.weight = self.embeddings.token_emb.weight

        # Scaled init for residual output projections (GPT-2 trick)
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """
        idx:     [batch_size, seq_len] token ids
        targets: [batch_size, seq_len] next-token ids (optional)
        """
        B, T = idx.shape
        assert T <= self.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"
        )

        x = self.embeddings(idx)
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressively generate tokens. Call model.eval() first
        so dropout is turned off.
        """
        if temperature <= 0:
            raise ValueError("temperature must be greater than 0")

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# --- Sanity test ---
if __name__ == "__main__":
    print("\n--- Testing the Full GPT Model ---")
    VOCAB_SIZE = 350
    EMBED_SIZE = 256
    NUM_HEADS = 4
    NUM_LAYERS = 4
    MAX_SEQ_LEN = 128
    BATCH_SIZE = 2
    SEQ_LEN = 8

    model = GPT(
        vocab_size=VOCAB_SIZE,
        embed_size=EMBED_SIZE,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
    )

    # parameters() counts tied weights only once
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    dummy_input = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    dummy_targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

    logits, loss = model(dummy_input, targets=dummy_targets)
    print(f"Logits shape: {logits.shape} -> expected [{BATCH_SIZE}, {SEQ_LEN}, {VOCAB_SIZE}]")
    print(f"Loss: {loss.item():.4f} -> expected about {math.log(VOCAB_SIZE):.2f}")

    # Gradient test: gradients must reach tied embeddings and first attention layer
    loss.backward()
    assert model.embeddings.token_emb.weight.grad is not None
    assert model.blocks[0].attn.c_attn.weight.grad is not None
    print("Gradient test passed!")

    model.eval()
    prompt = torch.tensor([[1, 2, 3]])
    generated = model.generate(prompt, max_new_tokens=5, temperature=0.8, top_k=40)
    print(f"Generated shape: {generated.shape} -> expected [1, 8]")
    print("Success! Full GPT assembled and verified.")
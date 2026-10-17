import torch 
import torch.nn as nn

from model.attention import MultiHeadAttention
from model.feed_forward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self,embed_size=256, num_heads=4, dropout=0.1):
        super().__init__()

        # 1. first normalization and attention
        self.ln_1 = nn.LayerNorm(embed_size)
        self.attn = MultiHeadAttention(embed_size,num_heads)

        # 2. second normalization and feed forward
        self.ln_2 = nn.LayerNorm(embed_size)
        self.ffn = FeedForward(embed_size,dropout)

    def forward(self,x):
        # 1. Attention phase with Residual Connection (x + ...)
        # Notice the Pre-Norm: we normalize x BEFORE sending it to attention
        x = x + self.attn(self.ln_1(x))

        # 2. Feed-Forward phase with Residual Connection (x + ...)
        # Again, Pre-Norm: we normalize x BEFORE sending it to the FFN
        x = x + self.ffn(self.ln_2(x))

        return x

# --- Sanity Test Block ---
if __name__ == "__main__":
    print("\n--- Testing the Complete Transformer Block ---")
    BATCH_SIZE = 2
    SEQ_LEN = 10
    EMBED_SIZE = 256
    
    # Initialize the block
    block = TransformerBlock(embed_size=EMBED_SIZE)
    
    # Dummy input
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_SIZE)
    print(f"Input shape to Block:  {dummy_input.shape}")
    
    # Pass it through the entire block
    output = block(dummy_input)
    print(f"Output shape from Block: {output.shape}")
    
    assert dummy_input.shape == output.shape, "Shapes must match!"
    print("Success! The Transformer Block is fully wired and functional.")
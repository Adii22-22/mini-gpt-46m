import torch 
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, embed_size=256,dropout=0.1):
        super().__init__()

        hidden_dim = 4 * embed_size

        self.net = nn.Sequential(
            nn.Linear(embed_size,hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,embed_size),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        return self.net(x)

# --- Sanity Test Block ---
if __name__ == "__main__":
    print("\n--- Testing the Feed-Forward Layer ---")
    BATCH_SIZE = 2
    SEQ_LEN = 10
    EMBED_SIZE = 256
    
    ffn = FeedForward(embed_size=EMBED_SIZE)
    
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_SIZE)
    print(f"Input shape:  {dummy_input.shape}")
    
    output = ffn(dummy_input)
    print(f"Output shape: {output.shape}")
    
    assert dummy_input.shape == output.shape, "Input and output shapes must match!"
    print("Success! Feed-Forward layer processed the tokens cleanly.")
import torch 
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size=256, num_heads=4):
        super().__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads

        self.head_dim = embed_size // num_heads
        assert self.head_dim * num_heads == embed_size, "embed_size must be divisible by num_heads"

        #also c refers to Conv1D as of one Dimentional Convolutional
        # read c_attn as the attention input layer
        self.c_attn = nn.Linear(embed_size,3 * embed_size,bias=False)

        #read c_proj as the output projection layer
        self.c_proj = nn.Linear(embed_size,embed_size,bias=False)


    def forward(self,x):

        # B = batch size , T = sequence length , C=Embedding size
        B,T,C = x.size()

        # 1. Pass the input as a single layer 
        qkv = self.c_attn(x)

        # 2. Split the massive output into 3 equal chunks
        q,k,v = qkv.split(self.embed_size,dim=2)

        # 3. REshape the matrices to isolate individual heads
        # From: [B, T, C] -> [B, T, num_heads, head_dim]
        # Then transpose to: [B, num_heads, T, head_dim]
        k = k.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        q = q.view(B,T,self.num_heads,self.head_dim).transpose(1,2)
        v = v.view(B,T,self.num_heads,self.head_dim).transpose(1,2)

        # OPTIMIZATION 2: FlashAttention via PyTorch 2.0
        # is_causal=True automatically applies the "mask" so words cannot look into the future.
        # This handles the matrix multiplication, scaling, masking, and softmax internally.
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        out = self.c_proj(y)
        return out 

if __name__ == "__main__":
    print("\n--- Testing the Multi-Head Attention Layer ---")
    BATCH_SIZE = 2
    SEQ_LEN = 10
    EMBED_SIZE = 256
    NUM_HEADS = 4
    
    # Initialize our optimized layer
    mha = MultiHeadAttention(embed_size=EMBED_SIZE, num_heads=NUM_HEADS)
    
    # Create a dummy tensor representing the output from your Embeddings layer
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_SIZE)
    print(f"Input shape (from embeddings): {dummy_input.shape}")
    
    # Pass it through Attention
    output = mha(dummy_input)
    print(f"Output shape (from attention): {output.shape}")
    
    assert dummy_input.shape == output.shape, "Input and output shapes must match!"
    print("Success! The attention mechanism processed the sequence perfectly.")
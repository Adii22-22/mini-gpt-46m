import torch
import torch.nn as nn

class GPTEmbedding(nn.Module):
    def __init__(self,vocab_size,embed_size,max_seq_len):
        """
        vocab_size = how many unique token we have (example 350)
        embed_size = total count of number use to describe a single token (Example "python" with embed_size 3 = [0.3,-1.0,4.0]) 
        max_seq_len = maximum number of word the model can look at ones (ex: 128)
        """
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size,embed_size)
        self.postional_emb = nn.Embedding(max_seq_len,embed_size)

    def forward(self,input_ids):
        """
        input_ids shape = [batch_size x  seq_len]
        """
        batch_size,seq_len = input_ids.shape

        # Create a sequence of position integers: [0, 1, 2, ..., seq_len - 1]
        # We ensure it sits on the same device (CPU/GPU) as the input text

        positions = torch.arange(0,seq_len,device=input_ids.device)

        token_vectors = self.token_emb(input_ids)
        positional_vectors = self.postional_emb(positions)

        final_embeddings = token_vectors + positional_vectors

        return final_embeddings

if __name__ == "__main__":
    VOCAB_SIZE = 350
    EMBED_SIZE = 64
    MAX_SEQ_LEN = 128

    embedding = GPTEmbedding(VOCAB_SIZE, EMBED_SIZE, MAX_SEQ_LEN)

    # Simulate a batch of 2 sequences, each containing 10 tokens
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 10))

    output = embedding(input_ids)

    print("Input shape:", input_ids.shape)
    print("Output shape:", output.shape)

    assert output.shape == (2, 10, EMBED_SIZE)
    print("Embedding test passed!")
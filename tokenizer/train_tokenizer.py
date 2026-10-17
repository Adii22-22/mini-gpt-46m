import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from datasets import load_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.json")

VOCAB_SIZE = 4096          # must match config.VOCAB_SIZE
NUM_STORIES = 100_000      # more text = better merges than 1000 stories
DATASET = "noanabeshima/TinyStoriesV2"   # cleaned, GPT-4-only version

dataset = load_dataset(
    DATASET,
    split=f"train[:{NUM_STORIES}]"
)


def text_batches(batch_size=1000):
    # Feed text in chunks so we don't hold everything in one big list
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]["text"]


# Initialize BPE
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=["[UNK]", "<|endoftext|>"],
    # Always include all 256 byte symbols so any text can be encoded
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    show_progress=True,
)

tokenizer.train_from_iterator(text_batches(), trainer=trainer, length=len(dataset))

tokenizer.save(TOKENIZER_PATH)

print(f"Tokenizer saved: {TOKENIZER_PATH}")
print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
print(f"<|endoftext|> id: {tokenizer.token_to_id('<|endoftext|>')}")
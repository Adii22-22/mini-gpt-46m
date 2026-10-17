import torch 
from torch.utils.data import Dataset
from datasets import load_dataset

class StoryDataset(Dataset):
    def __init__(self,split_size="train[:10000]"):
        """

        """
        print(f"loading TinyStories ({split_size})...")

        self.raw_data = load_dataset("roneneldan/TinyStories", split=split_size)
        print(f"Loaded {len(self.raw_data)} stories")

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        text  = self.raw_data[idx]["text"]
        return text


if __name__ == "__main__":
    dataset = StoryDataset(split_size="train[:100]")
    print(dataset[0][:300] + "...")        
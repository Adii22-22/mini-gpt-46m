# Mini GPT (~46.5M)

A small GPT-style language model (46,455,808 parameters) written from scratch in PyTorch and trained on a general-purpose text mix: educational web pages plus synthetic textbooks, blog posts and stories. Give it a prompt and it continues the text.

This is a learning project. Every piece (tokenizer, transformer, training loop, text generation) is written by hand so it is easy to read and modify.

## Features

- Decoder-only transformer in the GPT-2 style: pre-LayerNorm, GELU, learned position embeddings, tied input/output embeddings
- Fast causal attention through PyTorch `scaled_dot_product_attention`
- Custom byte-level BPE tokenizer with a 16,384-token vocabulary
- Training loop with warmup + cosine learning-rate decay, AdamW, gradient clipping and mixed precision (bf16 or fp16)
- Resumable training: re-run the script after a Colab disconnect and it continues from the last checkpoint
- Text generation with temperature and top-k sampling

## Model

| Setting | Value |
|---|---|
| Parameters | 46,455,808 |
| Layers | 12 |
| Attention heads | 8 |
| Embedding size | 512 |
| Feed-forward size | 2,048 (4x embedding) |
| Context length | 512 tokens |
| Vocabulary size | 16,384 |
| Dropout | 0.0 |

All settings live in [`config.py`](config.py).

## Training data

Trained on the [SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus), mixed roughly 5:1 by tokens:

| Source | Subset | Training tokens | What it is |
|---|---|---|---|
| FineWeb-Edu | `fineweb-edu-dedup` | 580M | Educational web pages |
| Cosmopedia v2 | `cosmopedia-v2` | 120M | Synthetic textbooks, blog posts and stories |

That is about 700M training tokens. A further 1.5M tokens per source are held out for validation.

The tokenizer is trained on a sample of the same data (40,000 FineWeb-Edu and 15,000 Cosmopedia documents). Besides `<|endoftext|>` it reserves `<|user|>` and `<|assistant|>` tokens for possible chat fine-tuning later.

## Training setup

- Steps: 42,000, with a batch of 32 sequences x 512 tokens (about 688M tokens, roughly one pass over the data)
- Optimizer: AdamW (betas 0.9 / 0.95, weight decay 0.1)
- Learning rate: peak 6e-4, 1,000 warmup steps, cosine decay to 10% of peak
- Gradient clipping at 1.0
- Validation every 2,000 steps on both held-out sets
- Hardware: **ADD YOUR GPU AND TRAINING TIME**

The script saves `gpt_best.pt` (lowest validation loss), `tokenizer.json` and a resumable `ckpt.pt`.

## Project structure

```
.
├── model/
│   ├── gpt.py                # full GPT model + generate()
│   ├── attention.py          # multi-head causal attention
│   ├── feed_forward.py       # MLP block
│   ├── transformer_block.py  # attention + MLP with residuals
│   └── embedding.py          # token + position embeddings
├── tokenizer/
│   └── tokenizer.json        # trained tokenizer
├── config.py                 # model + training hyperparameters
├── train.py                  # training script (CUDA GPU, built for Google Colab)
├── generate.py               # generate text from a checkpoint
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/Adii22-22/mini-gpt-46m
cd mini-gpt-46m

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
```

## Usage

### 1. Train (Google Colab or any CUDA GPU)

In Colab, choose a GPU runtime, mount Google Drive, then run the script:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
python train.py
```

Checkpoints are saved to the folder set by `SAVE_DIR` in `train.py`. If Colab disconnects, run the script again and it resumes automatically.

### 2. Generate text

```bash
python generate.py --prompt "The Photosynthesis is the process by which green plants"
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--ckpt` | `gpt_best.pt` | checkpoint file to load |
| `--tokenizer` | `tokenizer/tokenizer.json` | tokenizer file |
| `--prompt` | `Once upon a time` | starting text |
| `--tokens` | 200 | max new tokens to generate |
| `--temperature` | 0.8 | higher = more random |
| `--top_k` | 40 | sample only from the top-k likely tokens |
| `--samples` | 3 | number of outputs to print |
| `--seed` | none | set for repeatable output |

Example:

```bash
python generate.py --prompt "Once upon a time" --temperature 0.7 --samples 2 --seed 42
```

## Pretrained weights

The weight file (`gpt_best.pt`, roughly 180 MB) is too large for a normal GitHub commit, so it is hosted separately:

- Download: **Soon will be updated** 
- Place `gpt_best.pt` in the project root, then run `generate.py`.

The tokenizer in `tokenizer/` must be the one the weights were trained with.

## Results

- Best validation loss: **fineweb_edu 3.4969 | cosmopedia 2.6521 | mean 3.0745**
- Sample output: 

```
--- Sample 1 ---
In science class, we learned that the water cycle has three main steps: evaporation, condensation, and precipitation. First, we wanted to understand why the water cycle is so important for the ecosystem and the ecosystem. Second, we wanted to understand how the water cycle works. We wanted to know how the water cycle works, and why it is important for the ecosystem. Finally we wanted to understand the different ways water flow is done by the water cycle
```

## Limitations

- With about 46M parameters, the model writes fluent-looking text but often gets facts wrong or repeats itself.
- It is a base model, not instruction-tuned, so it continues text instead of answering questions.
- The context length is fixed at 512 tokens because it uses learned position embeddings.
- English only.

## Acknowledgements

- [SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) (FineWeb-Edu and Cosmopedia v2) by Hugging Face
- Architecture based on the GPT-2 / nanoGPT family of models

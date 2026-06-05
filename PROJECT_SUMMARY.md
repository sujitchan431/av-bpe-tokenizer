# Training LLM from Scratch — BPE Tokenizer

## Project Overview

Built a **Byte-Pair Encoding (BPE) tokenizer from scratch** using HuggingFace `tokenizers` library on the WikiText-2 dataset. Covers the full tokenizer training pipeline: data loading, cleaning, deduplication, BPE training, evaluation, and saving as a HuggingFace-compatible tokenizer.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Tokenizer Framework | HuggingFace `tokenizers` (BPE, BpeTrainer, TemplateProcessing) |
| HF Wrapper | `transformers.PreTrainedTokenizerFast` |
| Dataset | WikiText-2 via `datasets` (Salesforce/wikitext) |
| Language | Python 3.x |

## File Structure

```
Training LLM from scratch/
└── assignment/
    ├── bpe_tokenizer.py         ← Main tokenizer training script (252 lines)
    ├── train_corpus.txt         ← Cleaned training text (written during run)
    └── bpe_tokenizer/
        ├── tokenizer.json       ← Saved tokenizer (vocab + merges)
        └── tokenizer_config.json ← HF tokenizer config
```

## Pipeline Steps

```
1. Load WikiText-2 (train/val/test splits)
2. Clean: remove <unk>, collapse whitespace, strip empty lines
3. Deduplicate across all splits
4. Write training corpus to train_corpus.txt
5. Train BPE tokenizer (vocab_size=30K, min_frequency=2)
6. Add post-processor: [CLS] $A [SEP] template
7. Evaluate: avg tokens/sentence, compression ratio, [UNK] rate
8. Wrap in PreTrainedTokenizerFast (HF-compatible)
9. Save to bpe_tokenizer/ → reload → demo encode/decode
```

## Tokenizer Configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | BPE (Byte-Pair Encoding) |
| Vocab size target | 30,000 tokens |
| Minimum merge frequency | 2 |
| Special tokens | `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]` |
| Normalizer | `Strip()` (whitespace only, preserves case) |
| Pre-tokenizer | `Whitespace()` |
| Post-processor | `TemplateProcessing`: `[CLS] $A [SEP]` / pair: `[CLS] $A [SEP] $B [SEP]` |

## Evaluation Metrics

- **Avg tokens/sentence** — measured on validation + test splits
- **Median tokens/sentence**
- **Compression ratio** (chars/token — higher = more efficient)
- **[UNK] rate** — sentences containing unknown tokens (%)
- **Consistency check** — same text always produces same token IDs

## Outputs

- `bpe_tokenizer/tokenizer.json` — complete vocabulary, merge rules, special token config
- `bpe_tokenizer/tokenizer_config.json` — HuggingFace compatible wrapper config
- Demo: encode → decode → pair encoding → round-trip verification

## Work Completed

- [x] WikiText-2 loading via HuggingFace datasets
- [x] Text cleaning + deduplication pipeline
- [x] BPE training (30K vocab, min_freq=2)
- [x] CLS/SEP post-processing template
- [x] Evaluation on val + test (3 metrics)
- [x] Consistency check
- [x] Save as HuggingFace `PreTrainedTokenizerFast`
- [x] Reload + encode/decode + pair encoding demo

## Complexity

**Medium-High** — First LLM training component. Building a production-quality tokenizer requires understanding BPE algorithm, special token handling, post-processors, HF serialization format, and evaluation methodology.

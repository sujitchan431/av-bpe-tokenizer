<div align="center">

# 🔤 BPE Tokenizer from Scratch

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Tokenizers-yellow?logo=huggingface)](https://huggingface.co/docs/tokenizers)
[![WikiText-2](https://img.shields.io/badge/Dataset-WikiText--2-lightgrey)](https://huggingface.co/datasets/Salesforce/wikitext)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Train a Byte-Pair Encoding (BPE) tokenizer from scratch on WikiText-2, evaluate coverage and compression, then save as a HuggingFace-compatible `PreTrainedTokenizerFast`.

**🎓 Part of the [Analytics Vidhya GenAI Pinnacle Plus Program](https://www.analyticsvidhya.com/)**

</div>

---

## 📋 Overview

First step toward "Training LLM from Scratch" — building the tokenizer before any model training. Covers the complete tokenizer pipeline: data loading, text cleaning, deduplication, BPE training with special tokens and post-processors, evaluation on val/test splits, and HuggingFace serialization.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Tokenizer | HuggingFace `tokenizers` (Rust-backed, fast) |
| HF Wrapper | `PreTrainedTokenizerFast` |
| Dataset | WikiText-2 via `datasets` |
| Language | Python 3.x |

---

## 📁 Project Structure

```
Training LLM from scratch/
└── assignment/
    ├── bpe_tokenizer.py         ← Training script (252 lines)
    ├── train_corpus.txt         ← Cleaned training text
    └── bpe_tokenizer/
        ├── tokenizer.json       ← Vocabulary + merge rules
        └── tokenizer_config.json
```

---

## 🚀 Run

```bash
pip install tokenizers transformers datasets
python assignment/bpe_tokenizer.py
```

---

## ⚙️ Tokenizer Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| Algorithm | BPE | Best balance of vocab coverage and fertility |
| Vocab size | 30,000 tokens | Standard BERT-scale vocabulary |
| Min merge frequency | 2 | Only merge subword pairs seen ≥2× |
| Normalizer | `Strip()` | Preserve case (Wikipedia is case-sensitive) |
| Pre-tokenizer | `Whitespace()` | Word-level initial split |
| Post-processor | `[CLS] $A [SEP]` | BERT-compatible sequence wrapping |
| Special tokens | `[PAD] [UNK] [CLS] [SEP] [MASK]` | Full BERT special token set |

---

## 🔄 Pipeline

```
WikiText-2 (train/val/test)
    ↓ clean_text(): remove <unk>, collapse whitespace
    ↓ deduplicate (set-based)
    ↓ write to train_corpus.txt
    ↓
BpeTrainer(vocab_size=30000, min_frequency=2)
    ↓
TemplateProcessing([CLS] $A [SEP])
    ↓
Evaluate on val + test:
    → avg tokens/sentence
    → compression ratio (chars/token)
    → [UNK] rate (% sentences with unknown tokens)
    → consistency check (same text → same IDs)
    ↓
PreTrainedTokenizerFast.save_pretrained("bpe_tokenizer/")
```

---

## 📊 What Good Metrics Look Like

| Metric | Target |
|--------|--------|
| Compression ratio | 4–6 chars/token (good subword split) |
| [UNK] rate (val) | < 5% (high coverage) |
| Consistency | 100% (deterministic) |

---

## 💡 Key Learnings

- **BPE algorithm** — frequency-based iterative merging of character pairs
- Why BPE beats character-level (shorter sequences) and word-level (no OOV problem)
- `min_frequency=2` — prevents rare noise merges
- **TemplateProcessing** — adding `[CLS]`/`[SEP]` at the tokenizer level, not post-hoc
- **`PreTrainedTokenizerFast`** — HuggingFace wrapper enabling `from_pretrained()` compatibility
- Pair encoding — `[CLS] A [SEP] B [SEP]` pattern for sentence-pair tasks
- Compression ratio as a tokenizer quality metric

---

## 🎓 Program Context

**Analytics Vidhya GenAI Pinnacle Plus Program** — Training LLM from Scratch module (tokenizer component).

---

## 📄 License

MIT © 2026 [sujitchan431](https://github.com/sujitchan431)

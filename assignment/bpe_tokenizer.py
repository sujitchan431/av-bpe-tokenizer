"""
BPE Tokenizer from scratch using WikiText-2 dataset.
Covers: load, clean, train, evaluate, save/reload.
"""

import re
import sys
import json
from collections import Counter
from datasets import load_dataset
from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.normalizers import Strip
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
import statistics

# Force UTF-8 output so Unicode chars print cleanly on Windows
sys.stdout.reconfigure(encoding="utf-8")


# ─── 1. LOAD DATASET ────────────────────────────────────────────────────────

print("=" * 60)
print("STEP 1: Loading WikiText-2 dataset")
print("=" * 60)

dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1")

for split, ds in dataset.items():
    print(f"  {split}: {len(ds):,} rows")

print(f"\nColumn names: {dataset['train'].column_names}")
print(f"\nSample rows (train):")
for i, row in enumerate(dataset["train"].select(range(5))):
    print(f"  [{i}] {repr(row['text'][:80])}")


# ─── 2. DATA CLEANING & PREPROCESSING ───────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: Cleaning & Deduplication")
print("=" * 60)


def clean_text(text: str) -> str:
    """Remove <unk>, normalize whitespace, strip leading/trailing space."""
    text = text.replace("<unk>", "")
    # Collapse multiple spaces/tabs into one space
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()
    return text


def preprocess_split(split_data):
    """Clean and deduplicate a split; return list of non-empty strings."""
    seen = set()
    cleaned = []
    raw_count = len(split_data)
    empty_count = 0
    dup_count = 0

    for row in split_data:
        text = clean_text(row["text"])
        if not text:
            empty_count += 1
            continue
        if text in seen:
            dup_count += 1
            continue
        seen.add(text)
        cleaned.append(text)

    print(f"  Raw rows       : {raw_count:,}")
    print(f"  Empty removed  : {empty_count:,}")
    print(f"  Duplicates     : {dup_count:,}")
    print(f"  Final rows     : {len(cleaned):,}")
    return cleaned


print("\nTraining split:")
train_texts = preprocess_split(dataset["train"])

print("\nValidation split:")
val_texts = preprocess_split(dataset["validation"])

print("\nTest split:")
test_texts = preprocess_split(dataset["test"])

# Write training corpus to a temp file (tokenizers library reads from files)
corpus_path = "train_corpus.txt"
with open(corpus_path, "w", encoding="utf-8") as f:
    for line in train_texts:
        f.write(line + "\n")

print(f"\nCorpus written to '{corpus_path}' ({len(train_texts):,} lines)")


# ─── 3. TOKENIZER TRAINING ───────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3: Training BPE Tokenizer")
print("=" * 60)

VOCAB_SIZE = 30_000
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

# Initialise BPE model with unknown-token wired in
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Normaliser: strip extra whitespace only (preserve case for wiki text)
tokenizer.normalizer = Strip()

# Pre-tokeniser: split on whitespace
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=SPECIAL_TOKENS,
    min_frequency=2,          # merge only pairs that appear ≥2 times
    show_progress=True,
)

print(f"\nVocab size target : {VOCAB_SIZE:,}")
print(f"Special tokens    : {SPECIAL_TOKENS}")
print(f"Min merge freq    : 2")
print("\nTraining …")

tokenizer.train(files=[corpus_path], trainer=trainer)

# Post-processor: wrap encoded sequences with [CLS] … [SEP]
cls_id = tokenizer.token_to_id("[CLS]")
sep_id = tokenizer.token_to_id("[SEP]")

tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
)

actual_vocab = tokenizer.get_vocab_size()
print(f"\nActual vocab size : {actual_vocab:,}")
print("Training complete.")


# ─── 4. TOKENIZER EVALUATION ────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: Evaluation")
print("=" * 60)


def evaluate_split(name: str, texts: list):
    """Compute vocab coverage, avg tokens/sentence, compression ratio."""
    token_counts = []
    char_counts = []
    unknown_sentences = 0
    unk_id = tokenizer.token_to_id("[UNK]")

    for text in texts:
        enc = tokenizer.encode(text)
        token_counts.append(len(enc.ids))
        char_counts.append(len(text))
        if unk_id in enc.ids:
            unknown_sentences += 1

    avg_tokens = statistics.mean(token_counts)
    median_tokens = statistics.median(token_counts)
    total_chars = sum(char_counts)
    total_tokens = sum(token_counts)
    compression = total_chars / total_tokens if total_tokens else 0
    unk_rate = unknown_sentences / len(texts) * 100 if texts else 0

    print(f"\n  [{name}]")
    print(f"    Sentences           : {len(texts):,}")
    print(f"    Avg tokens/sentence : {avg_tokens:.1f}")
    print(f"    Median tokens/sent  : {median_tokens:.1f}")
    print(f"    Compression ratio   : {compression:.2f} chars/token")
    print(f"    Sentences w/ [UNK]  : {unknown_sentences:,} ({unk_rate:.1f}%)")
    return avg_tokens, compression


print(f"\nVocabulary size : {actual_vocab:,}")

val_avg, val_cr   = evaluate_split("Validation", val_texts)
test_avg, test_cr = evaluate_split("Test",       test_texts)

# Tokenisation consistency check — same text → same ids every time
print("\n  [Consistency check]")
sample = "The cat sat on the mat near the university library."
enc1 = tokenizer.encode(sample).ids
enc2 = tokenizer.encode(sample).ids
consistent = enc1 == enc2
print(f"    Input  : {repr(sample)}")
print(f"    Tokens : {tokenizer.encode(sample).tokens}")
print(f"    IDs    : {enc1}")
print(f"    Consistent across calls : {consistent}")


# ─── 5. SAVE & RELOAD ───────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5: Save & Reload")
print("=" * 60)

SAVE_DIR = "bpe_tokenizer"

# Wrap in PreTrainedTokenizerFast so it's HF-compatible
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

hf_tokenizer.save_pretrained(SAVE_DIR)
print(f"\nTokenizer saved to '{SAVE_DIR}/'")

# Reload
reloaded = PreTrainedTokenizerFast.from_pretrained(SAVE_DIR)
print("Tokenizer reloaded successfully.")

# Encode / decode demo
demo_text = "Hugging Face tokenizers are used for natural language processing tasks."
encoded = reloaded(demo_text, return_tensors=None)
print(f"\nDemo encode:")
print(f"  Input   : {repr(demo_text)}")
print(f"  Input IDs  : {encoded['input_ids']}")
print(f"  Tokens  : {reloaded.convert_ids_to_tokens(encoded['input_ids'])}")

decoded = reloaded.decode(encoded["input_ids"], skip_special_tokens=True)
print(f"\nDemo decode:")
print(f"  Output  : {repr(decoded)}")

# Pair encoding
text_a = "The model learns from data."
text_b = "Data drives machine learning."
pair_enc = reloaded(text_a, text_b)
print(f"\nPair encode:")
print(f"  Text A  : {repr(text_a)}")
print(f"  Text B  : {repr(text_b)}")
print(f"  Input IDs  : {pair_enc['input_ids']}")
print(f"  Tokens  : {reloaded.convert_ids_to_tokens(pair_enc['input_ids'])}")

print("\n" + "=" * 60)
print("Done.")
print("=" * 60)

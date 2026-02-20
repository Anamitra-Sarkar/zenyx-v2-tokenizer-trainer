---
language:
  - en
  - code
  - math
license: apache-2.0
tags:
  - tokenizer
  - bpe
  - byte-level-bpe
  - zenyx
  - reasoning
  - math
  - code
  - multilingual-code
pipeline_tag: text-generation
libraries:
  - transformers
  - tokenizers
---

# Zenyx-v2 Tokenizer

> **HuggingFace Hub:** [`Arko007/zenyx-v2-tokenizer`](https://huggingface.co/Arko007/zenyx-v2-tokenizer)  
> **Tokenizer Trainer Repo:** [`Anamitra-Sarkar/zenyx-v2-tokenizer-trainer`](https://github.com/Anamitra-Sarkar/zenyx-v2-tokenizer-trainer)  
> **Pretraining Repo:** [`Anamitra-Sarkar/zenyx-v2-pretrain`](https://github.com/Anamitra-Sarkar/zenyx-v2-pretrain)

The official tokenizer for the **Zenyx-v2** family of language models. Zenyx-v2 is a Nano-Titan architecture LLM (85M unique parameters, 32 effective layers via weight-sharing recurrence) designed for strong math and code reasoning within a 14,336-token context window.

This tokenizer was trained from scratch using Byte-Level BPE on a carefully mixed 1 GB corpus (40% math · 40% code · 20% English), chosen to match the exact pretraining data distribution of Zenyx-v2-base.

---

## Quick Start

```python
from transformers import PreTrainedTokenizerFast

tok = PreTrainedTokenizerFast.from_pretrained("Arko007/zenyx-v2-tokenizer")

# Basic usage
ids = tok.encode("def fibonacci(n): return n if n<=1 else fibonacci(n-1)+fibonacci(n-2)")
print(tok.decode(ids))   # perfect roundtrip
print(f"Tokens: {len(ids)}")

# Math example
ids = tok.encode("∫₀^∞ e^{-x²} dx = √π/2")
print(tok.decode(ids))   # ✓ roundtrip

# Chain-of-thought with special tokens
ids = tok.encode("<think>\n2+2=4\n</think>\n<verify>\n✓\n</verify>")
print(tok.decode(ids))   # ✓ roundtrip
```

---

## Technical Specifications

| Property | Value |
|---|---|
| Algorithm | Byte-Level BPE |
| Vocabulary size | 32,768 |
| Training corpus | 1 GB |
| Data mix | 40% math · 40% code · 20% English |
| `model_max_length` | 131,072 (set conservatively; model trains at 14,336) |
| BOS token | `<\|endoftext\|>` (id=0) |
| EOS token | `<\|endoftext\|>` (id=0) |
| PAD token | `<\|pad\|>` (id=1) |
| UNK token | `<\|unk\|>` (id=2) |
| `fuse_unk` | `False` |
| `min_frequency` | 2 |
| Prefix space | `False` |

---

## Training Data

The tokenizer was trained on a 1 GB mixed corpus streamed directly into the BPE trainer — no spool file was written to disk. Sources were consumed sequentially per language to keep RAM flat on Kaggle's 13 GB kernel limit.

### Math — 400 MB (40%)

**Source:** [`HuggingFaceTB/finemath`](https://huggingface.co/datasets/HuggingFaceTB/finemath) — `finemath-4plus` split  
High-quality mathematical content filtered to educational score ≥ 4. This covers olympiad problems, textbook solutions, LaTeX-heavy derivations, and numerical reasoning chains.

### Code — 400 MB (40%)

**Source:** [`bigcode/starcoderdata`](https://huggingface.co/datasets/bigcode/starcoderdata) — 25 languages, streamed sequentially  
Languages in priority order:

| Tier | Languages |
|---|---|
| Tier 1 — highest demand | Python, JavaScript, TypeScript, Java, C, C++, C#, Go, Rust |
| Tier 2 — widely used | Kotlin, PHP, Ruby, Shell, SQL, HTML, CSS, Markdown |
| Tier 3 — useful extras | YAML, JSON, Dockerfile, CUDA, R, Dart, Swift, Scala |

Each language received an equal byte budget (`total_code_bytes / 25`). Documents shorter than 20 characters were discarded. Between languages, the dataset object was explicitly deleted and `gc.collect()` called to release all parquet/HTTP buffers.

### English — 200 MB (20%)

**Source:** [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — `sample-10BT` split, `score >= 3.0` filter  
High-quality web-scraped educational English text. The quality filter ensures only teacher-level or textbook-level content contributes to the tokenizer's English coverage.

---

## Pre-tokenizer Design

The pre-tokenizer is a `Sequence` of three stages applied in order:

**Stage 1 — Regex Split (GPT-4-style pattern, Rust engine)**

```
's|'t|'re|'ve|'m|'ll|'d
| ?\p{L}+
| ?[^\s\p{L}]+[\r\n]*
|\s*[\r\n]+
|\s+
```

This isolates contractions, word tokens (Unicode letters), punctuation+symbols, newlines, and whitespace into separate pre-token groups. Whitespace is attached to the *preceding* token (space-before style), which matches the GPT-2/LLaMA convention and improves compression on natural-language text.

**Stage 2 — Digit Isolation (`individual_digits=True`)**

Every decimal digit (0–9) is split into its own pre-token before BPE merging. This prevents the tokenizer from learning multi-digit merges like `42` or `1024`, which would make arithmetic generalisation harder at fine-tuning time. After isolation, `3.14159` becomes exactly 6 digit tokens plus the decimal point — fully reversible.

Unicode numeric characters (e.g., Bengali digits `০১২…`, Roman numerals `ⅱ`) are normalised to their ASCII equivalents before the split, so they tokenise identically to ASCII digits.

**Stage 3 — ByteLevel (`add_prefix_space=False`, `use_regex=False`)**

Encodes remaining bytes as printable ASCII surrogates (the standard 256-character byte-level alphabet). This guarantees lossless roundtrips for any UTF-8 input — no character can ever be OOV.

---

## Special Tokens

| Token | ID | Purpose |
|---|---|---|
| `<\|endoftext\|>` | 0 | BOS / EOS — document boundary |
| `<\|pad\|>` | 1 | Padding — ignored in loss via `labels != PAD_ID` |
| `<\|unk\|>` | 2 | Unknown — unreachable in practice (byte-level fallback) |
| `<think>` | 3 | Opens a chain-of-thought reasoning block |
| `</think>` | 4 | Closes a chain-of-thought reasoning block |
| `<scratchpad>` | 5 | Marks intermediate scratchpad computation |
| `<verify>` | 6 | Marks a verification / answer-checking block |

The `<think>`, `</think>`, `<scratchpad>`, and `<verify>` tokens are reserved for supervised fine-tuning on reasoning traces (e.g., GSM8K, MATH, ARC-style CoT). They are never seen during pretraining but are part of the vocabulary from day one so no embedding re-initialisation is needed at the SFT stage.

---

## Compression Characteristics

With a 1 GB training corpus (small relative to GPT-4's tokenizer corpus), the tokenizer is well-calibrated for math and code but achieves modest natural-language compression. Expected ratios on held-out content:

| Content Type | Expected Characters/Token |
|---|---|
| Python source code | ~4.5 – 5.5 |
| LaTeX mathematics | ~3.5 – 4.5 |
| Structured English prose | ~4.0 – 5.0 |
| Mixed math + code | ~4.5 – 5.0 |

Digit isolation intentionally reduces compression on numeric-heavy text in exchange for better arithmetic alignment at fine-tuning.

---

## Postprocessor and Decoder

- **Post-processor:** `ByteLevel(trim_offsets=False)` — preserves exact byte offsets for span-level tasks.
- **Decoder:** `ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=True)` — recovers original whitespace exactly, including leading spaces on words. Roundtrip is lossless for any UTF-8 string.

---

## Text Sanitisation

Before feeding text to the BPE trainer, every document passes through `sanitise_text()`:

- Empty or whitespace-only strings are discarded (`None` returned).
- Null bytes (`\x00`) are stripped.
- Malformed UTF-8 sequences are replaced via `encode/decode` with `errors='replace'`.
- Non-ASCII numeric characters (e.g., `①`, `²`) are normalised to ASCII equivalents using `unicodedata.numeric()`.

This ensures no corrupt byte sequences enter the BPE merge frequency counts.

---

## Relationship to Zenyx-v2-base

This tokenizer is the **only** tokenizer compatible with [Zenyx-v2-base](https://huggingface.co/Arko007/zenyx-v2-base). The model's embedding table has exactly 32,768 rows (`VOCAB_SIZE = 32_768`), asserted at training startup:

```python
assert len(tokenizer) == VOCAB_SIZE, f"Vocab mismatch: {len(tokenizer)} vs {VOCAB_SIZE}"
```

The pretraining script uses `PAD_ID = tokenizer.convert_tokens_to_ids("<|pad|>")` (= 1) to construct the loss mask — padding tokens contribute zero loss. `EOS_ID = 0` is used as the document boundary separator in the packed sequence stream.

---

## Citation

If you use this tokenizer in your research, please cite:

```bibtex
@misc{zenyx-v2-tokenizer,
  author       = {Anamitra Sarkar},
  title        = {Zenyx-v2 Tokenizer: Byte-Level BPE for Math and Code Reasoning},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/Arko007/zenyx-v2-tokenizer}},
  note         = {32k vocabulary, 1GB mixed corpus (FineMath-4+ / StarCoderData / FineWeb-Edu)}
}
```

---

## License

Apache 2.0. The training data sources are licensed separately:
- FineMath: [ODC-By](https://opendatacommons.org/licenses/by/)
- StarCoderData: [BigCode OpenRAIL-M](https://huggingface.co/spaces/bigcode/bigcode-model-license-agreement)
- FineWeb-Edu: [ODC-By](https://opendatacommons.org/licenses/by/)

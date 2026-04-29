# privacy-filter-redactor

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/aaronaco/privacy-filter-redactor-lib-demo)
[![Model](https://img.shields.io/badge/🤗%20Model-openai%2Fprivacy--filter-blue)](https://huggingface.co/openai/privacy-filter)
[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-green)](LICENSE)
[![Model License: Apache 2.0](https://img.shields.io/badge/Model%20License-Apache%202.0-lightgrey)](https://www.apache.org/licenses/LICENSE-2.0)

A local PII detection and redaction library powered by OpenAI's [`privacy-filter`](https://huggingface.co/openai/privacy-filter) model, extended with a **Viterbi CRF decoder** and a developer-friendly Python API with calibrated precision/recall presets.

**The model runs entirely on your machine. No data ever leaves your environment.**

**[Try the live demo →](https://huggingface.co/spaces/aaronaco/privacy-filter-redactor-lib-demo)**

## Overview

This library wraps the open-weight **OpenAI Privacy Filter** model released in 2026, providing a clean Python interface for PII detection and redaction. While the original model release provided raw weights and a CLI tool (`opf`), this library (`privacy-filter-redactor-lib`) contributes:

- A **Viterbi CRF decoder** with calibrated sensitivity presets (High Recall / High Precision / Balanced), implementing the operating-point calibration methodology described in the original model card.
- **Automated subword-to-string alignment** for clean span extraction with character-level offsets.
- A **seamless developer API** designed for integration into data pipelines, LLM preprocessing, and enterprise workflows.

## Model Background

### Architecture: OpenAI Privacy Filter

Privacy Filter [(OpenAI, 2026)](#references) is a **bidirectional token classification model** for PII detection in unstructured text. Its architecture departs from conventional masked-language-model approaches: the model was first **pretrained autoregressively** (following the gpt-oss pretraining routine), then converted into a bidirectional token classifier by replacing the language-model output head with a token-classification head and relaxing the causal attention mask into a **bidirectional banded attention pattern** (band size 128; effective window 257 tokens including self).

Key architectural properties:

| Property | Value |
|---|---|
| Total parameters | 1.5B |
| Active parameters (MoE routing) | 50M |
| Architecture | Sparse MoE transformer, 8 blocks |
| Attention | Grouped-query attention (14Q / 2KV heads), RoPE |
| Feed-forward | Sparse MoE, 128 experts, top-4 routing per token |
| Residual stream width | d_model = 640 |
| Context window | 128,000 tokens |
| Output classes | 33 (BIOES × 8 labels + background O) |

The sparse Mixture-of-Experts feed-forward layers give the model a 1.5B parameter footprint while keeping only ~50M parameters active per forward pass, enabling deployment on consumer hardware and in-browser inference.

### Training Procedure

Training proceeded in two stages. In the first stage, the model was pretrained as a generative language model on a large text corpus. In the second stage, the architecture was modified for bidirectional classification and post-trained with a **supervised token-level BIOES objective** on a mix of publicly available data and internally generated synthetic datasets, covering realistic natural text and targeted privacy-pattern diversity. Ground truth annotations for publicly available data without existing labels were produced using a GPT-5 family model under a 2×2 annotation protocol (two prompt formats × two reasoning settings).

Synthetic training data was constructed by applying format-matching augmentation to public datasets, inserting spans into natural context, and applying automated quality controls.

### Sequence Decoding

Rather than taking an independent argmax per token, Privacy Filter applies a **constrained Viterbi decoder** with linear-chain transition scoring. The decoder enforces valid BIOES boundary transitions and scores complete label paths using start, transition, and end terms, plus six transition-bias parameters controlling background persistence, span entry, span continuation, span closure, and boundary-to-boundary handoff. This library exposes those bias parameters as three named operating points described below.

### Label Taxonomy

The model detects 8 privacy span categories, each expanded into BIOES boundary tags (B-, I-, E-, S-) plus background O, yielding 33 output classes:

| Label | Description |
|---|---|
| `private_person` | Name of a private individual, including usernames and handles |
| `private_email` | Email address used for personal communication |
| `private_phone` | Phone number associated with a private person |
| `private_address` | Physical location or address associated with a private person |
| `private_url` | Web URL or IP address meant for a private audience |
| `private_date` | Date of birth, birth year, or other identifying datetime |
| `account_number` | Credit card number, bank account number, or other account identifier |
| `secret` | API key, password, or other credential |

> **Note:** Labels are intentionally broad and cover attributes that identify real people. Placeholder values (e.g., example API keys, template addresses) are out of scope by design.

## Installation

This library is designed to be installed directly from GitHub. [uv](https://docs.astral.sh/uv/) is recommended for dependency management.

### For projects (recommended)

```bash
uv add "git+https://github.com/aaronaco/privacy-filter-redactor-lib.git"
```

Pin to a specific tag for reproducibility:

```bash
uv add "git+https://github.com/aaronaco/privacy-filter-redactor-lib.git" --tag v0.1.1
```

### For scripts or ad-hoc usage

```bash
uv pip install "git+https://github.com/aaronaco/privacy-filter-redactor-lib.git"
```

> `torch` and `transformers` are resolved automatically by `uv`. The model (~3 GB) downloads on first use and is cached at `~/.cache/huggingface/hub`.

## Usage

```python
from privacy_filter_redactor import PIIRedactor

# Auto-detects CPU/CUDA
redactor = PIIRedactor()

text = "Contact Alice Smith at alice@example.com or 555-0199."
```

### Redact PII

```python
redacted = redactor.redact(text)
# "Contact [PRIVATE_PERSON] at [PRIVATE_EMAIL] or [PRIVATE_PHONE]."
```

### Detect PII (with positions)

```python
entities = redactor.detect(text)
for ent in entities:
    print(f"Found {ent.label} at {ent.start}:{ent.end}")
```

### Custom placeholders

```python
# Fixed string
print(redactor.redact(text, placeholder="***"))

# Callable (e.g., hash-based pseudonymization)
import hashlib
def hasher(label, val):
    return f"[{label}:{hashlib.md5(val.encode()).hexdigest()[:6]}]"

print(redactor.redact(text, placeholder=hasher))
```

### LLM pipeline integration

```python
from privacy_filter_redactor import PIIRedactor
import openai

redactor = PIIRedactor()
client = openai.OpenAI(api_key="...")

user_input = "My name is Alice Smith, email alice@example.com. Summarize this."
clean_input = redactor.redact(user_input)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": clean_input}]
)
```

## Configuration

### Operating Points (Precision / Recall Tradeoff)

The library exposes three decoding modes corresponding to different Viterbi bias shift configurations, as described in the OpenAI model card:

```python
from privacy_filter_redactor import DecodingMode

redactor.redact(text, mode=DecodingMode.BALANCED)       # default
redactor.redact(text, mode=DecodingMode.HIGH_RECALL)    # -5.0 bias shift; broader redaction
redactor.redact(text, mode=DecodingMode.HIGH_PRECISION) # +5.0 bias shift; conservative redaction
```

`HIGH_RECALL` is appropriate when missing PII is more costly than over-redaction (e.g., data sanitization before storage). `HIGH_PRECISION` is appropriate when over-redaction degrades downstream utility (e.g., document review workflows).

### Long Document Processing

The model's 128,000 token context window eliminates the need for chunking in most real-world workloads:

```python
redacted_content = redactor.redact_file("path/to/large_doc.txt")
```

## Evaluation Results

The following results are taken directly from the OpenAI Privacy Filter model card [(OpenAI, 2026)](#references).

### PII-Masking-300k Benchmark

Evaluated on the test holdout of [PII-Masking-300k](https://huggingface.co/datasets/ai4privacy/pii-masking-300k), a large multilingual synthetic benchmark. "Corrected" metrics exclude spans identified as missing labels in an adjudication step (GPT-5.4 reasoning + human validation, 100% agreement on 100 reviewed spans).

| Dataset Version | Precision (tokens) | Recall (tokens) | F1 (tokens) | F1 (spans) |
|---|---|---|---|---|
| Baseline | 0.940 | 0.980 | 0.960 | 0.926 |
| Corrected | 0.968 | 0.981 | 0.974 | 0.942 |

### Multilingual Performance (PII-Masking-300k)

| Language | Examples | Recall | Precision | F1 |
|---|---|---|---|---|
| English | 7,946 | 0.965 | 0.905 | 0.934 |
| French | 8,413 | 0.969 | 0.889 | 0.927 |
| German | 8,120 | 0.965 | 0.890 | 0.926 |
| Spanish | 7,816 | 0.968 | 0.901 | 0.933 |
| Italian | 7,976 | 0.959 | 0.886 | 0.921 |
| Dutch | 7,457 | 0.937 | 0.892 | 0.914 |

### Credential Detection (CredData)

Evaluated on the [CredData](https://github.com/Samsung/CredData) benchmark for `secret` detection in codebases. Lower span-level F1 reflects boundary discrepancies on otherwise partially-detected spans.

| Dataset Version | Precision (tokens) | Recall (tokens) | F1 (tokens) | F1 (spans) |
|---|---|---|---|---|
| Baseline | 0.747 | 0.965 | 0.842 | 0.617 |
| Corrected | 0.750 | 0.965 | 0.844 | 0.624 |

### Fine-tuning Efficiency

When domain adaptation is needed, Privacy Filter reaches strong performance on small amounts of in-domain data. On the SPY dataset (legal questions and medical consultations), training on just 10% of the training split drives token-level F1 above **96%**.

| Train Fraction | Best Epoch | Precision | Recall | F1 |
|---|---|---|---|---|
| 0% | — | 0.414 | 0.795 | 0.545 |
| 1% | 13 | 0.888 | 0.871 | 0.879 |
| 10% | 39 | 0.963 | 0.960 | 0.962 |
| 50% | 18 | 0.983 | 0.982 | 0.983 |

## Limitations

The following known failure modes are documented in the original model card:

- **Under-detection** of uncommon names, regional naming conventions, initials, or domain-specific identifiers.
- **Over-redaction** of public entities, locations, or common nouns in ambiguous local contexts.
- **Fragmented span boundaries** in mixed-format text, heavy punctuation, or explicit line breaks.
- **Missed secrets** for novel credential formats, project-specific token patterns, or split secrets.
- **Adversarial formatting** (spacing, phonetic alphabet, at-dot obfuscation) degrades precision.
- **One-hop alias resolution** (defining an alias earlier in context and using it later) is not reliably detected; recall degrades as the referent distance grows.

Privacy Filter is a redaction aid, not an anonymization or compliance guarantee. It should be used as one layer within a holistic privacy-by-design system. High-sensitivity deployments (medical, legal, financial, HR) should retain human review paths.

## Contributing

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

### Development Setup

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and sync
git clone https://github.com/aaronaco/privacy-filter-redactor-lib.git
cd privacy-filter-redactor-lib
uv sync --all-groups
```

### Common Workflows

```bash
# Logic tests only — no model download required
SKIP_HEAVY=true uv run pytest

# Full test suite (requires ~3 GB model download)
uv run pytest

# Lint and format
uv run ruff check --fix
uv run ruff format
```

### Architecture

- **`src/privacy_filter_redactor/redactor.py`** — Main API. Handles model loading, tokenizer character-offset mapping, and string surgery.
- **`src/privacy_filter_redactor/decoder.py`** — Viterbi CRF implementation. Enforces BIOES consistency and operating-point bias shifts.
- **`src/privacy_filter_redactor/entities.py`** — Core data structures for detected entities.

### Proposing Changes

1. Create a feature branch.
2. Ensure all logic tests pass (`SKIP_HEAVY=true`).
3. Run `ruff format .` and `ruff check .`.
4. Update `CHANGELOG.md` following [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.
5. Submit a Pull Request.

## References & Citations

OpenAI. (2026). **OpenAI Privacy Filter** [Model card]. https://cdn.openai.com/pdf/c66281ed-b638-456a-8ce1-97e9f5264a90/OpenAI-Privacy-Filter-Model-Card.pdf

OpenAI. (2026). **openai/privacy-filter** [Model weights]. Hugging Face. https://huggingface.co/openai/privacy-filter

```bibtex
@misc{openai2026privacyfilter,
  title        = {OpenAI Privacy Filter},
  author       = {OpenAI},
  year         = {2026},
  howpublished = {Model Card},
  url          = {https://cdn.openai.com/pdf/c66281ed-b638-456a-8ce1-97e9f5264a90/OpenAI-Privacy-Filter-Model-Card.pdf}
}
```

## License

The code in this repository is licensed under the **MIT License**.

The underlying model ([`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter)) is released by OpenAI under the **Apache 2.0 License**, permitting commercial use, modification, and redistribution with attribution.

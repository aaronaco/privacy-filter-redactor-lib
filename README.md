# privacy-filter-redactor-lib

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/aaronaco/privacy-filter-redactor-lib-demo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Local PII (Personally Identifiable Information) detection and redaction python library, powered by OpenAI's `privacy-filter` model and optimized with a Viterbi CRF decoder.

## Demo

Try the library directly in your browser:
**[Launch Hugging Face Space Demo](https://huggingface.co/spaces/aaronaco/privacy-filter-redactor-lib-demo)**

## About

This library builds upon the open-weight **OpenAI Privacy Filter** model released in 2026. The original model uses a Sparse Mixture-of-Experts (MoE) transformer architecture (1.5B total / 50M active parameters) trained specifically for bidirectional token-classification of sensitive information.

- **Original Model**: [`openai/privacy-filter` on Hugging Face](https://huggingface.co/openai/privacy-filter)
- **Original Paper / Model Card**: [OpenAI Privacy Filter Model Card (PDF)](https://cdn.openai.com/pdf/c66281ed-b638-456a-8ce1-97e9f5264a90/OpenAI-Privacy-Filter-Model-Card.pdf)

While the original release provided the raw weights, this library (`privacy-filter-redactor-lib`) introduces a custom **Viterbi CRF Decoder** with calibrated sensitivity presets (High Recall / High Precision), automated subword-to-string alignment, and a seamless developer API designed for integration.

## Features

- **Local-First**: Runs entirely on your machine via `transformers` and `torch`. No data leaves your infrastructure.
- **Smart Decoding**: Uses a Viterbi CRF decoder to ensure coherent PII spans and boundary stability.
- **Long Context**: Leverages a 128k token context window to redact entire documents without chunking.
- **Precision/Recall Presets**: Simple `balanced`, `high_recall`, and `high_precision` modes.
- **Customizable**: Redact using default labels, custom strings, or hash-based callables.

## Installation

This library is designed to be installed directly from GitHub. It is highly recommended to use [uv](https://docs.astral.sh/uv/) for dependency management.

### For projects (recommended)

Add this library as a dependency to your project:

```bash
uv add "git+https://github.com/aaronaco/privacy-filter-redactor-lib.git"
```

To ensure stability, it is best practice to pin to a specific tag:

```bash
uv add "git+https://github.com/aaronaco/privacy-filter-redactor-lib.git" --tag v0.1.1
```

### For scripts or ad-hoc usage

Install directly into your current environment:

```bash
uv pip install "git+https://github.com/aaronaco/privacy-filter-redactor-lib.git"
```

_Note: This library requires `torch` and `transformers`. `uv` will resolve these automatically during installation._

## Quick Start

```python
from privacy_filter_redactor import PIIRedactor

# Initialize the redactor (auto-detects CPU/CUDA)
redactor = PIIRedactor()

text = "Contact Alice Smith at alice@example.com or 555-0199."

# 1. Simple Redaction
redacted = redactor.redact(text)
print(redacted)
# Output: "Contact [PRIVATE_PERSON] at [PRIVATE_EMAIL] or [PRIVATE_PHONE]."

# 2. Entity Detection
entities = redactor.detect(text)
for ent in entities:
    print(f"Found {ent.label} at {ent.start}:{ent.end}")

# 3. Custom Placeholders
# Fixed string
print(redactor.redact(text, placeholder="***"))

# Custom callable (e.g., hashing)
import hashlib
def hasher(label, val):
    return f"[{label}:{hashlib.md5(val.encode()).hexdigest()[:6]}]"

print(redactor.redact(text, placeholder=hasher))
```

## Advanced Usage

### Precision vs. Recall Modes

The library supports three decoding modes to control sensitivity, calibrated using official Viterbi bias shifts detailed in the OpenAI paper:

- `balanced`: The default setting, optimized for general use.
- `high_recall`: Favors catching more PII at the risk of higher false positives (applies a -5.0 bias shift).
- `high_precision`: Favors accuracy to minimize over-redaction (applies a +5.0 bias shift).

```python
from privacy_filter_redactor import DecodingMode
redacted = redactor.redact(text, mode=DecodingMode.HIGH_RECALL)
```

### Redacting Files

Process large files efficiently using the massive 128k context window:

```python
redacted_content = redactor.redact_file("path/to/large_doc.txt")
```

## Supported Categories

The model detects 8 specific privacy categories using BIOES taxonomy:

- `private_person`: Names of individuals.
- `private_email`: Email addresses.
- `private_phone`: Telephone numbers.
- `private_address`: Physical locations.
- `private_url`: Personal URLs.
- `private_date`: Dates of birth or other sensitive dates.
- `account_number`: Financial or system ID numbers.
- `secret`: Catch-all for credentials, API keys, and passwords.

---

## Contributing

Contributions are welcome! This project uses [uv](https://docs.astral.sh/uv/) for dependency management and development workflows.

### Development Setup

1. **Install uv**:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2. **Clone and Sync**:
    ```bash
    git clone https://github.com/aaronaco/privacy-filter-redactor-lib.git
    cd privacy-filter-redactor-lib
    uv sync --all-groups
    ```

### Common Workflows

Use `uv run` to execute commands within the project context:

- **Run Logic Tests (Fast)**: Tests the Viterbi decoder without downloading the 3GB model.
    ```bash
    SKIP_HEAVY=true uv run pytest
    ```
- **Run Full Test Suite (Requires 3GB model download)**:
    ```bash
    uv run pytest
    ```
- **Lint & Format**:
    ```bash
    uv run ruff check --fix
    uv run ruff format
    ```

### Architecture Overview

- **`src/privacy_filter_redactor/redactor.py`**: The main API entry point. Handles model loading, tokenizer character-offset mapping, and string surgery.
- **`src/privacy_filter_redactor/decoder.py`**: The Viterbi CRF implementation. This is the "brain" that enforces BIOES consistency and operating point shifts.
- **`src/privacy_filter_redactor/entities.py`**: Core data structures for detected entities.

### Proposing Changes

1. Create a feature branch.
2. Ensure all logic tests pass (`SKIP_HEAVY=true`).
3. Run `ruff format .` and `ruff check .` to maintain code quality.
4. Update `CHANGELOG.md` following the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.
5. Submit a Pull Request.

## Acknowledgements & Citation

If you use this work in your research or applications, please consider attributing the foundational model:

```bibtex
@misc{openai2026privacyfilter,
  title={OpenAI Privacy Filter},
  author={OpenAI},
  year={2026},
  url={https://cdn.openai.com/pdf/c66281ed-b638-456a-8ce1-97e9f5264a90/OpenAI-Privacy-Filter-Model-Card.pdf}
}
```

## 📜 License

MIT

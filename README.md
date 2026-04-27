# privacy-filter-redactor-lib

A high-performance Python library for local PII (Personally Identifiable Information) detection and redaction, powered by OpenAI's `privacy-filter` model.

## Features

- **Local-First**: Runs entirely on your machine via `transformers` and `torch`. No data leaves your infrastructure.
- **Smart Decoding**: Uses a Viterbi CRF decoder to ensure coherent PII spans and boundary stability.
- **Long Context**: Leverages a 128k token context window to redact entire documents without chunking.
- **Precision/Recall Presets**: Simple `balanced`, `high_recall`, and `high_precision` modes.
- **Customizable**: Redact using default labels, custom strings, or hash-based callables.

## Installation

```bash
pip install privacy-filter-redactor-lib
```

_Note: Ensure you have `torch` and `transformers` installed in your environment._

## Quick Start

```python
from privacy_filter_redactor import PIIRedactor

# Initialize the redactor (auto-detects CPU/CUDA)
redactor = PIIRedactor()

text = "Contact Jason at jason.statham@example.com or 555-0199."

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

The library supports three decoding modes to control sensitivity:

- `balanced`: The default setting, optimized for general use.
- `high_recall`: Favors catching more PII at the risk of higher false positives.
- `high_precision`: Favors accuracy to minimize over-redaction.

```python
redacted = redactor.redact(text, mode="high_recall")
```

### Redacting Files

Process large files efficiently using the massive 128k context window:

```python
redacted_content = redactor.redact_file("path/to/large_doc.txt")
```

## Supported Categories

The model detects 8 specific privacy categories:

- `private_person`: Names of individuals.
- `private_email`: Email addresses.
- `private_phone`: Telephone numbers.
- `private_address`: Physical locations.
- `private_url`: Personal URLs.
- `private_date`: Dates of birth or other sensitive dates.
- `account_number`: Financial or system ID numbers.
- `secret`: Catch-all for credentials, API keys, and passwords.

## License

MIT

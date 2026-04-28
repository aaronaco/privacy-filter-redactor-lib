---
title: Privacy Filter Redactor
emoji: 🛡️
colorFrom: blue
colorTo: slate
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
---

# Privacy Filter Redactor Demo

This is a demonstration of the `privacy-filter-redactor-lib`, a local-first PII redaction engine.

## Model Details
- **Architecture**: 1.5B/50M MoE Token Classifier
- **Decoding**: Viterbi CRF
- **Privacy**: No data leaves the Hugging Face Space memory during inference.

## How to use
1. Enter your text in the left panel.
2. Select a sensitivity mode (Balanced is usually best).
3. Click "Redact Text".
4. Review the redacted result and the detailed entity table below.

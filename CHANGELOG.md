# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-28

### Added

- High-performance local PII detection and redaction powered by OpenAI's `privacy-filter` model.
- One-liner API with `redact()`, `detect()`, `redact_with_details()`, and `redact_file()` methods.
- Support for `balanced`, `high_recall`, and `high_precision` sensitivity presets.
- Viterbi CRF decoder to ensure BIOES tag consistency and boundary stability.
- Subword-to-string alignment engine with automated whitespace trimming for precise redaction.
- Flexible placeholder support for fixed strings and custom callable logic.
- Native 128k token context window support for processing large documents.
- Structured `pytest` suite with isolated logic tests and heavy model-based validation.
- Modern `uv` based build system and CI/CD workflow for automated verification.
- This `CHANGELOG.md` file to track project evolution.

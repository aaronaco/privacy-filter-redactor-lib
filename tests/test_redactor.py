import pytest

from privacy_filter_redactor import DecodingMode


def test_basic_redaction(redactor):
    """Verify standard redaction of common PII types."""
    text = "Call Alice at 555-0101."
    redacted = redactor.redact(text)

    assert "[PRIVATE_PERSON]" in redacted
    assert "[PRIVATE_PHONE]" in redacted
    assert "Alice" not in redacted
    assert "555-0101" not in redacted


def test_detect_returns_correct_metadata(redactor):
    """Ensure detect() returns valid DetectedEntity objects."""
    text = "My email is test@example.com"
    entities = redactor.detect(text)

    assert len(entities) == 1
    ent = entities[0]
    assert ent.label == "private_email"
    assert ent.text == "test@example.com"
    assert isinstance(ent.start, int)
    assert isinstance(ent.end, int)
    assert 0 <= ent.score <= 1.0


def test_custom_string_placeholder(redactor):
    """Verify that a simple string placeholder works correctly."""
    text = "Alice Smith's number is 555-0101"
    redacted = redactor.redact(text, placeholder="REDACTED")

    assert redacted.count("REDACTED") == 2
    assert "Alice" not in redacted


def test_custom_callable_placeholder(redactor):
    """Verify that a callable placeholder is executed correctly."""

    def custom_logic(label, text):
        return f"<{label}>"

    text = "Alice Smith"
    redacted = redactor.redact(text, placeholder=custom_logic)
    assert redacted == "<private_person>"


def test_redact_with_details(redactor):
    """Verify the convenience method returns both text and list."""
    text = "Alice Smith"
    redacted, entities = redactor.redact_with_details(text)

    assert isinstance(redacted, str)
    assert len(entities) == 1
    assert entities[0].text == "Alice Smith"


def test_empty_string_behavior(redactor):
    """Ensure the library handles empty input gracefully."""
    assert redactor.detect("") == []
    assert redactor.redact("") == ""


def test_mode_switching_does_not_crash(redactor):
    """Ensure we can switch between modes without errors."""
    text = "Contact Alice."
    for mode in DecodingMode:
        redactor.redact(text, mode=mode)


def test_whitespace_trimming(redactor):
    """
    Verify that our custom _trim_whitespace_from_offsets works.
    The tokenizer often includes leading spaces in spans.
    """
    text = "Name: Alice Smith"
    # Manual check of the internal helper
    start = 5  # " Alice Smith"
    end = 17
    new_start, new_end = redactor._trim_whitespace_from_offsets(text, start, end)
    assert text[new_start:new_end] == "Alice Smith"
    assert new_start == 6


@pytest.mark.parametrize(
    "input_text",
    [
        "!!!@@@###",  # Special characters
        "1234567890",  # Numbers only
        "\n\t  \n",  # Whitespace only
    ],
)
def test_edge_case_inputs(redactor, input_text):
    """Smoke test for various atypical string inputs."""
    result = redactor.redact(input_text)
    assert isinstance(result, str)


def test_redact_file_logic(redactor, tmp_path):
    """Verify redact_file correctly reads and redacts a file."""
    p = tmp_path / "test.txt"
    p.write_text("My name is Alice Smith.", encoding="utf-8")

    redacted = redactor.redact_file(p)
    assert "Alice" not in redacted
    assert "[PRIVATE_PERSON]" in redacted

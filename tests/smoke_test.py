import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from privacy_filter_redactor import PIIRedactor


def main():
    print("Initializing PIIRedactor...")
    redactor = PIIRedactor()

    text = "Hello, my name is Jason Statham. My email is jason.statham@example.com and my phone number is 555-0199. I live at 123 Main St, New York, NY."  # noqa: E501

    print(f"\nOriginal text: {text}")

    print("\nDetecting entities...")
    entities = redactor.detect(text)
    for ent in entities:
        print(
            f" - Found {ent.label}: '{ent.text}' at {ent.start}:{ent.end} (score: {ent.score:.4f})"
        )

    print("\nRedacting with default placeholders:")
    redacted = redactor.redact(text)
    print(redacted)

    print("\nRedacting with custom string '***':")
    redacted_custom = redactor.redact(text, placeholder="***")
    print(redacted_custom)

    print("\nRedacting with custom callable (hashing):")

    def hash_placeholder(label, text):
        import hashlib

        return f"[{label}:{hashlib.md5(text.encode()).hexdigest()[:6]}]"

    redacted_callable = redactor.redact(text, placeholder=hash_placeholder)
    print(redacted_callable)

    print("\nSmoke test complete!")


if __name__ == "__main__":
    main()

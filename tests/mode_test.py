import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from privacy_filter_redactor import PIIRedactor

def main():
    print("Initializing PIIRedactor...")
    redactor = PIIRedactor()
    
    text = "The user mentioned Jason Statham and also referenced a specific key like 'sk-1234567890abcdef'."
    
    print(f"\nOriginal text: {text}")
    
    for mode in ["high_precision", "balanced", "high_recall"]:
        print(f"\n--- Testing mode: {mode} ---")
        entities = redactor.detect(text, mode=mode)
        print(f"Detected {len(entities)} entities:")
        for ent in entities:
            print(f" - [{ent.label}] '{ent.text}' (score: {ent.score:.4f})")
        
        redacted = redactor.redact(text, mode=mode)
        print(f"Redacted: {redacted}")

    print("\nMode switching test complete!")

if __name__ == "__main__":
    main()

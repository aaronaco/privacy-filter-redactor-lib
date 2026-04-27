from dataclasses import dataclass


@dataclass(frozen=True)
class DetectedEntity:
    """Represents a PII entity detected in text.

    Attributes:
        label: The category of PII (e.g., 'private_person', 'private_email').
        start: The starting character index in the original text.
        end: The ending character index in the original text.
        text: The actual text content of the entity.
        score: The confidence score or logit path probability.
    """
    label: str
    start: int
    end: int
    text: str
    score: float

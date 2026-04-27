from collections.abc import Callable
from pathlib import Path
from typing import Final

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .decoder import BIOTag, DecodingMode, ViterbiCRFDecoder
from .entities import DetectedEntity

# API Constants
DEFAULT_MODEL_ID: Final[str] = "openai/privacy-filter"


class PIIRedactor:
    """
    Main API for PII detection and redaction using OpenAI's privacy-filter model.

    This class handles the lifecycle of the transformer model, including tokenization,
    MoE inference, Viterbi decoding for BIOES consistency, and character-level
    string surgery for redaction.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        default_mode: DecodingMode = DecodingMode.BALANCED,
    ):
        """
        Initializes the redactor and moves the model to the target device.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.default_mode = default_mode

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_id, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

        # Decoders are pre-initialized to allow rapid mode-switching without overhead.
        self.decoders = {
            mode: ViterbiCRFDecoder(self.model.config.id2label, mode=mode) for mode in DecodingMode
        }

    def _process(self, text: str, mode: DecodingMode | None = None) -> list[DetectedEntity]:
        """
        Runs the full detection pipeline: tokenization, inference, and Viterbi decoding.
        """
        if not text:
            return []

        active_mode = mode or self.default_mode
        active_decoder = self.decoders.get(active_mode, self.decoders[self.default_mode])

        # Tokenize with offset mapping for character-level alignment
        encoding = self.tokenizer(
            text, return_offsets_mapping=True, return_tensors="pt", truncation=False
        ).to(self.device)

        offset_mapping = encoding.pop("offset_mapping")[0].cpu().numpy()

        with torch.no_grad():
            outputs = self.model(**encoding)
            token_logits = outputs.logits[0]

        best_label_indices = active_decoder.decode(token_logits)

        return self._map_indices_to_entities(text, best_label_indices, offset_mapping, token_logits)

    def _map_indices_to_entities(
        self,
        original_text: str,
        label_indices: list[int],
        offsets: list[tuple[int, int]],
        logits: torch.Tensor,
    ) -> list[DetectedEntity]:
        """
        Converts decoded label indices back into structured DetectedEntity objects.
        """
        detected_entities: list[DetectedEntity] = []
        partial_entity_metadata = None

        for i, label_id in enumerate(label_indices):
            label_name = self.model.config.id2label[label_id]
            start_char, end_char = offsets[i]

            # Skip special tokens (BOS/EOS/PAD) typically mapped to 0,0
            if start_char == 0 and end_char == 0 and i != 0:
                continue

            # Special check for leading BOS tokens that might have whitespace
            if i == 0 and original_text[0:end_char].strip() == "" and start_char == 0:
                if start_char == 0 and end_char == 0:
                    continue

            tag_part = label_name[0] if label_name != BIOTag.OUTSIDE else BIOTag.OUTSIDE
            category_name = label_name[2:] if label_name != BIOTag.OUTSIDE else None

            token_score = torch.softmax(logits[i], dim=-1)[label_id].item()

            if tag_part == BIOTag.SINGLE:
                start, end = self._trim_whitespace_from_offsets(
                    original_text, int(start_char), int(end_char)
                )
                detected_entities.append(
                    DetectedEntity(
                        label=category_name,
                        start=start,
                        end=end,
                        text=original_text[start:end],
                        score=token_score,
                    )
                )
                partial_entity_metadata = None

            elif tag_part == BIOTag.BEGIN:
                partial_entity_metadata = {
                    "label": category_name,
                    "start": int(start_char),
                    "end": int(end_char),
                    "token_scores": [token_score],
                }

            elif (
                tag_part == BIOTag.INSIDE
                and partial_entity_metadata
                and category_name == partial_entity_metadata["label"]
            ):
                partial_entity_metadata["end"] = int(end_char)
                partial_entity_metadata["token_scores"].append(token_score)

            elif (
                tag_part == BIOTag.END
                and partial_entity_metadata
                and category_name == partial_entity_metadata["label"]
            ):
                partial_entity_metadata["end"] = int(end_char)
                partial_entity_metadata["token_scores"].append(token_score)

                final_start, final_end = self._trim_whitespace_from_offsets(
                    original_text, partial_entity_metadata["start"], partial_entity_metadata["end"]
                )

                mean_confidence = sum(partial_entity_metadata["token_scores"]) / len(
                    partial_entity_metadata["token_scores"]
                )

                detected_entities.append(
                    DetectedEntity(
                        label=partial_entity_metadata["label"],
                        start=final_start,
                        end=final_end,
                        text=original_text[final_start:final_end],
                        score=mean_confidence,
                    )
                )
                partial_entity_metadata = None
            else:
                partial_entity_metadata = None

        return detected_entities

    def _trim_whitespace_from_offsets(self, text: str, start: int, end: int) -> tuple[int, int]:
        """
        Adjusts span boundaries to exclude leading/trailing whitespace captured by the tokenizer.
        """
        extracted_text = text[start:end]
        stripped_text = extracted_text.strip()

        if not stripped_text:
            return start, end

        actual_start = start + extracted_text.find(stripped_text)
        actual_end = actual_start + len(stripped_text)
        return actual_start, actual_end

    def detect(self, text: str, mode: DecodingMode | None = None) -> list[DetectedEntity]:
        """
        Returns a list of DetectedEntity objects containing labels and spans.
        """
        return self._process(text, mode)

    def redact(
        self,
        text: str,
        mode: DecodingMode | None = None,
        placeholder: str | Callable[[str, str], str] | None = None,
    ) -> str:
        """
        Returns a redacted version of the input string.
        """
        entities = self.detect(text, mode)
        return self._apply_redaction_to_text(text, entities, placeholder)

    def redact_with_details(
        self,
        text: str,
        mode: DecodingMode | None = None,
        placeholder: str | Callable[[str, str], str] | None = None,
    ) -> tuple[str, list[DetectedEntity]]:
        """
        Convenience method returning both redacted text and raw entity metadata.
        """
        entities = self.detect(text, mode)
        redacted_text = self._apply_redaction_to_text(text, entities, placeholder)
        return redacted_text, entities

    def redact_file(
        self,
        file_path: str | Path,
        mode: DecodingMode | None = None,
        placeholder: str | Callable[[str, str], str] | None = None,
    ) -> str:
        """
        Processes an entire file, leveraging the 128k context window.
        """
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
        return self.redact(content, mode, placeholder)

    def _apply_redaction_to_text(
        self,
        original_text: str,
        detected_entities: list[DetectedEntity],
        placeholder_logic: str | Callable[[str, str], str] | None = None,
    ) -> str:
        """
        Performs in-place string substitution using character offsets.
        Reverse-order iteration prevents index-shift during slicing.
        """
        if not detected_entities:
            return original_text

        # Reverse sort to maintain offset integrity while mutating the string
        sorted_entities = sorted(detected_entities, key=lambda entity: entity.start, reverse=True)

        text_buffer = list(original_text)
        for entity in sorted_entities:
            if callable(placeholder_logic):
                replacement_text = placeholder_logic(entity.label, entity.text)
            elif isinstance(placeholder_logic, str):
                replacement_text = placeholder_logic
            else:
                replacement_text = f"[{entity.label.upper()}]"

            text_buffer[entity.start : entity.end] = list(replacement_text)

        return "".join(text_buffer)

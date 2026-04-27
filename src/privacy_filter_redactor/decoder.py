from enum import StrEnum
from typing import Final

import torch


class DecodingMode(StrEnum):
    """Available sensitivity presets for the Viterbi decoder."""

    BALANCED = "balanced"
    HIGH_RECALL = "high_recall"
    HIGH_PRECISION = "high_precision"


class BIOTag(StrEnum):
    """BIOES boundary tags used by the model taxonomy."""

    BEGIN = "B"
    INSIDE = "I"
    OUTSIDE = "O"
    END = "E"
    SINGLE = "S"


# Decoder Constants
DEFAULT_BIAS_SHIFT: Final[float] = 5.0
START_TAG_PENALTY: Final[float] = 100.0  # Penalty for starting a sequence with I or E tags


class ViterbiCRFDecoder:
    """
    A Viterbi decoder for BIOES-tagged token classification.

    Optimizes sequence-level coherence by enforcing valid tag transitions
    and providing tunable sensitivity presets.
    """

    def __init__(
        self,
        id2label: dict[int, str],
        mode: DecodingMode = DecodingMode.BALANCED,
    ):
        self.id2label = id2label
        self.label2id = {label_name: label_id for label_id, label_name in id2label.items()}
        self.num_labels = len(id2label)
        self.mode = mode

        self.transition_biases = self._get_transition_biases(mode)
        self.transition_matrix = self._build_transition_matrix()

    def _get_transition_biases(self, mode: DecodingMode) -> dict[str, float]:
        """
        Calculates bias values that shift the decoder's sensitivity tradeoff.
        """
        if mode == DecodingMode.HIGH_RECALL:
            return {
                "background_stay": -DEFAULT_BIAS_SHIFT,
                "background_to_start": DEFAULT_BIAS_SHIFT,
                "inside_to_continue": DEFAULT_BIAS_SHIFT,
                "inside_to_end": -DEFAULT_BIAS_SHIFT,
                "end_to_background": -DEFAULT_BIAS_SHIFT,
                "end_to_start": DEFAULT_BIAS_SHIFT,
            }
        elif mode == DecodingMode.HIGH_PRECISION:
            return {
                "background_stay": DEFAULT_BIAS_SHIFT,
                "background_to_start": -DEFAULT_BIAS_SHIFT,
                "inside_to_continue": -DEFAULT_BIAS_SHIFT,
                "inside_to_end": DEFAULT_BIAS_SHIFT,
                "end_to_background": DEFAULT_BIAS_SHIFT,
                "end_to_start": -DEFAULT_BIAS_SHIFT,
            }

        # Balanced mode (default) uses zero bias
        bias_keys = [
            "background_stay",
            "background_to_start",
            "inside_to_continue",
            "inside_to_end",
            "end_to_background",
            "end_to_start",
        ]
        return dict.fromkeys(bias_keys, 0.0)

    def _build_transition_matrix(self) -> torch.Tensor:
        """
        Builds a [num_labels, num_labels] transition score matrix.
        Enforces BIOES tag consistency rules.
        """
        matrix = torch.full((self.num_labels, self.num_labels), -float("inf"))

        for i in range(self.num_labels):
            label_i = self.id2label[i]
            tag_i, cat_i = self._parse_label(label_i)

            for j in range(self.num_labels):
                label_j = self.id2label[j]
                tag_j, cat_j = self._parse_label(label_j)

                score = 0.0
                is_valid_transition = False

                # Rule 1: Stay in background
                if tag_i == BIOTag.OUTSIDE and tag_j == BIOTag.OUTSIDE:
                    score += self.transition_biases["background_stay"]
                    is_valid_transition = True

                # Rule 2: Start a new span from background
                elif tag_i == BIOTag.OUTSIDE and tag_j in [BIOTag.BEGIN, BIOTag.SINGLE]:
                    score += self.transition_biases["background_to_start"]
                    is_valid_transition = True

                # Rule 3: Continue or end a span
                elif (
                    tag_i == BIOTag.BEGIN
                    and tag_j in [BIOTag.INSIDE, BIOTag.END]
                    and cat_i == cat_j
                ):
                    is_continue = tag_j == BIOTag.INSIDE
                    bias_key = "inside_to_continue" if is_continue else "inside_to_end"
                    score += self.transition_biases[bias_key]
                    is_valid_transition = True

                # Rule 4: Continue or end an already internal span
                elif (
                    tag_i == BIOTag.INSIDE
                    and tag_j in [BIOTag.INSIDE, BIOTag.END]
                    and cat_i == cat_j
                ):
                    is_continue = tag_j == BIOTag.INSIDE
                    bias_key = "inside_to_continue" if is_continue else "inside_to_end"
                    score += self.transition_biases[bias_key]
                    is_valid_transition = True

                # Rule 5: Return to background after span closure
                elif tag_i in [BIOTag.END, BIOTag.SINGLE] and tag_j == BIOTag.OUTSIDE:
                    score += self.transition_biases["end_to_background"]
                    is_valid_transition = True

                # Rule 6: Immediate span handoff (e.g., E-Person to B-Email)
                elif tag_i in [BIOTag.END, BIOTag.SINGLE] and tag_j in [
                    BIOTag.BEGIN,
                    BIOTag.SINGLE,
                ]:
                    score += self.transition_biases["end_to_start"]
                    is_valid_transition = True

                if is_valid_transition:
                    matrix[i, j] = score

        return matrix

    def _parse_label(self, label: str) -> tuple[BIOTag, str | None]:
        """Parses a raw model label into its constituent tag and category."""
        if label == BIOTag.OUTSIDE:
            return BIOTag.OUTSIDE, None

        tag_part, cat_part = label.split("-", 1)
        return BIOTag(tag_part), cat_part

    def decode(self, token_logits: torch.Tensor) -> list[int]:
        """
        Performs Viterbi decoding to find the optimal label sequence.
        """
        sequence_length, num_labels = token_logits.shape
        if sequence_length == 0:
            return []

        # Ensure transition matrix is on the same device as input logits
        device_transitions = self.transition_matrix.to(token_logits.device)

        # Initialization: Sequences must not start with 'Inside' or 'End' tags.
        current_scores = token_logits[0].clone()
        for i in range(num_labels):
            tag, _ = self._parse_label(self.id2label[i])
            if tag in [BIOTag.INSIDE, BIOTag.END]:
                current_scores[i] -= START_TAG_PENALTY

        path_backpointers = torch.zeros(
            (sequence_length, num_labels), dtype=torch.long, device=token_logits.device
        )

        for t in range(1, sequence_length):
            # Compute path scores for all possible transitions to current step
            viterbi_scores = current_scores.unsqueeze(1) + device_transitions

            best_prev_scores, best_prev_labels = torch.max(viterbi_scores, dim=0)

            current_scores = best_prev_scores + token_logits[t]
            path_backpointers[t] = best_prev_labels

        # Reconstruct path via traceback
        best_path = [0] * sequence_length
        last_label_id = torch.argmax(current_scores).item()
        best_path[sequence_length - 1] = last_label_id

        for t in range(sequence_length - 1, 0, -1):
            last_label_id = path_backpointers[t, last_label_id].item()
            best_path[t - 1] = last_label_id

        return best_path

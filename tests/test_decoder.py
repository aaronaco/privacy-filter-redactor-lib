import torch
import pytest
from privacy_filter_redactor.decoder import ViterbiCRFDecoder, DecodingMode, BIOTag


@pytest.fixture
def id2label():
    """Simple id2label map for testing."""
    return {0: "O", 1: "B-test", 2: "I-test", 3: "E-test", 4: "S-test"}


def test_viterbi_coherence(id2label):
    """Verify that Viterbi prevents illegal transitions."""
    decoder = ViterbiCRFDecoder(id2label, mode=DecodingMode.BALANCED)

    # Logic: Even if 'I-test' has higher raw score, it shouldn't follow 'O'
    # Step 0: 'O' is high
    # Step 1: 'I-test' is high, but illegal from 'O'
    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0, 0.0, 0.0],  # Step 0: 'O' is clear winner
            [0.0, 0.0, 10.0, 0.0, 0.0],  # Step 1: 'I-test' is high, but illegal from 'O'
        ]
    )

    path = decoder.decode(logits)
    # Correct path should be O -> S-test or O -> O, but definitely not O -> I-test
    # Because O -> I-test is -inf in the transition matrix.
    labels = [id2label[i] for i in path]
    assert labels[1] != "I-test"


def test_mode_sensitivity_difference(id2label):
    """Verify that different modes actually shift the results on ambiguous logits."""
    # Step 0: 'O' is winner by 1.0
    # High recall (bias 10) should flip this easily
    logits = torch.tensor([
        [5.0, 4.0, 0.0, 0.0, 4.0],
    ])

    decoder_precision = ViterbiCRFDecoder(id2label, mode=DecodingMode.HIGH_PRECISION)
    decoder_recall = ViterbiCRFDecoder(id2label, mode=DecodingMode.HIGH_RECALL)

    path_p = decoder_precision.decode(logits)
    path_r = decoder_recall.decode(logits)

    # Recall should be more likely to pick the PII tag (1 or 4) than Precision
    assert id2label[path_p[0]] == "O"
    assert id2label[path_r[0]] != "O"


def test_split_label_logic(id2label):
    """Unit test for the internal label parsing."""
    decoder = ViterbiCRFDecoder(id2label)
    tag, cat = decoder._parse_label("B-private_person")
    assert tag == BIOTag.BEGIN
    assert cat == "private_person"

    tag_o, cat_o = decoder._parse_label("O")
    assert tag_o == BIOTag.OUTSIDE
    assert cat_o is None

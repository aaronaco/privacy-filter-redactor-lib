import pytest
from privacy_filter_redactor import PIIRedactor

@pytest.fixture(scope="session")
def redactor():
    """
    Provides a session-scoped PIIRedactor instance.
    Loading the model is expensive, so reuse it across tests.
    """
    return PIIRedactor()

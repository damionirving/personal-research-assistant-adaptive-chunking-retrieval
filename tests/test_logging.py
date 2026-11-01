import json
import logging

from pra_core.config import Settings
from pra_core.logging import configure_logging, reset_trace_id, set_trace_id


def test_json_logging_outputs_structured_record(capsys) -> None:
    configure_logging(Settings(environment="test", log_format="json"))
    logging.getLogger("pra.test").info("health_checked", extra={"request_id": "abc"})

    captured = capsys.readouterr()
    payload = json.loads(captured.err)

    assert payload["level"] == "INFO"
    assert payload["message"] == "health_checked"
    assert payload["request_id"] == "abc"
    assert payload["service"] == "personal-research-assistant"
    assert payload["environment"] == "test"


def test_json_logging_includes_trace_id_from_context(capsys) -> None:
    configure_logging(Settings(environment="test", log_format="json"))
    token = set_trace_id("trace-123")
    try:
        logging.getLogger("pra.test").info("request_started")
    finally:
        reset_trace_id(token)

    captured = capsys.readouterr()
    payload = json.loads(captured.err)

    assert payload["trace_id"] == "trace-123"

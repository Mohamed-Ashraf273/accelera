import pytest

from accelera.src.e2e.e2e import E2EBase
from accelera.src.e2e.text import E2E as TextE2E


class DummyE2E(E2EBase):
    def _run(self, content, config=None, graph=None):
        return content, config, graph


def test_base_forwards_content_config_and_graph():
    graph = object()

    result = DummyE2E()("content", config={"key": "value"}, graph=graph)

    assert result == ("content", {"key": "value"}, graph)


def test_text_e2e_rejects_unsupported_content():
    with pytest.raises(ValueError, match="Google Drive URL or a pandas DataFrame"):
        TextE2E()(object(), config={})

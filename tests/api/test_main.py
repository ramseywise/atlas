"""Tests for api/main.py route resource-lifecycle and error handling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_get_metric_closes_driver_on_exception():
    with patch("core.knowledge.graph.GraphDatabase") as mock_gdb:
        mock_driver = MagicMock()
        mock_gdb.driver.return_value = mock_driver
        with patch(
            "core.knowledge.graph.AtlasGraph.lookup_metric",
            side_effect=RuntimeError("boom"),
        ):
            resp = client.get("/knowledge/metric", params={"name": "burn_ratio"})
    assert resp.status_code == 500
    mock_driver.close.assert_called_once()


def test_get_segments_closes_driver_on_exception():
    with patch("core.knowledge.graph.GraphDatabase") as mock_gdb:
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_session.run.side_effect = RuntimeError("boom")
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)
        mock_gdb.driver.return_value = mock_driver
        resp = client.get("/segments")
    assert resp.status_code == 500
    mock_driver.close.assert_called_once()

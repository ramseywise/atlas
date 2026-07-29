"""Tests for AtlasGraph context manager behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.knowledge.graph import AtlasGraph


def _make_graph_with_mock_driver() -> tuple[AtlasGraph, MagicMock]:
    with patch("core.knowledge.graph.GraphDatabase") as mock_gdb:
        mock_driver = MagicMock()
        mock_gdb.driver.return_value = mock_driver
        g = AtlasGraph(uri="bolt://fake", user="u", password="p")
    return g, mock_driver


def test_context_manager_closes_on_success():
    g, mock_driver = _make_graph_with_mock_driver()
    with g:
        pass
    mock_driver.close.assert_called_once()


def test_context_manager_closes_on_exception():
    g, mock_driver = _make_graph_with_mock_driver()
    with pytest.raises(ValueError), g:
        raise ValueError("boom")
    mock_driver.close.assert_called_once()

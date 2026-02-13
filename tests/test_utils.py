from __future__ import annotations

from types import SimpleNamespace

import pytest

from multimodal_extraction.utils import (
    check_value_in_list,
    compute_stats,
    get_console_logger,
    mask_secret,
    remove_path_from_ref,
)


def test_remove_path_from_ref_returns_basename():
    assert remove_path_from_ref("a/b/c/file.pdf") == "file.pdf"
    assert remove_path_from_ref("file.pdf") == "file.pdf"


def test_check_value_in_list_ok_and_error():
    check_value_in_list("a", ["a", "b"])
    with pytest.raises(ValueError):
        check_value_in_list("x", ["a", "b"])


def test_compute_stats():
    docs = [
        SimpleNamespace(page_content="abcd"),
        SimpleNamespace(page_content="abcdef"),
        SimpleNamespace(page_content="abcdefgh"),
    ]
    mean_v, std_v, p75_v = compute_stats(docs)
    assert mean_v == 6
    assert std_v == 2
    assert p75_v == 7


def test_mask_secret():
    assert mask_secret(None) == ""
    assert mask_secret("abcd", keep=2) == "ab**"
    assert mask_secret("ab", keep=2) == "**"


def test_get_console_logger_singleton_handlers():
    logger1 = get_console_logger()
    logger2 = get_console_logger()
    assert logger1 is logger2
    assert len(logger1.handlers) == 1
    assert logger1.propagate is False

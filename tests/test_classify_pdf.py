from __future__ import annotations

from dataclasses import dataclass

from multimodal_extraction.pdf import classify_pdf as mod


@dataclass
class _FakeObj:
    data: dict

    def get(self, key, default=None):
        return self.data.get(key, default)

    def get_object(self):
        return self

    def items(self):
        return self.data.items()


class _FakePage:
    def __init__(self, text: str, has_image: bool):
        self._text = text
        self._has_image = has_image

    def extract_text(self):
        return self._text

    def get(self, key, default=None):
        if key != "/Resources":
            return default
        if not self._has_image:
            return {}
        xobj = _FakeObj(
            {
                "img1": _FakeObj(
                    {
                        "/Subtype": "/Image",
                    }
                )
            }
        )
        return {"/XObject": xobj}


class _FakeReader:
    def __init__(self, pages):
        self.pages = pages


def test_classify_pdf_strong_text_wins(monkeypatch):
    # All sampled pages have images, but text signal is strong -> TEXT_PDF.
    fake_pages = [_FakePage(text=("a" * 3000), has_image=True) for _ in range(3)]
    monkeypatch.setattr(mod, "PdfReader", lambda _: _FakeReader(fake_pages))

    cfg = mod.ClassifyConfig(sample_pages=3, strong_text_chars=5000)
    label, reason = mod.classify_pdf("dummy.pdf", cfg)

    assert label == "TEXT_PDF"
    assert "image_ratio" in reason


def test_classify_pdf_scanned(monkeypatch):
    # No text + many image pages -> SCANNED_PDF.
    fake_pages = [_FakePage(text="", has_image=True) for _ in range(5)]
    monkeypatch.setattr(mod, "PdfReader", lambda _: _FakeReader(fake_pages))

    cfg = mod.ClassifyConfig(sample_pages=5, scanned_if_image_pages_ratio_ge=0.6)
    label, _ = mod.classify_pdf("dummy.pdf", cfg)

    assert label == "SCANNED_PDF"


def test_classify_pdf_mixed(monkeypatch):
    # Some text and high image ratio without strong text -> MIXED_OR_UNKNOWN.
    fake_pages = [
        _FakePage(text="short text", has_image=True),
        _FakePage(text="tiny", has_image=True),
        _FakePage(text="", has_image=True),
    ]
    monkeypatch.setattr(mod, "PdfReader", lambda _: _FakeReader(fake_pages))

    cfg = mod.ClassifyConfig(
        sample_pages=3,
        min_text_chars_doc=5,
        min_text_chars_page=2,
        strong_text_chars=9999,
        scanned_if_image_pages_ratio_ge=0.6,
    )
    label, _ = mod.classify_pdf("dummy.pdf", cfg)

    assert label == "MIXED_OR_UNKNOWN"


def test_classify_pdf_read_error(monkeypatch):
    def _raise(_):
        raise RuntimeError("boom")

    monkeypatch.setattr(mod, "PdfReader", _raise)
    cfg = mod.ClassifyConfig()
    label, reason = mod.classify_pdf("dummy.pdf", cfg)

    assert label == "MIXED_OR_UNKNOWN"
    assert reason == "read_error=RuntimeError"

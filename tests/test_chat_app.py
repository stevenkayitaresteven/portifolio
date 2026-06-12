"""The chat API: send text and files, get moderated messages back.

Runs fully offline — HF backends disabled; a stub detector forces the
"explicit" path where needed. Skips cleanly when FastAPI isn't installed.
"""
import io

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from safety.types import Category, Region, Severity
from safety.chat import create_chat_app
from safety.multimodal import MultimodalConfig, MultimodalModerator


class AlwaysExplicit:
    name = "stub_explicit"
    available = True

    def analyze(self, image):
        h, w = image.shape[:2]
        return [Region(0, 0, w, h, "stub", 0.97,
                       Category.NUDITY, Severity.HIGH, source="stub")]


def make_client(explicit_images: bool = False) -> TestClient:
    cfg = MultimodalConfig()
    cfg.use_hf_text = cfg.use_hf_image = cfg.use_hf_audio = False
    cfg.check_video_audio = False
    cfg.safety.use_nudenet = False
    mod = MultimodalModerator(cfg)
    if explicit_images:
        mod.image.filter.detectors.append(AlwaysExplicit())
    return TestClient(create_chat_app(moderator=mod))


def png_bytes(w=64, h=48, value=128) -> bytes:
    ok, buf = cv2.imencode(".png", np.full((h, w, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


def structured_png(w=64, h=48) -> bytes:
    """An image with edges/contrast, so a blur measurably changes pixels."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:, : w // 2] = (30, 90, 220)
    img[h // 2:, w // 2:] = (200, 180, 40)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


# --- text messages ---------------------------------------------------------------

def test_clean_text_message():
    client = make_client()
    r = client.post("/chat/send", data={"text": "hello there"})
    assert r.status_code == 200
    (msg,) = r.json()["messages"]
    assert msg["kind"] == "text" and msg["text"] == "hello there"
    assert not msg["flagged"]


def test_curse_words_are_masked_and_flagged():
    client = make_client()
    r = client.post("/chat/send", data={"text": "where is my fucking order"})
    (msg,) = r.json()["messages"]
    assert msg["flagged"] and msg["action"] == "mask"
    assert "f******" in msg["text"]
    assert "fucking" not in msg["text"]
    assert "toxic" in msg["categories"]


def test_empty_message_is_rejected():
    assert make_client().post("/chat/send", data={"text": "  "}).status_code == 400


# --- file rules ----------------------------------------------------------------------

def test_only_one_file_at_a_time():
    client = make_client()
    r = client.post("/chat/send", files=[
        ("files", ("a.png", io.BytesIO(png_bytes()), "image/png")),
        ("files", ("b.png", io.BytesIO(png_bytes()), "image/png")),
    ])
    assert r.status_code == 400
    assert "one file" in r.json()["detail"]


def test_executable_upload_is_blocked_not_relayed():
    client = make_client()
    r = client.post("/chat/send", files=[
        ("files", ("evil.exe", io.BytesIO(b"MZ..."), "application/x-msdownload")),
    ])
    assert r.status_code == 200
    (msg,) = r.json()["messages"]
    assert msg["kind"] == "file"
    assert msg["flagged"] and msg["action"] == "block"
    assert "cybersecurity" in msg["categories"]
    assert msg["media"] is None


def test_unsupported_file_type_is_rejected():
    client = make_client()
    r = client.post("/chat/send", files=[
        ("files", ("data.xyz", io.BytesIO(b"\x00\x01\x02"), "application/octet-stream")),
    ])
    assert r.status_code == 400


def test_corrupt_image_is_rejected():
    client = make_client()
    r = client.post("/chat/send", files=[
        ("files", ("x.png", io.BytesIO(b"not a png"), "image/png")),
    ])
    assert r.status_code == 400


# --- images ------------------------------------------------------------------------

def test_clean_image_is_delivered_unflagged():
    client = make_client()
    r = client.post("/chat/send", files=[
        ("files", ("pic.png", io.BytesIO(png_bytes()), "image/png")),
    ])
    (msg,) = r.json()["messages"]
    assert msg["kind"] == "image" and not msg["flagged"]
    media = client.get(msg["media"])
    assert media.status_code == 200
    assert media.headers["content-type"] == "image/jpeg"


def test_explicit_image_is_delivered_blurred():
    client = make_client(explicit_images=True)
    original = structured_png()
    r = client.post("/chat/send",
                    data={"text": "check this out"},
                    files=[("files", ("pic.png", io.BytesIO(original), "image/png"))])
    (msg,) = r.json()["messages"]
    assert msg["kind"] == "image"
    assert msg["flagged"] and msg["action"] == "blur"
    assert "nudity" in msg["categories"]
    assert msg["caption"] == "check this out"      # text rode along as caption

    served = client.get(msg["media"]).content       # what the receiver sees
    img_orig = cv2.imdecode(np.frombuffer(original, np.uint8), cv2.IMREAD_COLOR)
    img_sent = cv2.imdecode(np.frombuffer(served, np.uint8), cv2.IMREAD_COLOR)
    assert img_sent is not None
    assert img_sent.shape == img_orig.shape
    # blurred: served pixels must differ from a JPEG of the original
    ok, orig_jpg = cv2.imencode(".jpg", img_orig, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_round = cv2.imdecode(orig_jpg, cv2.IMREAD_COLOR)
    assert float(np.mean(np.abs(img_sent.astype(int) - img_round.astype(int)))) > 1.0


def test_profane_caption_is_masked_on_media_message():
    client = make_client()
    r = client.post("/chat/send",
                    data={"text": "you bitch"},
                    files=[("files", ("pic.png", io.BytesIO(png_bytes()), "image/png"))])
    (msg,) = r.json()["messages"]
    assert msg["caption_flagged"]
    assert "b****" in msg["caption"]


# --- documents -----------------------------------------------------------------------

def test_profane_document_ships_censored_copy():
    client = make_client()
    body = b"hello\nthis shit broke again\n"
    r = client.post("/chat/send", files=[
        ("files", ("log.txt", io.BytesIO(body), "text/plain")),
    ])
    (msg,) = r.json()["messages"]
    assert msg["kind"] == "text" and msg["flagged"]
    assert "s***" in msg["text"]
    served = client.get(msg["media"])
    assert served.status_code == 200
    assert b"shit" not in served.content
    assert b"s***" in served.content


# --- history & misc ---------------------------------------------------------------------

def test_messages_endpoint_returns_history():
    client = make_client()
    client.post("/chat/send", data={"text": "one"})
    client.post("/chat/send", data={"text": "two"})
    history = client.get("/chat/messages").json()["messages"]
    assert [m["text"] for m in history] == ["one", "two"]
    assert all(m["ts"] for m in history)


def test_unknown_media_404():
    assert make_client().get("/chat/media/deadbeef").status_code == 404


def test_health_reports_moderators():
    h = make_client().get("/chat/health").json()
    assert {"text", "image", "audio", "video"} <= set(h)


def test_index_serves_chat_page():
    r = make_client().get("/")
    assert r.status_code == 200
    assert "Sentinel Chat" in r.text

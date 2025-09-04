import io
import numpy as np
import cv2

from app import create_app

def make_dummy_jpg(w=180, h=180):
    # simple gray image with a dark square
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    cv2.rectangle(img, (40,40), (140,140), (60,60,60), -1)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return io.BytesIO(buf.tobytes())

def test_predict_and_overlay(tmp_path, monkeypatch):
    # build app with a temp UPLOAD_FOLDER to avoid clutter
    app = create_app()
    app.config["UPLOAD_FOLDER"] = str(tmp_path)

    client = app.test_client()

    # 1) /predict with a dummy image
    data = {"image": (make_dummy_jpg(), "dummy.jpg")}
    resp = client.post("/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200, resp.data
    j = resp.get_json()
    assert "label" in j and "confidence" in j and "image_url" in j

    # 2) GET overlay
    overlay_url = j["image_url"]
    resp2 = client.get(overlay_url)
    assert resp2.status_code == 200
    assert resp2.mimetype == "image/jpeg"

def test_missing_file_returns_400():
    app = create_app()
    client = app.test_client()
    resp = client.post("/predict", data={}, content_type="multipart/form-data")
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "image is required"

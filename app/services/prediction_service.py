import os, uuid, numpy as np, tensorflow as tf, cv2
from ..ml.model_registry import ModelRegistry
from ..io.image_adapter import Cv2ImageAdapter as IO

def _cfg(cfg, key):
    return getattr(cfg, key, cfg.get(key))

class PredictionService:
    def __init__(self):
        self.registry = None 

    def _ensure_registry(self, cfg):
        if self.registry is None:
            self.registry = ModelRegistry.get(cfg)

    def predict_with_explain(self, file_storage, cfg):
        self._ensure_registry(cfg)

        img_size      = _cfg(cfg, "IMG_SIZE")
        upload_folder = _cfg(cfg, "UPLOAD_FOLDER")
        labels        = _cfg(cfg, "LABELS")

        # I/O (Adapter)
        src_path = IO.save_file(file_storage, upload_folder)
        orig_bgr = IO.read_bgr(src_path)
        x = IO.preprocess_bgr_to_tensor(orig_bgr, img_size)

        # Inference
        preds = self.registry.model.predict(x)[0]
        pred_idx = int(np.argmax(preds))
        label = labels[pred_idx]
        confidence = round(float(np.max(preds) * 100), 2)

        # Grad-CAM
        heatmap = self._grad_cam(self.registry.grad_model, x, pred_idx)

        # Overlay
        overlay = IO.overlay_heatmap(orig_bgr, heatmap)
        overlay_id = uuid.uuid4().hex
        overlay_path = os.path.join(upload_folder, f"{overlay_id}.jpg")
        cv2.imwrite(overlay_path, overlay)

        return {"label": label, "confidence": confidence, "overlay_id": overlay_id}

    def _grad_cam(self, grad_model, x, pred_index):
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(x)
            loss = preds[:, pred_index]
        grads = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_out[0]
        heatmap = tf.reduce_sum(conv_out * pooled, axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()

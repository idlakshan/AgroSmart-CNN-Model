import os, cv2, uuid

class Cv2ImageAdapter:
    
    @staticmethod
    def save_file(file_storage, folder: str) -> str:
        os.makedirs(folder, exist_ok=True)
        filename = f"{uuid.uuid4().hex}_{file_storage.filename}"
        path = os.path.join(folder, filename)
        file_storage.save(path)
        return path

    @staticmethod
    def read_bgr(path: str):
        return cv2.imread(path)

    @staticmethod
    def preprocess_bgr_to_tensor(bgr, img_size: int):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (img_size, img_size))
        x = rgb.astype("float32") / 255.0
        return x[None, ...] 

    @staticmethod
    def overlay_heatmap(orig_bgr, heatmap):
        h, w = orig_bgr.shape[:2]
        hm = cv2.resize(heatmap, (w, h))
        hm8 = (hm * 255).astype("uint8")
        colored = cv2.applyColorMap(hm8, cv2.COLORMAP_JET)
        return cv2.addWeighted(orig_bgr, 0.6, colored, 0.4, 0)

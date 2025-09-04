import os

class Config:
    IMG_SIZE = int(os.getenv("IMG_SIZE", 180))
    NUM_CLASSES = int(os.getenv("NUM_CLASSES", 5))
    WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "model/soil_model_weights_only.weights.h5")
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
    LABELS = {
        0: 'Clay soil',
        1: 'Loamy soil',
        2: 'Red soil',
        3: 'Sandy Loamy soil',
        4: 'Sandy soil'
    }

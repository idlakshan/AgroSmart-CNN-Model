import tensorflow as tf
from .model_factory import ModelFactory

def _cfg(cfg, key):
 
    return getattr(cfg, key, cfg.get(key))

class ModelRegistry:
    """Singleton: loads weights once, exposes model + grad_model."""
    _instance = None

    def __init__(self, cfg):
        self.img_size     = _cfg(cfg, "IMG_SIZE")
        self.num_classes  = _cfg(cfg, "NUM_CLASSES")
        self.weights_path = _cfg(cfg, "WEIGHTS_PATH")

        self.model = ModelFactory.build(self.img_size, self.num_classes)
        self.model.load_weights(self.weights_path)
        self.grad_model = self._build_grad_model()

    @classmethod
    def get(cls, cfg):
        if cls._instance is None:
            cls._instance = cls(cfg)
        return cls._instance

    def _build_grad_model(self):
        inputs = tf.keras.Input(shape=(self.img_size, self.img_size, 3))
        x = inputs
        last_conv = None
        for lyr in self.model.layers:
            x = lyr(x)
            if "Conv2D" in lyr.__class__.__name__:
                last_conv = x
        return tf.keras.Model(inputs, [last_conv, x])

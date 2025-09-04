import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, initializers

class ModelFactory:
    """Factory: builds CNN architecture (weights loaded elsewhere)."""
    @staticmethod
    def build(img_size: int, num_classes: int):
        He = initializers.HeNormal()
        l2 = regularizers.l2(5e-5)
        model = models.Sequential([
            layers.SeparableConv2D(32, 3, padding='same', activation='relu',
                                   depthwise_initializer=He, pointwise_initializer=He,
                                   depthwise_regularizer=l2, pointwise_regularizer=l2,
                                   input_shape=(img_size, img_size, 3)),
            layers.BatchNormalization(),
            layers.SeparableConv2D(32, 3, padding='same', activation='relu',
                                   depthwise_initializer=He, pointwise_initializer=He,
                                   depthwise_regularizer=l2, pointwise_regularizer=l2),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2), layers.Dropout(0.15),

            layers.SeparableConv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.SeparableConv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2), layers.Dropout(0.20),

            layers.SeparableConv2D(96, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.SeparableConv2D(96, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2), layers.Dropout(0.25),

            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.35),
            layers.Dense(96, activation='relu', kernel_initializer=He, kernel_regularizer=l2),
            layers.BatchNormalization(), layers.Dropout(0.45),
            layers.Dense(num_classes, activation='softmax', dtype='float32')
        ])
        _ = model(tf.zeros((1, img_size, img_size, 3))) 
        return model

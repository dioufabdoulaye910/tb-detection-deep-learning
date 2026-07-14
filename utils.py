import numpy as np
import tensorflow as tf
from PIL import Image


def load_model(model_path="tb_model.keras"):
    return tf.keras.models.load_model(
        model_path,
        compile=False
    )


def preprocess_image(image):
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.convert("RGB")
    image = image.resize((224, 224))

    img_array = np.asarray(image, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def predict(model, image):
    img = preprocess_image(image)
    prediction = model.predict(img, verbose=0)
    return float(prediction[0][0])
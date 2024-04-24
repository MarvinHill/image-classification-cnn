import tensorflow as tf
import numpy as np
import cv2
from PIL import Image


def main():
    trained_model = tf.keras.models.load_model('trained_model.keras')

    data_classes = np.load("classes.npy")
    print(data_classes)
    img = Image.open("img_1.png").resize((180, 180))
    # Convert to 3 shapes
    img = img.convert('RGB')
    img = np.array(img)

    result = trained_model.predict(img[None, :, :])
    max = np.argmax(result)
    character = data_classes[max]
    print(character)


if __name__ == "__main__":
    main()

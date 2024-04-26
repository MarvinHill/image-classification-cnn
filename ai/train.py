import tensorflow as tf
from os import listdir
from os.path import join, isdir
import numpy as np
import matplotlib.pyplot as plt


def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    print(tf.config.list_physical_devices())
    print("setup settings")
    BASE_PATH = './archive/simpsons_dataset/'

    BATCH_SIZE = 16
    IMG_HEIGHT = 180
    IMG_WIDTH = 180
    VALIDATION_SPLIT = 0.2
    EPOCHS = 15

    print("load class count")
    loadedfiles = [f for f in listdir(BASE_PATH) if isdir(join(BASE_PATH, f))]
    class_count = len(loadedfiles)
    print(loadedfiles)
    print("class: " + str(class_count))
    np.save("classes.npy", loadedfiles)

    print("load datasets 1")
    # load dataset
    traindataset = tf.keras.utils.image_dataset_from_directory(
        BASE_PATH,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        seed=123,
        batch_size=BATCH_SIZE,
        labels='inferred'
    )

    print("load datasets 2")
    valdataset = tf.keras.utils.image_dataset_from_directory(
        BASE_PATH,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        seed=123,
        batch_size=BATCH_SIZE,
        labels='inferred'
    )

    AUTOTUNE = tf.data.AUTOTUNE

    print("fetch datasets 1")
    traindataset = traindataset.cache().prefetch(buffer_size=AUTOTUNE)
    print("fetch datasets 2")
    valdataset = valdataset.cache().prefetch(buffer_size=AUTOTUNE)

    print("create model")
    # train model
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1. / 255),  # rgb value from 0 to 255 converts to 0 to 1
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(300, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(300, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(300, activation='relu'),
            tf.keras.layers.Dense(class_count, activation='relu'),
        ]
    )

    print("compile model")
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    print("fit model ")
    history = model.fit(
        traindataset,
        validation_data=valdataset,
        epochs=EPOCHS
    )

    plot_training_history(history, epochs=EPOCHS)

    # save model
    model.save('trained_model.keras')


def plot_training_history(history, epochs):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('accuracy.png')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('loss.png')
    plt.show()


if __name__ == "__main__":
    main()

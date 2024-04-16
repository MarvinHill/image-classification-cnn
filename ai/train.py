import tensorflow as tf
from os import listdir
from os.path import join, isdir

def main():
    print("setup settings")
    BASE_PATH = './archive/simpsons_dataset/'

    BATCH_SIZE = 32
    IMG_HEIGHT = 180
    IMG_WIDTH = 180
    VALIDATION_SPLIT = 0.2

    print("load class count")
    loadedfiles = [f for f in listdir(BASE_PATH) if isdir(join(BASE_PATH, f))]
    class_count = len(loadedfiles)
    print(loadedfiles)
    print("class: " + str(class_count))

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
            tf.keras.layers.Rescaling(1./255), # rgb value from 0 to 255 converts to 0 to 1
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(class_count),
        ]
    )
    

    print("compile model")
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    print("print summary")
    model.summary()
    
    print ("fit model ")
    model.fit(
        traindataset,
        validation_data=valdataset,
        epochs=30
    )

    # save model
    model.save('trained_model.keras')

if __name__ == "__main__":
    main();



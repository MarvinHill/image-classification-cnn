
def main():
    trained_model = tf.keras.models.load_model('trained_model.keras');

    trained_model.predict()
    # Todo - predict some test data
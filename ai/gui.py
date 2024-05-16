import os
import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf


class ImagePredictorApp:
    def __init__(self, master):
        self.master = master
        master.title("Bildauswahl und Vorhersage")
        master.geometry("1000x1000")

        self.selected_image_label = Label(master)
        self.selected_image_label.pack(pady=20)

        self.predicted_characters_label = Label(master, font=("Arial", 18))
        self.predicted_characters_label.pack(pady=10)

        self.select_button = tk.Button(master, text="Bild auswählen", command=self.select_image)
        self.select_button.pack(pady=20)

    def select_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.display_selected_image(file_path)
            characters_with_probabilities = self.predict_top_three(file_path)
            self.display_predicted_characters(characters_with_probabilities)

    def display_selected_image(self, file_path):
        image = Image.open(file_path)
        photo = ImageTk.PhotoImage(image)
        self.selected_image_label.config(image=photo)
        self.selected_image_label.image = photo

    def predict_top_three(self, image_path):
        trained_model = tf.keras.models.load_model('trained_model.keras')

        data_classes = np.load("classes.npy")
        img = Image.open(image_path).resize((180, 180))
        img = img.convert('RGB')
        img = np.array(img)

        result = trained_model.predict(img[None, :, :])
        print(result)
        top_three_indices = np.argsort(result[0])[-3:]
        top_three_probabilities = result[0][top_three_indices]
        top_three_characters = data_classes[top_three_indices]
        print(top_three_characters)

        characters_with_probabilities = zip(top_three_characters, top_three_probabilities)
        return characters_with_probabilities

    def display_predicted_characters(self, characters_with_probabilities):
        characters_with_probabilities = reversed(list(characters_with_probabilities))
        predicted_text = "\n".join([f"{character}: {probability*100:.2f}%" for character, probability in characters_with_probabilities])
        self.predicted_characters_label.config(text=f"{predicted_text}")


def main():
    if not os.path.exists("pictures"):
        os.makedirs("pictures")

    root = tk.Tk()
    app = ImagePredictorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

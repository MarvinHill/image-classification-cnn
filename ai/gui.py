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

        self.predicted_character_label = Label(master, font=("Arial", 18))
        self.predicted_character_label.pack(pady=10)

        self.select_button = tk.Button(master, text="Bild auswählen", command=self.select_image)
        self.select_button.pack(pady=20)

    def select_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.display_selected_image(file_path)
            character = self.predict(file_path)
            self.predicted_character_label.config(text=f"Vorhersage: {character}")

    def display_selected_image(self, file_path):
        image = Image.open(file_path)
        photo = ImageTk.PhotoImage(image)
        self.selected_image_label.config(image=photo)
        self.selected_image_label.image = photo

    def predict(self, image_path):
        trained_model = tf.keras.models.load_model('trained_model.keras')

        data_classes = np.load("classes.npy")
        print(data_classes)
        img = Image.open(image_path).resize((180, 180))
        # Convert to 3 shapes
        img = img.convert('RGB')
        img = np.array(img)

        result = trained_model.predict(img[None, :, :])
        max = np.argmax(result)
        character = data_classes[max]
        print(character)
        return character


def main():
    # Erstelle den Ordner "pictures", wenn er nicht existiert
    if not os.path.exists("pictures"):
        os.makedirs("pictures")

    root = tk.Tk()
    app = ImagePredictorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

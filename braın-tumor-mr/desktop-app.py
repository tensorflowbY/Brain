from tkinter import *
import random
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

model=load_model("base_model_vgg16_1.keras")

def predict():
    test_path = "C:\\Users\\batuy\\Documents\\VS Code\\braın-tumor-mr\\Data\\Testing\\"

    test_path_list = os.listdir(test_path)
    random_dir = random.choice(test_path_list)
    main_dir = os.path.join(test_path, random_dir)
    
    img_dir = os.listdir(main_dir)

    images = []

    for img_1 in img_dir:
        if img_1.endswith(".jpg") or img_1.endswith(".png"):
            images.append(img_1)

    random_img = random.sample(images, 1)

    for img_2 in random_img:
        resim_yolu = os.path.join(main_dir, img_2)
        img__ = Image.open(resim_yolu).convert('RGB')  
        img_resized = img__.resize((52, 52))  
        img_array = np.array(img_resized) / 255.0  
        img_array = np.expand_dims(img_array, axis=0)  

        # Tahmin yapma
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)  
        confidence = np.max(prediction) * 100
        true_class = os.path.basename(os.path.dirname(resim_yolu))

        """
        0 = glioma
        1 = meningioma
        2 = notumor
        3 = pituitary
        """

        img_resized = img_resized.resize((250,250))

        plt.imshow(img_resized)
        plt.title(f"""True: {true_class}, Predicted: {predicted_class}, 
                        İmage: {img_2}, Conf: {confidence:.2f}% """)
        
        plt.show()

root = Tk()
root.geometry("500x500")
root.title('Braın Tumor MR Predict')

text = Label(root, text='Braın Tumor MR Predict')
text.pack()

button = Button(root, text='Braın Photos', width=25, command=predict)
button.pack()

root.mainloop()
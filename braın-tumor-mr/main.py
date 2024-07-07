# %%
import tensorflow
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import pandas as np
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
# %%
import random
from PIL import Image
import matplotlib.pyplot as plt

target_path= "C:\\Users\\batuy\\Documents\\VS Code\\braın-tumor-mr\\Prediction check images\\Prediction check images\\Prediction check images\\"

list_dir = os.listdir(target_path)

target_size = (104, 104)

images = []

for img_1 in list_dir:
    if img_1.endswith(".png") or img_1.endswith(".jpg"):
        images.append(img_1)

random_img = random.sample(images, 3)

for img_2 in random_img:
    img_way = os.path.join(target_path, img_2)
    img__ = Image.open(img_way)
    img_resized = img__.resize(target_size)
    plt.imshow(img_resized)
    plt.title(img_2)
    plt.show()
# %%
target_size_default = (52,52,3)
# %%
train_data = ImageDataGenerator(rescale = 1./255, 
                                validation_split = 0.2, 
                                horizontal_flip = True, 
                                zoom_range = 0.2)

test_data = ImageDataGenerator(rescale = 1./255)
# %%
train_dir = "C:\\Users\\batuy\\Documents\\VS Code\\braın-tumor-mr\\Data\\Training"
test_dir = "C:\\Users\\batuy\\Documents\\VS Code\\braın-tumor-mr\\Data\\Testing"

train_data_dir = train_data.flow_from_directory(directory=train_dir,
                                                target_size=(target_size_default[0], target_size_default[1]),
                                                class_mode = "categorical",
                                                subset="training")

test_data_dir = test_data.flow_from_directory(directory=test_dir,
                                                target_size=(target_size_default[0], target_size_default[1]),
                                                class_mode = "categorical")

val_data_dir = train_data.flow_from_directory(directory=train_dir,
                                                target_size=(target_size_default[0], target_size_default[1]),
                                                class_mode = "categorical",
                                                subset="validation")
# %%
base_model = VGG16(weights="imagenet", include_top = False, input_shape=target_size_default)

for layer in base_model.layers:
    layer.trainable = False
# %%
x = base_model.output
x = Flatten()(x)
x = Dense(128,activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(32, activation="relu")(x)
prediction = Dense(4, activation="softmax")(x)
# %%
model = Model(inputs=base_model.input, outputs=prediction)

# model.load_weights("ednet_weights_optim.h5")
# %%
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# %%
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)

early_stopping = EarlyStopping(monitor="val_loss", patience=5, 
                               verbose=1, min_delta=2)

callbacks=[reduce_lr, early_stopping]
# %%
model.fit(train_data_dir, validation_data=val_data_dir, epochs=20, batch_size=34, callbacks=callbacks)
# %%
predict=model.evaluate(test_data_dir)
print(f"value : {predict}")
# %%
model.save("base_model_vgg16_1.keras")
# %%
from tensorflow.keras.models import load_model
# %%
vgg_16 = load_model("base_model_vgg16_1.keras")
# %%
predict=vgg_16.evaluate(test_data_dir)
print(f"value : {predict}")

# %%
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.models import load_model

model = load_model("base_model_vgg16_1.keras")

# Test veri setinden rastgele bir görüntü alınması
img_path = "C:\\Users\\batuy\\Documents\\VS Code\\braın-tumor-mr\\Data\\Testing\\Glioma_2.jpg"

# Sınıf etiketleri
class_labels = train_data_dir.class_indices
class_labels = {v: k for k, v in class_labels.items()}  # Sınıf indeksleri ve etiketleri eşleştir

# Resmi yükleyip boyutlandırma
img = image.load_img(img_path, target_size=(target_size_default[0], target_size_default[1]))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Tahmin
prediction = model.predict(img_array)
predicted_class_index = np.argmax(prediction)
predicted_class = class_labels.get(predicted_class_index, "Unknown")


# Gerçek sınıfı almak için test veri setinden etiketi çıkarın
true_class = os.path.basename(os.path.dirname(img_path))

# Resmi gösterme
plt.imshow(img)
plt.title(f"Predicted: {predicted_class}, True: {true_class}")
plt.axis('off')
plt.show()
# %%

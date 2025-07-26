# model_train.py
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import cv2

dataset_dir = "dataset"
categories = ["with_mask", "without_mask"]
data = []
labels = []

# Load and preprocess the data
for category in categories:
    path = os.path.join(dataset_dir, category)
    label = categories.index(category)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (224, 224))
        image = preprocess_input(image)
        data.append(image)
        labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)
labels = to_categorical(labels)

# Split
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2)

# Model: MobileNetV2 for transfer learning
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten()(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze base layers
for layer in baseModel.layers:
    layer.trainable = False

# Compile
model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=1e-4), metrics=["accuracy"])

# Train
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=5, batch_size=32)

# Save model
#model.save("mask_detector.model")
model.save("mask_detector.h5")
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input

# Check if data directories exist
val_dir = "./data/test"
train_dir = "./data/train"
for directory in [val_dir, train_dir]:
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        exit()

# Define data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

class Model:

    def __init__(self):
        self.model = Sequential()

        self.num_train = 4817
        self.num_val = 533
        self.batch_size = 64
        self.num_epoch = 50

        self.init_model()

    def init_model(self):
        self.model.add(Input(shape=(224, 224, 1)))
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(7, activation='softmax'))

        model_weights_path = "model.weights.h5"
        if os.path.isfile(model_weights_path):
            self.model.load_weights(model_weights_path)
        else:
            self.train()

    def plot_model_history(self, model_history):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        # Summarize history for accuracy
        axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'])
        axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'])

        axs[0].set_title('Model Accuracy')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].set_xticks(np.arange(1, len(model_history.history['accuracy']) + 1), len(model_history.history['accuracy'])/10)

        axs[0].legend(['train', 'val'], loc='best')

        # Summarize history for loss
        axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'])
        axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'])

        axs[1].set_title('Model Loss')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_xticks(np.arange(1, len(model_history.history['loss']) + 1), len(model_history.history['loss'])/10)

        axs[1].legend(['train', 'val'], loc='best')

        fig.savefig('plot.png')

        plt.show()

    def train(self):
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
        model_info = self.model.fit(
            train_generator,
            steps_per_epoch=self.num_train // self.batch_size,
            epochs=self.num_epoch,
            validation_data=validation_generator,
            validation_steps=self.num_val // self.batch_size
        )

        self.model.save_weights("model.weights.h5")
        self.model.save("model_full.h5")
        self.plot_model_history(model_info)

    def predict(self, cropped_img):
        prediction = self.model.predict(cropped_img)
        return int(np.argmax(prediction))

# Assuming this Model class is what we use to instantiate the model
# Instantiate the object
emotion_model = Model()
# To make a prediction using the predict method, pass the processed image
# prediction = emotion_model.predict(processed_image)

import os
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D,
    GlobalAveragePooling2D,
    Dense, Dropout, Input,Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ZIP_PATH = os.path.join(BASE_DIR,'mask_data.zip')
EXTRACT_PATH = os.path.join(BASE_DIR,'00_mask_dataset')


if not os.path.exists(EXTRACT_PATH):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_PATH)
        print('Connected Successfully')


# Training data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3
)


train_set = train_datagen.flow_from_directory(
    os.path.join(EXTRACT_PATH,'FMD_DATASET'),
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    color_mode='rgb',
    subset='training'
)
test_set = val_datagen.flow_from_directory(
    os.path.join(EXTRACT_PATH, 'FMD_DATASET'),
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    color_mode='rgb',
    subset = 'validation'
)


model = Sequential([
    Input(shape=(64, 64, 3)),

    Conv2D(32, 3, activation='relu', padding='same'),
    MaxPooling2D(pool_size=2,strides=2),

    Conv2D(64, 3, activation='relu', padding='same'),
    MaxPooling2D(pool_size=2, strides=2),

    Conv2D(64, 3, activation='relu', padding='same'),

    GlobalAveragePooling2D(),

    Dense(64, activation='relu'),
    Dropout(0.5),

    Dense(3, activation='softmax')
])


model.compile(optimizer=Adam(learning_rate=0.0005),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()


model_path = "last_mask_model_cnn.keras"


early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
model.fit(
    train_set,
    validation_data=test_set,
    epochs=50,
    callbacks = [early_stop]
)

model.save(model_path)
print(f'Model {model_path} Saved Successfully!')
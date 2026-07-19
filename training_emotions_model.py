import os
import zipfile

import tensorflow as tf
from keras_facenet import FaceNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class ANN:
    def __init__(self):
        self.model = None

    def compile_model(self):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

class CNN_OBJECT(ANN):
    IMG_SIZE = (160, 160)
    BATCH_SIZE = 16
    def __init__(self,model_extract_name,zip_file_path):
        super().__init__()

        #Model Info
        self.model_extract_name = model_extract_name

        self.base_model = None
        self.num_classes = None
        self.class_indices= None
        self.class_weight = None
        self.model_path = None

        #Zip Info
        self.zip_file_path = zip_file_path
        self.extract_path = os.path.join(os.path.dirname(self.zip_file_path),"00_emotions_dataset")

        #Train + Test handling
        self.train_data, self.test_data = self.handle_zip_train_test()

    def handle_zip_train_test(self):
        data_path = self.extract_zip()
        try:
            train_datagen = ImageDataGenerator(
                preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
                validation_split=0.3,
                width_shift_range=0.15,
                height_shift_range=0.15,
                brightness_range=[0.8, 1.2],
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True
            )

            val_datagen = ImageDataGenerator(
                preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
                validation_split=0.3
            )

            train_set = train_datagen.flow_from_directory(
                data_path,
                target_size=self.IMG_SIZE,
                batch_size=self.BATCH_SIZE,
                class_mode='categorical',
                subset='training',
                seed=42

            )

            test_set = val_datagen.flow_from_directory(
                data_path,
                target_size=self.IMG_SIZE,
                batch_size=self.BATCH_SIZE,
                class_mode='categorical',
                subset='validation',
                seed=42

            )

            self.num_classes = len(train_set.class_indices)
            self.class_indices = train_set.class_indices

            # Handle class imbalance (common in emotion datasets
            classes = np.unique(train_set.classes)
            weights = compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=train_set.classes
            )
            self.class_weight = dict(zip(classes, weights))

            print("Classes:", self.class_indices)
            print("Num classes:", self.num_classes)
            print("Class weights:", self.class_weight)

            return train_set, test_set

        except Exception as e:
            raise RuntimeError(f"ERROR: {e}")

    def extract_zip(self):
        if not os.path.exists(self.extract_path):
            if not os.path.exists(self.zip_file_path):
                raise FileNotFoundError(f"Zip file not found: {self.zip_file_path}")
            with zipfile.ZipFile(self.zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_path)
                print("Zip Opened Successfully")

        try:
            sub_dirs = os.listdir(self.extract_path)
            if len(sub_dirs) == 1:
                inner_path = os.path.join(self.extract_path, sub_dirs[0])
                if os.path.isdir(inner_path):
                    return inner_path
            return self.extract_path

        except Exception as e:
            raise RuntimeError(f"ERROR: {e}")

    def train_model(self):
        # Fit Model + Save with Fine-Tuning
        self.model = self.build_base_model()
        super().compile_model()
        self.model.summary()

        print("Classes:", self.class_indices)
        print("Num classes:", self.num_classes)

        # -------- Callbacks (initial training) -------- #
        es = EarlyStopping(monitor='val_loss',
            patience=12,
            min_delta=1e-4,
            restore_best_weights=True)

        self.model_path = os.path.join(os.getcwd(), self.model_extract_name)

        cp = ModelCheckpoint(self.model_extract_name,
            monitor='val_loss',
            save_best_only=True)

        rlr = ReduceLROnPlateau(monitor='val_loss',
            factor=0.3,
            patience=5,
            min_lr=1e-6)

        try:
            # -------- Training (frozen base) -------- #
            history = self.model.fit(
                self.train_data,
                validation_data=self.test_data,
                epochs=40,
                callbacks=[cp, es, rlr],
                class_weight=self.class_weight
            )

             # -------- Fine tuning -------- #
            self.base_model.trainable = True
            for layer in self.base_model.layers[:-30]:
                layer.trainable = False

            for layer in self.base_model.layers:
                if isinstance(layer, BatchNormalization):
                    layer.trainable = False

            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            fine_tune_cp = ModelCheckpoint(
                self.model_extract_name,
                monitor='val_loss',
                save_best_only=True
            )

            fine_history = self.model.fit(
                self.train_data,
                validation_data=self.test_data,
                epochs=50,
                callbacks=[fine_tune_cp, es, rlr],
                class_weight=self.class_weight
            )

            # Ensure the final (best-restored) weights are persisted.
            self.model.save(self.model_path)

            return history, fine_history

        except Exception as e:
            raise RuntimeError(f"ERROR during training: {e}")

    def build_base_model(self):
        facenet_full = FaceNet().model
        feature_map = facenet_full.get_layer('add_20').output

        self.base_model = Model(
            inputs=facenet_full.input,
            outputs=feature_map,
            name='facenet_backbone'
        )
        self.base_model.trainable = False

        x = self.base_model.output
        x = GlobalAveragePooling2D()(x)
        x = BatchNormalization()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        return Model(inputs=self.base_model.input, outputs=outputs)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
zip_path = os.path.join(BASE_DIR, 'emotions_data.zip')

cnn = CNN_OBJECT(model_extract_name="face_net_emotions_6_classes.keras",zip_file_path=zip_path)
cnn.train_model()
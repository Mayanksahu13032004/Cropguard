# In train_model.py

# import os
import os
# ✅ FIX: Change 'tensorflow.keras' to just 'keras' for these specific utilities.
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
# ... (rest of your script remains the same) ...
# --------------------------
# Paths
# --------------------------
DATASET_PATH = "dataset"   # dataset folder inside backend
MODEL_PATH = "model.h5"
LABELS_PATH = "labels.txt"

# --------------------------
# Image Data Generator
# --------------------------
img_size = (128, 128)  # Resize all images
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,     # 80% train, 20% validation
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# --------------------------
# Save class labels
# --------------------------
labels = list(train_gen.class_indices.keys())
with open(LABELS_PATH, "w") as f:
    for lbl in labels:
        f.write(lbl + "\n")
print("✅ Saved labels:", labels)

# --------------------------
# Build CNN Model
# --------------------------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    
    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(len(labels), activation="softmax")
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# --------------------------
# Train the model
# --------------------------
checkpoint = ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,    # increase for better accuracy
    callbacks=[checkpoint]
)

print("✅ Training complete! Best model saved as:", MODEL_PATH)

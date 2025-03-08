import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# Enable GPU memory growth to prevent TensorFlow from allocating all GPU memory at once
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"Found {len(physical_devices)} GPU(s)")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print(f"Memory growth enabled for {device}")
else:
    print("No GPU found, using CPU instead")

# Define the input shape (30 frames, 224x224 resolution, 3 channels)
input_shape = (30, 224, 224, 3)

# Function to convert video to frames with preprocessing
def video_to_frames(video_path, max_frames=30, img_size=(224, 224)):
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize and normalize pixel values to [-1, 1]
        frame = cv2.resize(frame, img_size)
        frame = frame / 127.5 - 1.0
        frames.append(frame)
        
    cap.release()
    
    # Pad or truncate frames to max_frames
    if len(frames) < max_frames:
        # Use the last frame for padding to maintain motion context
        last_frame = frames[-1] if frames else np.zeros((img_size[1], img_size[0], 3))
        frames += [last_frame] * (max_frames - len(frames))
    else:
        frames = frames[:max_frames]
        
    return np.array(frames)

# Improved data generator with data augmentation
def data_generator(folder_path, batch_size=16, max_frames=30, img_size=(224, 224), class_labels=None, is_training=False):
    while True:
        video_data = []
        labels = []
        
        # Loop through each class folder
        for class_name, class_index in class_labels.items():
            class_path = os.path.join(folder_path, class_name)
            if not os.path.exists(class_path):
                print(f"Warning: {class_path} does not exist")
                continue
                
            video_files = [f for f in os.listdir(class_path) if f.endswith('.mp4') or f.endswith('.avi')]
            
            # Shuffle files for better training
            if is_training:
                np.random.shuffle(video_files)
            
            for video_file in video_files:
                video_path = os.path.join(class_path, video_file)
                frames = video_to_frames(video_path, max_frames, img_size)
                
                # Apply data augmentation if training
                if is_training:
                    # Random horizontal flip with 50% probability
                    if np.random.random() > 0.5:
                        frames = frames[:, :, ::-1, :]
                    
                    # Random brightness/contrast variation
                    if np.random.random() > 0.5:
                        brightness_factor = np.random.uniform(0.8, 1.2)
                        frames = np.clip(frames * brightness_factor, -1.0, 1.0)
                
                video_data.append(frames)
                labels.append(class_index)
                
                # If we have reached batch_size, yield the data
                if len(video_data) == batch_size:
                    yield np.array(video_data), tf.keras.utils.to_categorical(labels, num_classes=len(class_labels))
                    video_data = []
                    labels = []
        
        # If there are remaining videos in the batch, yield them
        if len(video_data) > 0:
            yield np.array(video_data), tf.keras.utils.to_categorical(labels, num_classes=len(class_labels))

# Path to the dataset folders (train, val, test)
train_folder_path = r'C:\\Users\\jagru\\OneDrive\\Desktop\\dataset_v2\\dataset_v2\\train'
val_folder_path = r'C:\\Users\\jagru\\OneDrive\\Desktop\\dataset_v2\\dataset_v2\\val'

# Get all class folders
class_folders = [folder for folder in os.listdir(train_folder_path) if os.path.isdir(os.path.join(train_folder_path, folder))]
class_labels = {folder: idx for idx, folder in enumerate(class_folders)}
num_classes = len(class_labels)
print(f"Found {num_classes} classes: {list(class_labels.keys())}")

# Build an improved model architecture
def build_model(input_shape, num_classes):
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Initial 3D Conv layer
    x = layers.Conv3D(32, kernel_size=(3, 3, 3), padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    
    # Second 3D Conv block
    x = layers.Conv3D(64, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    
    # Third 3D Conv block
    x = layers.Conv3D(128, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(1, 2, 2))(x)
    
    # Fourth 3D Conv block
    x = layers.Conv3D(256, kernel_size=(3, 3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    
    # Using a fixed reshape for TimeDistributed layers
    # This approach is more compatible with TensorFlow's static graph
    # After 3D convolutions and pooling, we'll have reduced the spatial dimensions
    # Calculate the expected shape after 3D convs and pooling:
    # Original: (30, 224, 224, 3)
    # After pooling: (15, 14, 14, 256)
    x = layers.Reshape((15, 14*14*256))(x)
    
    # Dense layer to reduce dimensions before LSTM
    x = layers.TimeDistributed(layers.Dense(512))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Add bidirectional LSTM layers for temporal features
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Bidirectional(layers.LSTM(128))(x)
    x = layers.Dropout(0.5)(x)
    
    # Dense layers with dropout for classification
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs, outputs)
    return model

# Alternative simplified model for better compatibility
def build_simplified_model(input_shape, num_classes):
    model = models.Sequential([
        # 3D CNN layers
        layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling3D(pool_size=(1, 2, 2)),
        
        layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling3D(pool_size=(1, 2, 2)),
        
        layers.Conv3D(128, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling3D(pool_size=(1, 2, 2)),
        
        layers.Conv3D(256, kernel_size=(3, 3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling3D(pool_size=(2, 2, 2)),
        
        # Flatten the 3D output to feed into LSTM
        layers.TimeDistributed(layers.Flatten()),
        
        # LSTM layers
        layers.Bidirectional(layers.LSTM(256, return_sequences=True)),
        layers.Dropout(0.5),
        layers.Bidirectional(layers.LSTM(128)),
        layers.Dropout(0.5),
        
        # Dense classification layers
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Try to build the model using both approaches
try:
    print("Building advanced functional model...")
    model = build_model(input_shape, num_classes)
except Exception as e:
    print(f"Error with functional model: {e}")
    print("Falling back to simplified sequential model...")
    model = build_simplified_model(input_shape, num_classes)

# Try to use mixed precision if available
try:
    # Use mixed precision for faster training on compatible GPUs
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision training enabled")
except Exception as e:
    print(f"Mixed precision not available: {e}")
    print("Using default precision")

# Compile with a more effective optimizer and learning rate schedule
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_accuracy')]
)

# Display model summary
model.summary()

# Set batch size and other parameters
batch_size = 8  # Reduced to handle memory constraints
epochs = 5      # Reduced epochs with better architecture

# Calculate steps based on dataset size
# Estimate 200 videos per class for training, 50 for validation (adjust these numbers based on your dataset)
estimated_train_samples = len(class_labels) * 50  # Adjusted to a more conservative estimate
estimated_val_samples = len(class_labels) * 20
steps_per_epoch = max(1, estimated_train_samples // batch_size)
validation_steps = max(1, estimated_val_samples // batch_size)

print(f"Training with {steps_per_epoch} steps per epoch, {validation_steps} validation steps")

# Create data generators for training and validation
train_gen = data_generator(train_folder_path, batch_size=batch_size, 
                          class_labels=class_labels, is_training=True)
val_gen = data_generator(val_folder_path, batch_size=batch_size, 
                        class_labels=class_labels, is_training=False)

# Callbacks for better training
callbacks_list = [
    # Save the best model based on validation accuracy
    callbacks.ModelCheckpoint(
        filepath='best_sign_language_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    # Early stopping to prevent overfitting
    callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        verbose=1,
        restore_best_weights=True
    ),
    # Reduce learning rate when validation loss plateaus
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
    # TensorBoard logging
    callbacks.TensorBoard(
        log_dir='./logs',
        histogram_freq=1
    )
]

# Train the model with the data generator
try:
    history = model.fit(
        train_gen, 
        epochs=epochs, 
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen, 
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        verbose=1
    )

    # Save the final model
    model.save('final_sign_language_model.h5')

    # Plot training history
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 5))

        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
    except Exception as e:
        print(f"Could not generate plots: {e}")
        
except Exception as e:
    print(f"Error during training: {e}")
    print("Try reducing batch size or simplifying the model further if you encounter memory issues.")
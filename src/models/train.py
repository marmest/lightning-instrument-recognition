import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, GlobalMaxPooling2D, MaxPooling2D, Flatten, TimeDistributed, Dense, ZeroPadding2D, Dropout, BatchNormalization, LSTM, GRU, Bidirectional, ReLU, LeakyReLU, Reshape
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam

def create_model_mel(input_shape=(128, 100, 1)):
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Conv2D(8, kernel_size=(3,3), padding='same', input_shape=input_shape))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.33))
    model.add(tf.keras.layers.Conv2D(16, kernel_size=(3,3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.33))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(3,3)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    model.add(tf.keras.layers.Conv2D(24, kernel_size=(3,3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.33))
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.33))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(3,3)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.33))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.33))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(3,3)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    model.add(tf.keras.layers.Conv2D(256, kernel_size=(3,3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.33))
    model.add(tf.keras.layers.Conv2D(512, kernel_size=(3,3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.33))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(3,3)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    model.add(tf.keras.layers.GlobalMaxPooling2D())
    
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(11, activation='softmax'))
    
    model.summary()
    
    return model

def create_model_modgd(input_shape=(128, 98, 1)):
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Conv2D(8, kernel_size=(3,3), padding='same', input_shape=input_shape))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.33))
    model.add(tf.keras.layers.Conv2D(16, kernel_size=(3,3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.33))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(3,3)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    model.add(tf.keras.layers.Conv2D(24, kernel_size=(3,3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.33))
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.33))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(3,3)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.33))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.33))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(3,3)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    model.add(tf.keras.layers.Conv2D(256, kernel_size=(3,3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.33))
    model.add(tf.keras.layers.Conv2D(512, kernel_size=(3,3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.33))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(3,3)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    model.add(tf.keras.layers.GlobalMaxPooling2D())
    
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(11, activation='softmax'))
    
    model.summary()
    
    return model

def create_model_pitch(input_shape=(36, 100, 1)):
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Conv2D(24, kernel_size=(3,3), padding='same', input_shape=input_shape))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.33))
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.33))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(3,3)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.33))
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3,3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.33))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(3,3)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    model.add(tf.keras.layers.Conv2D(256, kernel_size=(3,3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.33))
    model.add(tf.keras.layers.Conv2D(512, kernel_size=(3,3), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.33))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=(3,3)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    model.add(tf.keras.layers.GlobalMaxPooling2D())
    
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(11, activation='softmax'))
    
    model.summary()
    
    return model

def train_model(model, X_train, X_val, y_train, y_val, epochs, path):
    
    checkpoint = ModelCheckpoint(path + '_{epoch:02d}.h5', verbose=1, mode='auto')
    
    callbacks = [checkpoint]
    
    
    model.fit(x=X_train, y=y_train, batch_size=128, epochs=epochs,
              validation_data=(X_val, y_val), 
              callbacks=callbacks
             )

    return model



# Load X_train, X_val and y_train, y_val
X_train_mel = np.load("data/X_train_mel.npy")
X_train_mel = np.reshape(X_train_mel, (X_train_mel.shape[0], X_train_mel.shape[1], X_train_mel.shape[2], 1))

X_train_modgd = np.load("data/X_train_modgd.npy")
X_train_modgd = np.reshape(X_train_modgd, (X_train_modgd.shape[0], X_train_modgd.shape[1], X_train_modgd.shape[2], 1))

X_train_pitch = np.load("data/X_train_pitch.npy")
X_train_pitch = np.reshape(X_train_pitch, (X_train_pitch.shape[0], X_train_pitch.shape[1], X_train_pitch.shape[2], 1))

y_train = np.load("data/y_train.npy")
print("X_train, y_train loaded!")

X_val_mel = np.load("data/X_val_mel.npy")
X_val_mel = np.reshape(X_val_mel, (X_val_mel.shape[0], X_val_mel.shape[1], X_val_mel.shape[2], 1))

X_val_modgd = np.load("data/X_val_modgd.npy")
X_val_modgd = np.reshape(X_val_modgd, (X_val_modgd.shape[0], X_val_modgd.shape[1], X_val_modgd.shape[2], 1))

X_val_pitch = np.load("data/X_val_pitch.npy")
X_val_pitch = np.reshape(X_val_pitch, (X_val_pitch.shape[0], X_val_pitch.shape[1], X_val_pitch.shape[2], 1))

y_val = np.load("data/y_val.npy")
print("X_val, y_val loaded!")

print("Mel model training!")
model_mel = create_model_mel()
model_mel.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model_mel = train_model(model_mel, X_train_mel, X_val_mel, y_train, y_val, epochs = 30, path = "checkpoints/cnn_mel_85")

print("Modgd model training!")
model_modgd = create_model_modgd()
model_modgd.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model_modgd = train_model(model_modgd, X_train_modgd, X_val_modgd, y_train, y_val, epochs = 30, path = "checkpoints/cnn_modgd_85")

print("Pitch model training!")
model_pitch = create_model_mel()
model_pitch.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model_pitch = train_model(model_pitch, X_train_pitch, X_val_pitch, y_train, y_val, epochs = 30, path = "checkpoints/cnn_pitch_85") 
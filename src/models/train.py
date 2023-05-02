import numpy as np
import pickle
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


#hyper parameters
epochs = 30
batch_size = 128
lr = 0.001

#model architecture
class CNN_mel(nn.Module):
    def __init__(self):
        super(CNN_mel, self).__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.leakyrelu1 = nn.LeakyReLU(negative_slope=0.33)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.leakyrelu2 = nn.LeakyReLU(negative_slope=0.33)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.dropout1 = nn.Dropout2d(p=0.25)

        self.conv3 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.leakyrelu3 = nn.LeakyReLU(negative_slope=0.33)
        self.conv4 = nn.Conv2d(24, 32, kernel_size=3, padding=1)
        self.leakyrelu4 = nn.LeakyReLU(negative_slope=0.33)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.dropout2 = nn.Dropout2d(p=0.25)

        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.leakyrelu5 = nn.LeakyReLU(negative_slope=0.33)
        self.conv6 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.leakyrelu6 = nn.LeakyReLU(negative_slope=0.33)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.dropout3 = nn.Dropout2d(p=0.25)

        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.leakyrelu7 = nn.LeakyReLU(negative_slope=0.33)
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.leakyrelu8 = nn.LeakyReLU(negative_slope=0.33)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.dropout4 = nn.Dropout2d(p=0.25)

        self.globalpool = nn.AdaptiveMaxPool2d((1,1))

        self.fc1 = nn.Linear(512, 1024)
        self.dropout5 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 11)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyrelu1(x)
        x = self.conv2(x)
        x = self.leakyrelu2(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.leakyrelu3(x)
        x = self.conv4(x)
        x = self.leakyrelu4(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv5(x)
        x = self.leakyrelu5(x)
        x = self.conv6(x)
        x = self.leakyrelu6(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        x = self.conv7(x)
        x = self.leakyrelu7(x)
        x = self.conv8(x)
        x = self.leakyrelu8(x)
        x = self.pool4(x)
        x = self.dropout4(x)
        x = self.globalpool(x)
        x = x.view(-1, 512)
        x = self.fc1(x)
        x = self.dropout5(x)
        x = self.fc2(x)

        return x

#training loop
def train(model, optimizer, criterion, train_loader, epochs, path='../../models/checkpoints/'):
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), path + '_epoch{:02d}.pt'.format(epoch+1))

        print(f'Epoch: {epoch}, Loss: {loss.item():.5f}')

model = CNN_mel()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

X_train_mel = torch.load('../../data/processed/X_train_mel.pt')
X_train_mel = X_train_mel.view(X_train_mel.size(0), 1, X_train_mel.size(1), X_train_mel.size(2))
y_train = torch.load('../../data/processed/y_train.pt')
dataset = TensorDataset(X_train_mel, y_train)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print('X_train, y_train loaded!')

train(model, optimizer, criterion, train_loader, epochs)
sys.exit()
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
X_train_mel = np.load("../../data/processed/X_train_mel.npy")
X_train_mel = np.reshape(X_train_mel, (X_train_mel.shape[0], X_train_mel.shape[1], X_train_mel.shape[2], 1))

X_train_modgd = np.load("../../data/processed/X_train_modgd.npy")
X_train_modgd = np.reshape(X_train_modgd, (X_train_modgd.shape[0], X_train_modgd.shape[1], X_train_modgd.shape[2], 1))

X_train_pitch = np.load("../../data/processed/X_train_pitch.npy")
X_train_pitch = np.reshape(X_train_pitch, (X_train_pitch.shape[0], X_train_pitch.shape[1], X_train_pitch.shape[2], 1))

y_train = np.load("data/y_train.npy")
print("X_train, y_train loaded!")

X_val_mel = np.load("../../data/processed/X_val_mel.npy")
X_val_mel = np.reshape(X_val_mel, (X_val_mel.shape[0], X_val_mel.shape[1], X_val_mel.shape[2], 1))

X_val_modgd = np.load("../../data/processed/X_val_modgd.npy")
X_val_modgd = np.reshape(X_val_modgd, (X_val_modgd.shape[0], X_val_modgd.shape[1], X_val_modgd.shape[2], 1))

X_val_pitch = np.load("../../data/processed/X_val_pitch.npy")
X_val_pitch = np.reshape(X_val_pitch, (X_val_pitch.shape[0], X_val_pitch.shape[1], X_val_pitch.shape[2], 1))

y_val = np.load("../../data/processed/y_val.npy")
print("X_val, y_val loaded!")

print("Mel model training!")
model_mel = create_model_mel()
model_mel.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model_mel = train_model(model_mel, X_train_mel, X_val_mel, y_train, y_val, epochs = 30, path = "../../models/checkpoints/cnn_mel_85")

print("Modgd model training!")
model_modgd = create_model_modgd()
model_modgd.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model_modgd = train_model(model_modgd, X_train_modgd, X_val_modgd, y_train, y_val, epochs = 30, path = "../../models/checkpoints/cnn_modgd_85")

print("Pitch model training!")
model_pitch = create_model_mel()
model_pitch.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model_pitch = train_model(model_pitch, X_train_pitch, X_val_pitch, y_train, y_val, epochs = 30, path = "../../models/checkpoints/cnn_pitch_85") 
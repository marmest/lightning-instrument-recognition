import hydra
from hydra import compose, initialize

import numpy as np
import sys
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# global initialization
initialize(version_base=None, config_path="../../configs")
cfg = compose(config_name="config")


#hyper parameters
epochs = cfg.training.epochs
batch_size = cfg.training.batch_size
lr = cfg.training.learning_rate

#device config
print(torch.cuda.is_available())
print(torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
class CNN_modgd(nn.Module):
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
    
class CNN_pitch(nn.Module):
    def __init__(self):
        super(CNN_pitch, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 24, kernel_size=(3, 3), padding='same')
        self.leakyrelu1 = nn.LeakyReLU(negative_slope=0.33)
        self.conv2 = nn.Conv2d(24, 32, kernel_size=(3, 3), padding='same')
        self.leakyrelu2 = nn.LeakyReLU(negative_slope=0.33)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.dropout1 = nn.Dropout(p=0.25)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding='same')
        self.leakyrelu3 = nn.LeakyReLU(negative_slope=0.33)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding='same')
        self.leakyrelu4 = nn.LeakyReLU(negative_slope=0.33)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.dropout2 = nn.Dropout(p=0.25)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding='same')
        self.leakyrelu5 = nn.LeakyReLU(negative_slope=0.33)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding='same')
        self.leakyrelu6 = nn.LeakyReLU(negative_slope=0.33)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.dropout3 = nn.Dropout(p=0.25)
        
        self.gpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        
        self.fc1 = nn.Linear(512, 1024)
        self.dropout4 = nn.Dropout(p=0.5)
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
        
        x = self.gpool(x)
        
        x = x.view(-1, 512)
        
        x = self.fc1(x)
        x = self.dropout4(x)
        x = self.fc2(x)

        return x

#training loop
def train(model, optimizer, criterion, train_loader, val_loader, epochs, path='../../models/cnn_mel.pt'):
    min_val_loss = 1000.0
    for epoch in range(epochs):
        #training
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        #validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                batch_val_loss = criterion(outputs, labels)
                val_loss += batch_val_loss.item() * inputs.size(0)
            val_loss /= len(val_loader.dataset)
            print(val_loss)
                

        # Save best model
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            model_scripted = torch.jit.script(model) # Export to TorchScript
            model_scripted.save(path) # Save

        print(f'Epoch: {epoch}, Train Loss: {loss:.5f}, Val Loss: {val_loss:.5f}, Min Val Loss: {min_val_loss:.5f}')

#mel training
model = CNN_mel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

X_train_mel = torch.load('../../data/processed/X_train_mel.pt')
X_val_mel = torch.load('../../data/processed/X_val_mel.pt')

y_train = torch.load('../../data/processed/y_train.pt')
y_val = torch.load('../../data/processed/y_val.pt')

train_dataset = TensorDataset(X_train_mel, y_train)
val_dataset = TensorDataset(X_val_mel, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print('X_train_mel, y_train_mel loaded!')

train(model, optimizer, criterion, train_loader, val_loader, epochs)

#modgd training
model = CNN_mel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

X_train_modgd = torch.load('../../data/processed/X_train_modgd.pt')
X_val_modgd = torch.load('../../data/processed/X_val_modgd.pt')

y_train = torch.load('../../data/processed/y_train.pt')
y_val = torch.load('../../data/processed/y_val.pt')

train_dataset = TensorDataset(X_train_modgd, y_train)
val_dataset = TensorDataset(X_val_modgd, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print('X_train_modgd, y_train_modgd loaded!')

train(model, optimizer, criterion, train_loader, val_loader, epochs, path='../../models/cnn_modgd.pt')

#pitchgram training
model = CNN_pitch().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

X_train_pitch = torch.load('../../data/processed/X_train_pitch.pt')
X_val_pitch = torch.load('../../data/processed/X_val_pitch.pt')

y_train = torch.load('../../data/processed/y_train.pt')
y_val = torch.load('../../data/processed/y_val.pt')

train_dataset = TensorDataset(X_train_pitch, y_train)
val_dataset = TensorDataset(X_val_pitch, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print('X_train_pitch, y_train_pitch loaded!')

train(model, optimizer, criterion, train_loader, val_loader, epochs, path='../../models/cnn_pitch.pt')
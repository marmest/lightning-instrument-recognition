import hydra
from hydra import compose, initialize

import numpy as np
import torch
import torch.nn as nn
import pickle
import sys
import os

# global initialization
initialize(version_base=None, config_path="../../configs")
cfg = compose(config_name="config")

#device config
print(torch.cuda.is_available())
print(torch.__version__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_acc(y1, y2):
    ret = 0
    for i, _ in enumerate(y1):
        ret += int(y1[i] == y2[i])
    ret /= len(y1)
    return ret

# constants
num_classes = cfg.constants.num_classes

# paths
models_dir = cfg.paths.models_folder

# label map
label_map = cfg.label_map

# methods
aggregation_method = cfg.aggregation.method
fusion_method = cfg.fusion.method

# fusion
theta_step = cfg.fusion.theta_step

models = []
model_names = []

for root, dirs, files in os.walk(models_dir):
    files.sort()
    for file in files:
        if file.endswith('.pt'):
            model = torch.jit.load(os.path.join(models_dir, file))
            model = model.to(device)
            model_names.append(file)
            models.append(model)
n = len(models)
print(model_names)

print("Load X_test and y_test")
f = open("../../data/processed/X_test_mel.pkl", "rb")
X_test_mel = pickle.load(f)
f.close()
print("Mel data loaded!")

f = open("../../data/processed/X_test_modgd.pkl", "rb")
X_test_modgd = pickle.load(f)
f.close()
print("Modgd data loaded!")

f = open("../../data/processed/X_test_pitch.pkl", "rb")
X_test_pitch = pickle.load(f)
f.close()
print("Pitch data loaded!")

f = open("../../data/processed/y_test.pkl", "rb")
y_test = pickle.load(f)
f.close()

X_test = [X_test_mel, X_test_modgd, X_test_pitch]

a = y_test
y_test = torch.zeros((len(a), num_classes))
for i in range(len(a)):
    y_test[i] = a[i][0]

print("Loaded!")

print("Computing probabilities")
probabilities = []

for i in range(n):
    tmp_probs = []
    model = models[i]
    model.eval()
    for (key, val) in X_test[i].items():
        val = val.to(device)
        
        #print(i, ":", key, "/", len(X_test[i]))
        
        # Model prediction
        #val = torch.view(val, (val.shape[0], val.shape[1], val.shape[2], 1))
        val = val.view(val.size(0), 1, val.size(1), val.size(2))
        print(val.shape)
        prediction = model(val)

        # Aggregation
        if aggregation_method == "s1":
            prediction = torch.mean(prediction, axis = 0)
        elif aggregation_method == "s2":
            prediction = torch.sum(prediction, axis = 0)
            m = torch.max(prediction)
            prediction /= m
        tmp_probs.append(prediction.detach().cpu().numpy())
        
    tmp_probs = np.array(tmp_probs)
    probabilities.append(tmp_probs)

probabilities = np.array(probabilities)

print("Probabilities computed!")

y_pred = np.zeros((probabilities.shape[1], probabilities.shape[2]))
mul_probabilities = np.zeros((probabilities.shape[1], probabilities.shape[2]))
fusion = np.zeros((num_classes, 3))
thetas = np.arange(0, 1.01, theta_step)

if fusion_method == 1:
    
    for k in range(mul_probabilities.shape[1]):
        best_acc = 0.0
        for theta1 in thetas:
            for theta2 in thetas:
                for theta3 in thetas:
                    tmp_pred1 = np.array(probabilities[0, :, k]) - theta1
                    tmp_pred2 = np.array(probabilities[1, :, k]) - theta2
                    tmp_pred3 = np.array(probabilities[2, :, k]) - theta3

                    tmp_pred1[tmp_pred1 > 0] = 1
                    tmp_pred1[tmp_pred1 <= 0] = 0
                    tmp_pred2[tmp_pred2 > 0] = 1
                    tmp_pred2[tmp_pred2 <= 0] = 0
                    tmp_pred3[tmp_pred3 > 0] = 1
                    tmp_pred3[tmp_pred3 <= 0] = 0

                    tmp_pred = np.multiply(np.multiply(tmp_pred1, tmp_pred2), tmp_pred3)

                    acc = calculate_acc(tmp_pred, y_test[:, k])

                    if acc > best_acc:
                        best_acc = acc
                        y_pred[:, k] = tmp_pred
                        fusion[k, :] = [theta1, theta2, theta3]
                    
        print(fusion[k][0], ", ", fusion[k][1], ", ", fusion[k][2], sep = '')

np.save("fusion_thresholds.npy", fusion)
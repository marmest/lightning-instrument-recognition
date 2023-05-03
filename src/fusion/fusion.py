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

def calculate_f1(y1, y2):
    tp = 0
    fp = 0
    fn = 0
    for i, _ in enumerate(y1):
        if y1[i] == 1 and y2[i] == 1:
            tp += 1
        elif y1[i] == 1 and y2[i] == 0:
            fp += 1
        elif y1[i] == 0 and y2[i] == 1:
            fn += 1
    f1 = (2 * tp) / (2 * tp + fp + fn)
    return f1

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
version = cfg.fusion.version
optimization_metric = cfg.fusion.optimization_metric

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
num_models = len(models)
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

for i in range(num_models):
    tmp_probs = []
    model = models[i]
    model.eval()
    for (key, val) in X_test[i].items():

        val = val.to(device)
        
        # Model prediction
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
fusion = np.zeros((num_classes, num_models))
thetas = np.arange(0, 1.01, theta_step)

if fusion_method == 1:
    
    thetas_mel = thetas
    if "mel" not in version:
        thetas_mel = [-1]
    thetas_modgd = thetas
    if "modgd" not in version:
        thetas_modgd = [-1]
    thetas_pitch = thetas
    if "pitch" not in version:
        thetas_pitch = [-1]

    for k in range(num_classes):
        best_acc = 0.0
        best_f1 = 0.0
        for theta1 in thetas_mel:
            for theta2 in thetas_modgd:
                for theta3 in thetas_pitch:
                    tmp_pred1 = np.array(probabilities[0, :, k]) - theta1
                    tmp_pred2 = np.array(probabilities[1, :, k]) - theta2
                    tmp_pred3 = np.array(probabilities[2, :, k]) - theta3

                    tmp_pred1[tmp_pred1 > 0] = 1
                    tmp_pred1[tmp_pred1 <= 0] = 0
                    tmp_pred2[tmp_pred2 > 0] = 1
                    tmp_pred2[tmp_pred2 <= 0] = 0
                    tmp_pred3[tmp_pred3 > 0] = 1
                    tmp_pred3[tmp_pred3 <= 0] = 0

                    fusion_thetas = []
                    if theta1 == -1:
                        tmp_pred1 = np.ones(tmp_pred1.shape)
                    else:
                        fusion_thetas.append(theta1)
                    if theta2 == -1:
                        tmp_pred2 = np.ones(tmp_pred2.shape)
                    else:
                        fusion_thetas.append(theta2)
                    if theta1 == -1:
                        tmp_pred3 = np.ones(tmp_pred3.shape)
                    else:
                        fusion_thetas.append(theta3)

                    tmp_pred = np.multiply(np.multiply(tmp_pred1, tmp_pred2), tmp_pred3)

                    if optimization_metric == "acc":
                        acc = calculate_acc(tmp_pred, y_test[:, k])
                        if acc > best_acc:
                            best_acc = acc
                            y_pred[:, k] = tmp_pred
                            fusion[k, :] = fusion_thetas
                    
                    elif optimization_metric == "f1":
                        f1 = calculate_f1(tmp_pred, y_test[:, k])
                        if f1 > best_f1:
                            best_f1 = f1
                            y_pred[:, k] = tmp_pred
                            fusion[k, :] = fusion_thetas
                    
        print(fusion[k][0], ", ", fusion[k][1], ", ", fusion[k][2], sep = '')

# Save fusion thresholds
save_str = "fusion_thresholds_"
for model_name in version:
    save_str += model_name
    save_str += "_"
save_str += optimization_metric
save_str += ".npy"
#np.save("fusion_thresholds.npy", fusion)
np.save(save_str, fusion)
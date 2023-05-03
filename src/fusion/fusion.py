import numpy as np
import pickle
import sys
import os
from keras.models import load_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

def calculate_acc(y1, y2):
    ret = 0
    for i, _ in enumerate(y1):
        ret += int(y1[i] == y2[i])
    ret /= len(y1)
    return ret

models_dir = "../../models/"

label_map = {"cel": 0, "cla": 1, "flu": 2, "gac": 3, "gel": 4, "org": 5,
             "pia": 6, "sax": 7, "tru": 8, "vio": 9, "voi": 10}

models = []
model_names = []
for root, dirs, files in os.walk(models_dir):
    files.sort()
    for file in files:
        if file.endswith('.h5'):
            model = load_model(models_dir + "/" + file)
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
y_test = np.zeros((len(a), 11))
for i in range(len(a)):
    y_test[i] = a[i][0]

print("Loaded!")

print("Computing probabilities")
probabilities = []

for i in range(n):
    tmp_probs = []
    model = models[i]
    print(i)
    for (key, val) in X_test[i].items():
        
        #print(i, ":", key, "/", len(X_test[i]))
        
        # Model prediction
        val = np.reshape(val, (val.shape[0], val.shape[1], val.shape[2], 1))
        prediction = model.predict(val, verbose = 0)

        # Aggregation
        prediction = np.mean(prediction, axis = 0)
        #prediction = np.sum(prediction, axis = 0)
        #m = np.max(prediction)
        #prediction /= m
        tmp_probs.append(prediction)
        
    tmp_probs = np.array(tmp_probs)
    probabilities.append(tmp_probs)

probabilities = np.array(probabilities)

print("Probabilities computed!")

y_pred = np.zeros((probabilities.shape[1], probabilities.shape[2]))
mul_probabilities = np.zeros((probabilities.shape[1], probabilities.shape[2]))
fusion = np.zeros((11, 3))
thetas = np.arange(0, 1.01, 0.05)

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

#np.save("fusion_thresholds.npy", fusion)


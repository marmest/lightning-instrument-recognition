
import os
import requests
import pydub
from pydub import AudioSegment
from pydub.playback import play
import json

BASE = "http://127.0.0.1:5050/predict_test"


test_folder = "C:/Users/Fran/Documents/Lumen/lumen/dataset/raw/test_dataset" 


files = [file[:-4] for file in os.listdir(test_folder) if file.endswith('.wav')]

# Sort the files based on the numerical value of N
sorted_files = sorted(files, key=lambda x: int(x.split('_')[1]))
i = 0
file_dict = {}
for file in os.listdir(test_folder):
    file_path = os.path.join(test_folder, file)
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            audio_file = f.read()
        filename = os.path.basename(file_path)
        response = requests.post(BASE, files = {"audiofile" : audio_file})        
        file_dict[filename] = response.json()
        i = i + 1
        print(i)

with open("ogledno_rjesenje2.json", "w") as f:
    json.dump(file_dict, f, indent=4, separators=(", ", ": "))
    f.write("\n")


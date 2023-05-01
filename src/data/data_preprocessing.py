import numpy as np
import pickle
import librosa
import os
import scipy.signal 
from sklearn.model_selection import train_test_split
from collections import OrderedDict

step_perc = 1.0 #koliko nam je step kad segmentiramo spektrogram - default 100%

def create_melspectrogram(x, sr, n_fft, hop_length):
    
    X = librosa.core.stft(x, hop_length=hop_length, n_fft=n_fft)
    spectrogram = np.abs(X)
    mel_spectrogram = librosa.amplitude_to_db(spectrogram)
    
    return mel_spectrogram

def create_modgdgram(signal, sr, n_frequency_bins=128, frame_length=0.05, hop_length=0.01, alpha=0.9, gamma=0.5, nc=10):
    
    #convert to samples
    frame_length = int(frame_length * sr)
    hop_length = int(hop_length * sr)
    
    n_time_points = int(len(signal) / sr) * 98
   
    frames = librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length, axis=0)
    n_frames = len(frames)
   
    #compute modgdgram
    modgdgram = np.zeros((n_frames, frame_length))
    for index, x in enumerate(frames):
        #compute modgdgram for a frame
        n = np.arange(1, len(x) + 1)
        y = n * x
        X = scipy.fft.fft(x)
        XR = np.real(X)
        XI = np.imag(X)

        Y = scipy.fft.fft(y)
        YR = np.real(Y)
        YI = np.imag(Y)

        X_mag = np.abs(X) + 1e-10
        cepstrum = np.fft.ifft(np.log(X_mag)).real #complex cepstrum
        l_w = np.zeros_like(cepstrum)
        l_w[:nc] = 1.0 #low-order cepstral window
        cepstrum_smoothed = cepstrum * l_w
        S = np.abs(np.exp(np.fft.fft(cepstrum_smoothed))) #cepstrally smoothed version of X
        sign = np.sign(np.multiply(XR, YR) + np.multiply(XI, YI))
        module = np.power(np.abs(np.divide(np.multiply(XR, YR) + np.multiply(XI, YI), np.power(S, 2*gamma))), alpha)
        tau = np.multiply(sign, module)

        modgdgram[index] = tau

    modgdgram = np.transpose(modgdgram)
    modgdgram = librosa.amplitude_to_db(modgdgram)
    modgdgram = modgdgram[:n_frequency_bins, :n_time_points]
   
    return modgdgram

def create_pitchgram(x, sr, hop_length):
    
    pitchgram = librosa.feature.chroma_cqt(y=x, sr=sr, hop_length=hop_length, n_chroma=36)
    
    return pitchgram

def extract_from_file(filename, seconds = 1, instrument = "pol", api = False):

    # Read the data
    y, sr = librosa.load(filename, sr = 22050, mono = True)

    n_fft = 255
    hop_length = 220
    
    melspectrogram = create_melspectrogram(y, sr, n_fft=n_fft, hop_length=hop_length)
    modgdgram = create_modgdgram(y, sr)
    pitchgram = create_pitchgram(y, sr, hop_length=hop_length)

    # Segementation of the spectogram
    seg_dur_mel = 100 * seconds
    seg_dur_modgd = 98 * seconds
    seg_dur_pitch = 100 * seconds
    mels = []
    modgds = []
    pitchs = []
    for idx in range(0, melspectrogram.shape[1] - seg_dur_mel + 1, int(step_perc * seg_dur_mel)):
        mels.append(melspectrogram[:, idx : (idx + seg_dur_mel)])
    for idx in range(0, modgdgram.shape[1] - seg_dur_modgd + 1, int(step_perc * seg_dur_modgd)):
        modgds.append(modgdgram[:, idx : (idx + seg_dur_modgd)])
    for idx in range(0, pitchgram.shape[1] - seg_dur_pitch + 1, int(step_perc * seg_dur_pitch)):
        pitchs.append(pitchgram[:, idx : (idx + seg_dur_pitch)])   
    mels = np.array(mels)
    modgds = np.array(modgds)
    pitchs = np.array(pitchs)

    features = {}
    
    features["mels"] = mels
    features["modgds"] = modgds
    features["pitchs"] = pitchs
    if(api == False):
        features["labels"] = np.zeros([11])
        
        if(instrument == "pol"):
            with open(filename[:-4] + '.txt', 'r') as fp:
                lines = fp.readlines()
                for l in lines:
                    features["labels"][label_map[l[:3]]] = 1
                    
        else:
            features["labels"][label_map[instrument]] = 1
                
    
    return features

def main_training():
    
    features = []
    for instrument in label_map.keys():
        print("Entering folder ", train_folder + instrument)

        for root, dirs, files in os.walk(train_folder + instrument):
            total_files = len(files)

            count = 0
            for file in files:
                if file.endswith('.wav'):
                    count += 1
                    print(count, "/", total_files)
                    feat = extract_from_file(train_folder + instrument +  "/" + file, instrument=instrument)
                    features.append(feat)
                    
    return features

def main_testing():
    
    print("Entering folder ", test_folder)

    features = []
    for root, dirs, files in os.walk(test_folder):
        total_files = int(len(files)/2)

        count = 0
        for file in files:
            if file.endswith('.wav'):
                count += 1
                print(count, "/", total_files)
                feat = extract_from_file(test_folder + file)
                features.append(feat)
    return features

# function definitions
# ----------------------------------------------
# main code

if __name__ == "__main__":

    train_folder = '../dataset/raw/IRMAS_Training_Data/'
    test_folder = "../dataset/raw/IRMAS_Validation_Data/"

    label_map = {"cel": 0, "cla": 1, "flu": 2, "gac": 3, "gel": 4, "org": 5,
                "pia": 6, "sax": 7, "tru": 8, "vio": 9, "voi": 10}

    print("Training data preprocessing started!")
    train_data = main_training()
    print("Training data preprocessing finished!")

    print("Spliting train data!")

    X_mel = np.concatenate([feat["mels"] for feat in train_data])
    X_mel = np.expand_dims(X_mel, axis = 3)
    X_modgd = np.concatenate([feat["modgds"] for feat in train_data])
    X_modgd = np.expand_dims(X_modgd, axis = 3)
    X_pitch = np.concatenate([feat["pitchs"] for feat in train_data])
    X_pitch = np.expand_dims(X_pitch, axis = 3)

    y = np.concatenate([[feat["labels"] for i in range(len(feat["mels"]))] for feat in train_data])

    # Randomly split training data -> 85% training set, 15% validation set
    I = np.array([i for i in range(len(y))])
    I_train, I_val, _, _ = train_test_split(I, I, test_size = 0.15, random_state = 42)

    X_train_mel = np.zeros((int(0.85 * len(I)), 128, 100, 1))
    X_val_mel = np.zeros((len(I) - len(X_train_mel), 128, 100, 1))
    X_train_modgd = np.zeros((int(0.85 * len(I)), 128, 98, 1))
    X_val_modgd = np.zeros((len(I) - len(X_train_modgd), 128, 98, 1))
    X_train_pitch = np.zeros((int(0.85 * len(I)), 36, 100, 1))
    X_val_pitch = np.zeros((len(I) - len(X_train_pitch), 36, 100, 1))
    y_train = np.zeros((int(0.85 * len(I)), 11))
    y_val = np.zeros((len(I) - len(y_train), 11))

    for i in range(len(I_train)):
        X_train_mel[i] = X_mel[I_train[i]]
        X_train_modgd[i] = X_modgd[I_train[i]]
        X_train_pitch[i] = X_pitch[I_train[i]]
        y_train[i] = y[I_train[i]]
        
    for i in range(len(I_val)):
        X_val_mel[i] = X_mel[I_val[i]]
        X_val_modgd[i] = X_modgd[I_val[i]]
        X_val_pitch[i] = X_pitch[I_val[i]]
        y_val[i] = y[I_val[i]]

    print("Finished splitting!")

    # Save Numpy arrays and dictionaries
    print("Saving training data!")

    np.save("../../data/processed/X_train_mel.npy", X_train_mel)
    np.save("../../data/processed/X_val_mel.npy", X_val_mel)
    np.save("../../data/processed/X_train_modgd.npy", X_train_modgd)
    np.save("../../data/processed/X_val_modgd.npy", X_val_modgd)
    np.save("../../data/processed/X_train_pitch.npy", X_train_pitch)
    np.save("../../data/processed/X_val_pitch.npy", X_val_pitch)
    np.save("../../data/processed/y_train.npy", y_train)
    np.save("../../data/processed/y_val.npy", y_val)

    print("Saved training data!")

    print("Testing started!")
    test_data = main_testing()
    print("Testing finished!")

    print("Test data!")

    # Initialize the testing feature and label dictionaries
    X_test_mel = OrderedDict()
    X_test_modgd = OrderedDict()
    X_test_pitch = OrderedDict()
    y_test = OrderedDict()

    # Store the number of audio fragments per testing file
    num_fragments_per_file = [len(test_data[i]['mels']) for i, _ in enumerate(test_data)]

    # Fill the test data dictionaries
    for ix, _ in enumerate(test_data):
        # Initialize the feature and lable matrices for test file at index ix
        X_test_file_ix_mel = np.zeros((num_fragments_per_file[ix], 128, 100))
        X_test_file_ix_modgd = np.zeros((num_fragments_per_file[ix], 128, 98))
        X_test_file_ix_pitch = np.zeros((num_fragments_per_file[ix], 36, 100))
        y_test_file_ix = np.zeros((num_fragments_per_file[ix], 11))

        label = test_data[ix]["labels"]

        j = 0
        for feat in test_data[ix]['mels']:
            X_test_file_ix_mel[j,:,:] = feat
            y_test_file_ix[j,:] = label
            j+=1
        
        j = 0
        for feat in test_data[ix]['modgds']:
            X_test_file_ix_modgd[j,:,:] = feat
            j+=1

        j = 0
        for feat in test_data[ix]['pitchs']:
            X_test_file_ix_pitch[j,:,:] = feat
            j+=1

        X_test_mel[ix] = X_test_file_ix_mel
        X_test_modgd[ix] = X_test_file_ix_modgd
        X_test_pitch[ix] = X_test_file_ix_pitch
        y_test[ix] = y_test_file_ix

    print("Finished test data!")

    print("Saving test data!")

    f = open("../../data/processed/X_test_mel.pkl", "wb")
    pickle.dump(X_test_mel, f)
    f.close()

    f = open("../../data/processed/X_test_modgd.pkl", "wb")
    pickle.dump(X_test_modgd, f)
    f.close()

    f = open("../../data/processed/X_test_pitch.pkl", "wb")
    pickle.dump(X_test_pitch, f)
    f.close()

    f = open("../../data/processed/y_test.pkl", "wb")
    pickle.dump(y_test, f)
    f.close()

    print("Test data saved!")


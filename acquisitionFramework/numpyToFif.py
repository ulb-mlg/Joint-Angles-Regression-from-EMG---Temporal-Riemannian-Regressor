import numpy as np
import mne


name = ...

def loadData(path):
    emg = np.load(f"{path}/EMG_data.npy")
    label = np.load(f"{path}/label_data.npy")
    yts = np.load(f"{path}/label_ts_data.npy")
    shownPred = np.load(f"{path}/shown_pred_data.npy")
    confidence = np.load(f"{path}/label_confidence_data.npy")
    return emg, accel, gyro, label, yts, shownPred, confidence

dataset = "savedData/" + name

emg, accel, gyro, label, yts, shownPred, confidence = loadData(dataset)
trainTime = np.load(f"{dataset}/train_times.npy")

labelForFif = np.zeros((emg.shape[0], 15)) * np.nan
labelForFif[yts[yts < len(labelForFif)]] = label[yts < len(labelForFif)]

i = 0
while i < len(labelForFif):
    if not np.isnan(labelForFif[i, 0]):
        j = i+1
        while np.isnan(labelForFif[j, 0]) and j < len(labelForFif)-1:
            j+=1
        if j > i+1:
            dist = j-i
            diff = labelForFif[j] - labelForFif[i]
            for k in range(i+1, j):
                labelForFif[k] = labelForFif[i] + (k-i)*diff/dist
    i += 1

select = np.logical_not(np.isnan(labelForFif[:, 0]))

# Channel names and types
ch_names = [f'EMG {i+1}' for i in range(8)] \
            + [f"Angle {i+1}" for i in range(15)]
ch_types = ['emg'] * 8 + ["misc"] * 15

# Create MNE Info structure
info = mne.create_info(ch_names=ch_names, sfreq=500, ch_types=ch_types)


# Create Raw object from NumPy array
data = np.concatenate((emg[select].T, labelForFif[select].T))
raw = mne.io.RawArray(data, info)

# Create an annotation for model training time
for tt in trainTime:
    event_time = (yts[tt] - np.sum(np.isnan(labelForFif[5000, 0])))/500
    ann = mne.Annotations(onset=[event_time],
                          duration=[0],
                          description=["model trained"])
    raw.set_annotations(raw.annotations + ann)

# Save to .fif file
raw.save(f'{dataset}/{name}.fif', overwrite=True)
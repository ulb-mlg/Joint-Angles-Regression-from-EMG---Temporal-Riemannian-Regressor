import mne
name = ...
fif_file = f'savedData/{name}/{name}.fif'
raw = mne.io.read_raw_fif(fif_file, preload=True)
raw.plot(scalings='auto', title='All Channels Interactive Plot', show=True, block=True)
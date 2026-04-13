# Data acquisition framework for synchronized EMG and finger joint angles during free gestures

This framework enables to record EMG signals using the [__MindRove 8-channel EMG armband__](https://mindrove.com/product/emg-armband/). If you wish to use a different EMG system, you must adapt the `EMG.py` class

The recording of hand gestures uses [__MediaPipe__](https://mediapipe.readthedocs.io/en/latest/solutions/hands.html)

The visualization of hand gesture is based on the [__Leap Hand__](https://v1.leaphand.com/)

The framework can be configured using the `config.py` class

The software can then be started using the command
- `python main.py`

After a few seconds, the angles start to be streamed to the chosen visualization.
If EMG are activated in the configuration, EMG are continuously recorded and synchronized with joint angles.
You can save the recorded data by entering `s` in the console, or train a new EMG to joint angles model by entering `t` in the console.
To close the software, press `ESC` in the pybullet window, or press `ctrl+C` in the console.

Finally, `numpyToFif.py` can be used to convert the recorded `.npy` files to `.fif` files, compatible with the package `mne`, and 
`mne_visualizer.py` can be used to visualize the recorded EMG and kinematics signals.
"""
Configuration
"""

VIRTUAL = True # show virtual representation of the leap hand in pybullet
PHYSICAL = False # stream joint angles to the robotic leap hand
EMG = True # record EMG for machine learning
PRELOAD_EMG_MODEL = False # load a pretrained EMG->joint Angles model. Models are saved in savedModel after each training

# model
FREQ_BANDS = [(5, 150)]  # we used [(5, 150)] and [(15, 40), (40, 80), (80, 150)]
EMG_WINDOW_LENGTH = 150  # number of EMG frames per window (500Hz)
EMG_WINDOW_STEP = 50  # number of EMG frames between 2 windows in a single sequence (1 sample)
EMG_SEQUENCE_LENGTH = 10  # number of windows in a sequence that forms 1 sample

# windows
COM_CHANNEL = "COM7"

# linux
TTYUSB = "/dev/ttyUSB0"

from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from matplotlib import pyplot as plt
from IPython.core.debugger import set_trace
import random
from IPython.display import Image
import torchaudio
import librosa
import matplotlib.pyplot as plt
import librosa.display
import glob
import os.path

# try a dataset, save all spectrograms somewhere
filenames = glob.glob("/home/kchow/datasets/clips/*.mp3")
spectrogram_clips = []
mel_spectrogram_clips = []

restart = True

number_of_clips = 10

# if numpy loaded 
if not restart and os.path.isfile('spec_clips.npy') and os.path.isfile('mel_spec_clips.npy'):
    spectrogram_clips = np.load('spec_clips.npy')
    mel_spectrogram_clips = np.load('mel_spec_clips.npy')
else:
    for f in filenames:
        print(f)
        y, sr = librosa.load(f)
        D = np.abs(librosa.stft(y))**2
        D = D[:1024,:1939]
        S = librosa.feature.melspectrogram(S=D)
        S = S[:128,:1939]
        if np.shape(D)[0] != 1024 or np.shape(D)[1] != 1939 or np.shape(S)[0] != 128 or np.shape(S)[1] != 1939:
            continue
        spectrogram_clips.append(D)
        mel_spectrogram_clips.append(S)
        np.save('spec_clips_.npy', np.array(spectrogram_clips))
        np.save('mel_spec_clips_.npy', np.array(mel_spectrogram_clips))

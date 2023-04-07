# from models import Generator
from simple_models import Generator

import numpy as np
import matplotlib.pyplot as plt
import pyworld as pw
import torch
import torchaudio


def read_audio(wav_path):
    wav, fs = torchaudio.load(wav_path, normalize=True)
    print("og wav", wav.shape)

    wav = torchaudio.functional.resample(wav, orig_freq=fs, new_freq=44100)
    print("new wav", wav.shape)
    if wav.shape[0] == 2:
        wav = wav[:, 0]
    return wav, 44100

def plot_and_calculate_spectogram(wav_path):
    waveform, sample_rate = read_audio(wav_path)
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    print(num_channels)
    time_axis = torch.arange(0, num_frames) / sample_rate
    print(sample_rate)
    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        my_melspec, freqs, t, _ = axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
    plt.show(block=False)
    return my_melspec, freqs, t


def main():
    mel_spectrogram, _, _ = plot_and_calculate_spectogram(wav_path='../sample.wav')
    input_mel = torch.tensor(mel_spectrogram).float()

    speech_template = np.load('../all_pulse.npy')
    input_speech_template = torch.tensor(speech_template)
    
    print(input_speech_template.shape, input_mel.shape)
    generator = Generator()
    print(generator(input_mel))


if __name__ == '__main__':
    main()
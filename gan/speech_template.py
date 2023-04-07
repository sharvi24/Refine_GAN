import numpy as np
import matplotlib.pyplot as plt
import pyworld as pw
import torch
import torchaudio


def read_audio(wav_path):
    info = torchaudio.info(wav_path)
    duration_in_milisec = (info.num_frames / info.sample_rate) * 1000
    
    wav, sr = torchaudio.load(wav_path, normalize=True)
    # print("og wav", wav.shape)
    wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=44100)
    # print("new wav", wav.shape[0])
    
    ## Dealing with stereo audios
    if wav.shape[0] == 2:
        wav = wav[:, 0]
    return wav, 44100, duration_in_milisec

def compute_energy(wav):
    hop_length = 100
    frame_length = 100
    energy = np.array(
        [torch.sum(torch.square(wav[i:i+frame_length]))
                      for i in range(0, wav.shape[0], hop_length)] * hop_length
        )
    return energy

def add_noise_to_non_voices(wav, sample_rate):
    energy = compute_energy(wav)
    
    zeros_idxs_audio = []

    for i in range(wav.shape[0]):
        if wav[i] < 1e-5:
            zeros_idxs_audio.append(i)
    zeros_idxs_audio = np.array(zeros_idxs_audio)
    # set noise level
    noise_level = 1.0

    # generate uniform noise with same length as audio
    noise = np.random.uniform(0, noise_level, size=wav.shape[0])

    noisy_wav = np.zeros(wav.shape[0])

    for i in range(wav.shape[0]):
        noisy_wav[i] = wav[i]

    # add noise to audio signal
    for idx in zeros_idxs_audio:
        noisy_wav[idx] += (noise[idx] * energy[idx])

    # save noisy audio file

    noisy_wav = torch.from_numpy(np.array(noisy_wav, dtype='float32'))
    noisy_wav = torch.unsqueeze(noisy_wav, 0)
    torchaudio.save('noisy_audio_file.wav', noisy_wav, sample_rate)

    return noisy_wav

def plot_and_calculate_spectogram(wav_path):
    waveform, sample_rate, duration_in_milisec = read_audio(wav_path)
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape

    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        my_melspec, freqs, t, _ = axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
    plt.show(block=False)
    return my_melspec, freqs, t

# 1/ mean intensity of spec across time
def get_mean_intensity_time(t):
    return np.mean(t, 0)

def compute_speech_properties(wav_path, timestamps_count):
    wav, sr, duration_in_milisec = read_audio(wav_path)
    
    wav = wav.numpy()[0]
    wav = np.ascontiguousarray(wav.astype('float64'))
    f0, timeaxis = pw.dio(wav, sr, frame_period=3000/1032)
    
    print("comparison",duration_in_milisec/timestamps_count, 3000/1032)
    # wav = wav.numpy()
    # wav = np.ascontiguousarray(wav.astype('float64'))
    # print("shape of wav for pw", wav.shape)
    
    #f0, timeaxis = pw.dio(wav, sr, frame_period = duration_in_milisec/timestamps_count)
    f01 = pw.stonemask(wav, f0, timeaxis, sr)
    return [f0, f01, timeaxis]

def smoothen_pitch(raw_pitch):
    res = np.zeros(raw_pitch.shape)
    for i in range(res.shape[0]):
        res[i] = raw_pitch[i] + np.mean(raw_pitch)
    return res

def plt_sample_pulse(duration, value):
    # Set the time period and duration of the pulse
    time_period = duration  # seconds
    Amplitude = value
    pulse_duration = time_period  # seconds

    # Generate an array of time values from 0 to 1 second with a step size of 0.001 seconds
    t = np.arange(0, time_period, 0.001)

    # Generate the sine pulse
    frequency = 1.0 / time_period
    omega = 2 * np.pi * frequency
    phase = np.pi / 2  # phase shift to start at maximum
    pulse = Amplitude * np.sin(omega * t + phase)
    pulse[t > pulse_duration] = 0  # set pulse to zero after duration

    # Plot the pulse
    plt.plot(t, pulse)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('Sine Pulse with Time Period of 1 Second')
    plt.show()

def create_speech_template(intensity_per_time, time_periods, duration_in_milisec):
    all_pulse = list()
    for i in range(len(intensity_per_time)):
        # Set the time period and duration of the pulse
        time_period = time_periods[i]  # seconds
        amplitude = intensity_per_time[i]
        pulse_duration = time_period  # seconds

        # Generate an array of time values from 0 to 
        duration_in_sec = duration_in_milisec / 1000
        t = np.arange(0, time_period, time_period/(duration_in_sec * 44100)/len(intensity_per_time))

        # Generate the sine pulse
        frequency = 1.0 / time_period
        omega = 2 * np.pi * frequency
        phase = np.pi / 2  # phase shift to start at maximum
        pulse = amplitude * np.sin(omega * t + phase)
        pulse[t > pulse_duration] = 0  # set pulse to zero after duration
        all_pulse.extend(list(pulse))
    return np.array(all_pulse)

def get_speech_template(wav_path):
    # if the sampling frequency is 44100 hertz, a recording with a duration of 60 seconds will contain 2,646,000 samples.
    # therefore required number of samples for 3 sec audio = 44100*3 = 132300 samples
    
    wav, sample_rate, duration_in_milisec = read_audio(wav_path)
    
    noisy_wav = add_noise_to_non_voices(wav[0], sample_rate)
    
    my_melspec, freqs, t = plot_and_calculate_spectogram(wav_path)
    
    intensity_per_time = get_mean_intensity_time(my_melspec)
    
    raw_p, ref_p, timeaxis = compute_speech_properties(wav_path, timestamps_count = t)
    new_raw_p = smoothen_pitch(raw_p)
    time_periods = 1 / new_raw_p
    pulse_list = create_speech_template(intensity_per_time, time_periods, duration_in_milisec)
    return pulse_list

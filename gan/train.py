import torch

from models import Generator
from speech_template import (
    get_speech_template,
    plot_and_calculate_spectogram
)

def get_inputs(wav_path):
    speech_template = get_speech_template(wav_path)
    input_speech_template = torch.tensor(speech_template)
    input_speech_template = input_speech_template.unsqueeze(0)
    
    mel_spectrogram, _, _ = plot_and_calculate_spectogram(wav_path=wav_path)
    input_mel = torch.tensor(mel_spectrogram).float()
    return input_mel, input_speech_template

def main():
    wav_path = "../sample.wav"
    input_mel, input_speech_template = get_inputs(wav_path=wav_path)
    print('Successfully generated input mel and speech template')
    generator = Generator()
    generator = generator.float()
    print('Calling generator forward function')
    # forward call
    print(generator(input_mel.float(), input_speech_template.float()))


if __name__ == '__main__':
    main()
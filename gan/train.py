import torch

# from models import Generator
from simple_models import Generator
from speech_template import (
    get_speech_template,
    plot_and_calculate_spectogram
)

def main():
    wav_path = "../sample.wav"
    mel_spectrogram, _, _ = plot_and_calculate_spectogram(wav_path=wav_path)
    input_mel = torch.tensor(mel_spectrogram).float()

    speech_template = get_speech_template(wav_path)
    input_speech_template = torch.tensor(speech_template)
    
    print("input items shape", input_speech_template.shape, input_mel.shape)
    generator = Generator()
    # print(generator(input_mel))


if __name__ == '__main__':
    main()
import torch

# from models import Generator
from simple_models import Generator
from speech_template import (
    get_speech_template,
    plot_and_calculate_spectogram
)

def main():
    wav_path = "../sample.wav"
    
    speech_template = get_speech_template(wav_path)
    input_speech_template = torch.tensor(speech_template)
    print("Got input_speech_template --- success")
    print("input_speech_template shape", input_speech_template.shape) #torch.Size([133128])
    
    mel_spectrogram, _, _ = plot_and_calculate_spectogram(wav_path=wav_path)
    input_mel = torch.tensor(mel_spectrogram).float()
    print("Got input_mel --- success")
    print("input_mel shape", input_mel.shape)      #torch.Size([129, 1032])

    generator = Generator()
    print(generator(input_mel))
    #print(generator(input_speech_template))


if __name__ == '__main__':
    main()
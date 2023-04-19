# RefineGAN Implementation


This repository contains the implementation of RefineGAN in Pytorch.
Paper link: https://arxiv.org/pdf/2111.00962.pdf

**TLDR;** RefineGAN generates high-quality 44.1khz audio by using a convolutional U-Net architecture. The inputs are a log mel-spectrogram and the f0 (pitch over time) curve. This is used to calculate a speech template—a rough sketch of what the waveform looks like ************************************************************************************************************************in time domain at the output sample rate************************************************************************************************************************. A U-Net then refines this into the final time-domain output, conditioning on the mel-spectrogram at the innermost layer.

## Files:

- `gan/loss.py` has loss functions required
- `gan/models.py` has generator/discriminator structure
- `gan/speech_template.py` has creation of speech template
- `gan/train.py` has training of gan

## Usage

To start training, do:
```
bash run.sh
```

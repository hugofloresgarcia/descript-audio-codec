# Descript Audio Codec (.dac): High-Fidelity Audio Compression with Improved RVQGAN

This repository contains training and inference scripts
for the Descript Audio Codec (.dac), a high fidelity general
neural audio codec, introduced in the paper titled **High-Fidelity Audio Compression with Improved RVQGAN**.

![](https://static.arxiv.org/static/browse/0.3.4/images/icons/favicon-16x16.png) [arXiv Paper: High-Fidelity Audio Compression with Improved RVQGAN
](http://arxiv.org/abs/2306.06546) <br>
📈 [Demo Site](https://descript.notion.site/Descript-Audio-Codec-11389fce0ce2419891d6591a68f814d5)<br>
⚙ [Model Weights](https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.1/weights.pth)

👉 With Descript Audio Codec, you can compress **44.1 KHz audio** into discrete codes at a **low 8 kbps bitrate**.  <br>
🤌 That's approximately **90x compression** while maintaining exceptional fidelity and minimizing artifacts.  <br>
💪 Our universal model works on all domains (speech, environment, music, etc.), making it widely applicable to generative modeling of all audio.  <br>
👌 It can be used as a drop-in replacement for EnCodec for all audio language modeling applications (such as AudioLMs, MusicLMs, MusicGen, etc.) <br>

<p align="center">
<img src="./assets/comparsion_stats.png" alt="Comparison of compressions approaches. Our model achieves a higher compression factor compared to all baseline methods. Our model has a ~90x compression factor compared to 32x compression factor of EnCodec and 64x of SoundStream. Note that we operate at a target bitrate of 8 kbps, whereas EnCodec operates at 24 kbps and SoundStream at 6 kbps. We also operate at 44.1 kHz, whereas EnCodec operates at 48 kHz and SoundStream operates at 24 kHz." width=35%></p>


## Usage

### Installation

**important note**: you need my (hugo) fork of audiotools to use this fork of DAC. 
```
git clone https://github.com/hugofloresgarcia/audiotools.git
cd audiotools
pip install -e .
```

now, we can install DAC. 
```
git clone https://github.com/hugofloresgarcia/descript-audio-codec.git
cd descript-audio-codec
pip install -e .
```


### Programmatic Usage
```py
import dac
from audiotools import AudioSignal

# Download a model
model_path = dac.utils.download(model_type="44khz")
model = dac.DAC.load(model_path)

model.to('cuda')

# Load audio signal file
signal = AudioSignal('input.wav')

# Encode audio signal as one long file
# (may run out of GPU memory on long files)
signal.to(model.device)

x = model.preprocess(signal.audio_data, signal.sample_rate)
out = model.encode(x)
z = out["z"]

# Decode audio signal
y = model.decode(z)

# Alternatively, use the `compress` and `decompress` functions
# to compress long files.

signal = signal.cpu()
x = model.compress(signal)

# Save and load to and from disk
x.save("compressed.dac")
x = dac.DACFile.load("compressed.dac")

# Decompress it back to an AudioSignal
y = model.decompress(x)

# Write to file
y.write('output.wav')
```

## Training
The baseline model configuration can be trained using the following commands.

### Pre-requisites
Please install the correct dependencies
```
pip install -e ".[dev]"
```

## Environment setup (if you are using docker)

We have provided a Dockerfile and docker compose setup that makes running experiments easy.

To build the docker image do:

```
docker compose build
```

Then, to launch a container, do:

```
docker compose run -p 8888:8888 -p 6006:6006 dev
```

The port arguments (`-p`) are optional, but useful if you want to launch a Jupyter and Tensorboard instances within the container. The
default password for Jupyter is `password`, and the current directory
is mounted to `/u/home/src`, which also becomes the working directory.

Then, run your training command.


## Single GPU training
```
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --args.load conf/base.yml --save_path runs/base/
```

## Multi GPU training
```
torchrun --nproc_per_node gpu scripts/train.py --args.load conf/base.yml --save_path runs/base/
```

## Testing
We provide two test scripts to test CLI + training functionality. Please
make sure that the trainig pre-requisites are satisfied before launching these
tests. To launch these tests please run
```
python -m pytest tests
```

## Results

<p align="left">
<img src="./assets/objective_comparisons.png" width=75%></p>

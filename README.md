# Replica v0.0.1

All in one application to replicate human voice, image, even the mind.

## Usage
### ASR

## Installation
### MacOS Users
If you are a Mac user, make sure you meet the following conditions for training and inferencing with GPU:

- Mac computers with Apple silicon or AMD GPUs
- macOS 12.3 or later
- Xcode command-line tools installed by running `xcode-select --install`

_Other Macs can do inference with CPU only._

#### Create Environment
```bash
pyenv install 3.9
pyenv virtualenv 3.9 replica
pyenv activate replica
```

#### Install Packages
1. ffmpeg
```bash
brew install ffmpeg
```
2. torch & torchaudio
```bash
pip3 install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
```
To validate if your mac supports GPU, run ./test/mac_gpu_checker.py

3. other packages
```bash
pip install -r requirements.txt
```

## Pretrained Models
### ASR
For Chinese ASR (additionally), download models from [Damo ASR Model](https://modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/files), [Damo VAD Model](https://modelscope.cn/models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/files), and [Damo Punc Model](https://modelscope.cn/models/damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/files) and place them in `tools/damo_asr/models`.

## Credits

Special thanks to the following projects and contributors:

- [FFmpeg] (https://github.com/FFmpeg/FFmpeg)
- [GPT-SoVITS] (https://github.com/RVC-Boss/GPT-SoVITS)
- [audio-slicer] (https://github.com/openvpi/audio-slicer)
- [FunASR] (https://github.com/alibaba-damo-academy/FunASR)

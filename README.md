# Replica v0.0.1

All in one application to replicate human voice, image, even the mind.

## Usage
### ASR
https://github.com/Jeru2023/replica/assets/123569003/a7a77828-6558-44c3-b683-6b98726ac882

output: 主办方要我来讲一讲我自己的人生故事。我想了想，我只能说我在学生时期的时候，从来没有想过，有一天我能够拥有这样精彩的人生舞台。

```bash
import utils
import os
from service.asr_model import ASRModel

asr_model = ASRModel()

audio_sample = os.path.join(utils.get_root_path(), 'data', 'audio', 'sample_short.wav')
output = asr_model.inference(audio_sample)
print(output)
```
### Slicer
When processing long audio files we need to slice the audio into trunks.
```bash
import utils
from tools import slice_tool

in_path = utils.get_root_path() + '/data/audio/sample_long.wav'
out_folder = utils.get_root_path() + '/data/audio/slice_trunks'
out_file_prefix = 'sample'
slice_tool.slice_audio(in_path, out_folder, out_file_prefix, threshold=-40)
```

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
For Chinese ASR (additionally), 3 FunASR damo models will be downloaded automatically.
```bash
# voice recognition model
ASR_MODEL_NAME = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
# voice end-point detection model
VAD_MODEL_NAME = "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"
# punctuation model
PUNC_MODEL_NAME = "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
```

## Credits

Special thanks to the following projects and contributors:

- [FFmpeg] (https://github.com/FFmpeg/FFmpeg)
- [GPT-SoVITS] (https://github.com/RVC-Boss/GPT-SoVITS)
- [audio-slicer] (https://github.com/openvpi/audio-slicer)
- [FunASR] (https://github.com/alibaba-damo-academy/FunASR)

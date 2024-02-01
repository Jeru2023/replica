import utils
import os
from tools import asr_tool

asr_tool = asr_tool.ASRTool()

audio_sample = os.path.join(utils.get_root_path(), 'data', 'audio', 'sample.wav')
output = asr_tool.inference(audio_sample)
print(output)

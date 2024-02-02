import utils
import os
from infer.asr_model import ASRModel

asr_model = ASRModel()

audio_sample = os.path.join(utils.get_root_path(), 'data', 'audio', 'sample_short.wav')
output = asr_model.inference(audio_sample)
print(output)


audio_sample = os.path.join(utils.get_root_path(), 'data', 'audio', 'sample_long.wav')
output = asr_model.inference(audio_sample)
print(output)


from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
import pkg_resources

def get_root_path():
    package_path = pkg_resources.resource_filename(__name__, "")
    parent_path = os.path.dirname(package_path)
    return parent_path

asr_path = 'tools\damo_asr\models\speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'
vad_path = 'tools\damo_asr\models\speech_fsmn_vad_zh-cn-16k-common-pytorch'
punc_path = 'tools\damo_asr\models\punc_ct-transformer_zh-cn-common-vocab272727-pytorch'

root_path = get_root_path()
print(root_path)

inference_pipeline = pipeline(
    task = Tasks.auto_speech_recognition,
    model = os.path.join(root_path, asr_path),
    vad_model = os.path.join(root_path, vad_path),
    punc_model = os.path.join(root_path, punc_path)
)

audio_sample = os.path.join(root_path, 'data\\audio\sample.wav')
print(audio_sample)

# , batch_size_token=5000, batch_size_token_threshold_s=40, max_single_segment_time=6000
rec_result = inference_pipeline(input=audio_sample)
print(rec_result)





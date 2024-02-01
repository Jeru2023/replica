from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
import utils


root_path = utils.get_root_path()
print(root_path)

# set damo asr models path
asr_path = os.path.join(root_path, 'models', 'damo_asr',
                        'speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
vad_path = os.path.join(root_path, 'models', 'damo_asr',
                        'speech_fsmn_vad_zh-cn-16k-common-pytorch')
punc_path = os.path.join(root_path, 'models', 'damo_asr',
                        'punc_ct-transformer_zh-cn-common-vocab272727-pytorch')

print(asr_path)

# inference_pipeline = pipeline(
#     task = Tasks.auto_speech_recognition,
#     model = asr_path,
#     vad_model = vad_path,
#     punc_model = punc_path
# )
#
audio_sample = os.path.join(root_path, r'data/audio/sample.wav')
print(audio_sample)
#
# # , batch_size_token=5000, batch_size_token_threshold_s=40, max_single_segment_time=6000
# rec_result = inference_pipeline(input=audio_sample)
# print(rec_result)





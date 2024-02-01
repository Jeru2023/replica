from funasr import AutoModel
import os


# voice recognition model
ASR_MODEL_NAME = "damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
# voice end-point detection model
VAD_MODEL_NAME = "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"
# punctuation model
PUNC_MODEL_NAME = "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"


class ASRTool:
    def __init__(self):
        pass

    @staticmethod
    def get_model():
        model = AutoModel(model=ASR_MODEL_NAME, model_revision="v2.0.2",
                          vad_model=VAD_MODEL_NAME, vad_model_revision="v2.0.2",
                          punc_model=PUNC_MODEL_NAME, punc_model_revision="v2.0.3",
                          )
        return model

    # inference param: batch_size_token = 5000, batch_size_token_threshold_s = 40, max_single_segment_time = 6000
    def inference(self, audio_input):
        model = self.get_model()
        result = model.generate(audio_input)[0]['text']
        return result


if __name__ == '__main__':
    import utils
    asr_tool = ASRTool()
    audio_sample = os.path.join(utils.get_root_path(), 'data', 'audio', 'sample.wav')
    output = asr_tool.inference(audio_sample)
    print(output)

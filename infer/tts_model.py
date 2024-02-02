from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import utils
import os
import torch
from infer.persona_enum import PersonaEnum
import librosa
from module import cnhubert
from module.models import SynthesizerTrn
from module.mel_processing import spectrogram_torch
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from AR.models.t2s_lightning_module import Text2SemanticLightningModule


dict_language = {
    "中文": "zh",
    "英文": "en",
    "日文": "ja",
    "ZH": "zh",
    "EN": "en",
    "JA": "ja",
    "zh": "zh",
    "en": "en",
    "ja": "ja"
}


class DictToAttrRecursive:
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            if isinstance(value, dict):
                # 如果值是字典，递归调用构造函数
                setattr(self, key, DictToAttrRecursive(value))
            else:
                setattr(self, key, value)


class TTSModel:
    def __init__(self):
        self.root_path = utils.get_root_path()
        self.sovits_path = os.path.join(self.root_path, 'models', 's2G488k.pth')
        self.gpt_path = os.path.join(self.root_path, 'models', 's1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt')
        self.bert_path = os.path.join(self.root_path, 'models', 'chinese-roberta-wwm-ext-large')
        self.cnhubert_base_path = os.path.join(self.root_path, 'models', 'chinese-hubert-base')

        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_path)
        self.bert_model = AutoModelForMaskedLM.from_pretrained(self.bert_path)
        self.device = self.get_device()

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'

    def load_hps(self):
        dict_s2 = torch.load(self.sovits_path, map_location="cpu")
        hps = dict_s2["config"]
        hps = DictToAttrRecursive(hps)
        hps.model.semantic_frame_rate = "25hz"
        return hps

    def load_ssl_model(self):
        cnhubert.cnhubert_base_path = self.cnhubert_base_path
        ssl_model = cnhubert.get_model()
        ssl_model = ssl_model.to(self.device)
        return ssl_model

    def load_vq_model(self):
        sovits_dict = self.get_sovits_dict()
        hps = self.load_hps()
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length, n_speakers=hps.data.n_speakers, **hps.model)

        vq_model = vq_model.to(self.device)
        vq_model.eval()
        print(vq_model.load_state_dict(sovits_dict["weight"], strict=False))
        return vq_model

    def load_t2s_model(self):
        t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
        gpt_dict = self.get_gpt_dict()
        t2s_model.load_state_dict(gpt_dict["weight"])

        t2s_model = t2s_model.to(self.device)
        t2s_model.eval()
        total = sum([param.nelement() for param in t2s_model.parameters()])
        print("Number of parameter: %.2fM" % (total / 1e6))
        return t2s_model

    def get_gpt_dict(self):
        return torch.load(self.gpt_path, map_location="cpu")

    def get_sovits_dict(self):
        return torch.load(self.sovits_path, map_location="cpu")


    def get_bert_feature(self, text, word2ph):
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt")
            for i in inputs:
                inputs[i] = inputs[i].to(self.device)  # 输入是long不用管精度问题，精度随bert_model
            res = self.bert_model(**inputs, output_hidden_states=True)
            res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

        assert len(word2ph) == len(text)

        phone_level_feature = []
        for i in range(len(word2ph)):
            repeat_feature = res[i].repeat(word2ph[i], 1)
            phone_level_feature.append(repeat_feature)

        phone_level_feature = torch.cat(phone_level_feature, dim=0)
        # if(is_half==True):phone_level_feature=phone_level_feature.half()
        return phone_level_feature.T

    def get_spepc(hps, filename):
        hps = self.load_hps()
        audio = load_audio(filename, int(hps.data.sampling_rate))
        audio = torch.FloatTensor(audio)
        audio_norm = audio
        audio_norm = audio_norm.unsqueeze(0)
        spec = spectrogram_torch(audio_norm, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length,
                                 hps.data.win_length, center=False)
        return spec

    def get_tts_wav(self, persona_name, text, text_language):
        persona = PersonaEnum.get_persona_by_name(persona_name)
        prompt_text = persona.get_ref_text().strip("\n")
        ref_audio_path = persona.get_ref_audio()
        prompt_language = persona.get_language()
        texts = text.strip("\n")

        hps = self.load_hps()
        zero_wav = np.zeros(int(hps.data.sampling_rate * 0.3), dtype=np.float32)
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_audio_path, sr=16000)
            wav16k = torch.from_numpy(wav16k)
            zero_wav_torch = torch.from_numpy(zero_wav)

            wav16k = wav16k.to(self.device)
            zero_wav_torch = zero_wav_torch.to(self.device)

            wav16k = torch.cat([wav16k, zero_wav_torch])
            ssl_model = self.load_ssl_model()
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].transpose(1, 2)  # .float()

            vq_model = self.load_vq_model()
            codes = vq_model.extract_latent(ssl_content)
            prompt_semantic = codes[0, 0]

        prompt_language = dict_language[prompt_language]
        text_language = dict_language[text_language]
        phones1, word2ph1, norm_text1 = clean_text(prompt_text, prompt_language)
        phones1 = cleaned_text_to_sequence(phones1)

        audio_opt = []
        for text in texts:
            phones2, word2ph2, norm_text2 = clean_text(text, text_language)
            phones2 = cleaned_text_to_sequence(phones2)

            bert1 = self.get_bert_feature(norm_text1, word2ph1).to(self.device)
            bert2 = self.get_bert_feature(norm_text2, word2ph2).to(self.device)
            bert = torch.cat([bert1, bert2], 1)

            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(self.device).unsqueeze(0)
            bert = bert.to(self.device).unsqueeze(0)
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(self.device)
            prompt = prompt_semantic.unsqueeze(0).to(self.device)

            hz = 50
            config = self.get_gpt_dict()['config']
            max_sec = config['data']['max_sec']
            t2s_model = self.load_t2s_model()

            with torch.no_grad():
                # pred_semantic = t2s_model.model.infer(
                pred_semantic, idx = t2s_model.model.infer_panel(
                    all_phoneme_ids,
                    all_phoneme_len,
                    prompt,
                    bert,
                    # prompt_phone_len=ph_offset,
                    top_k=config['inference']['top_k'],
                    early_stop_num=hz * max_sec)

            # print(pred_semantic.shape,idx)

            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)  # .unsqueeze(0)#mq要多unsqueeze一次
            refer = get_spepc(hps, ref_audio_path)  # .to(device)

            else:
                refer = refer.to(self.device)
            # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
            audio = \
                vq_model.decode(pred_semantic, torch.LongTensor(phones2).to(self.device).unsqueeze(0),
                                refer).detach().cpu().numpy()[
                    0, 0]  ###试试重建不带上prompt部分
            audio_opt.append(audio)
            audio_opt.append(zero_wav)

        #print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
        yield hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)
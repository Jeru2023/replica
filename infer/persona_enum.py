from enum import Enum
import utils
import os


root_path = utils.get_root_path()


# 表定义
class PersonaEnum(Enum):
    LING_ZHI_LING = {
        "name": "林志玲",
        "gender": "女",
        "ref_audio": os.path.join(root_path, 'data', 'audio', 'persona', 'ling_zhi_ling.wav'),
        "ref_text": "主办方要我来讲一讲我自己的人生故事。我想了想，我只能说我在学生时期的时候，从来没有想过，有一天我能够拥有这样精彩的人生舞台。",
        "language": "zh",
    }


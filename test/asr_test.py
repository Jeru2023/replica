import utils
import os
from service.asr_model import ASRModel

asr_model = ASRModel()

# audio_sample = os.path.join(utils.get_root_path(), 'data', 'audio', 'sample_short.wav')
# output = asr_model.inference(audio_sample)
# print(output)


#audio_sample = os.path.join(utils.get_root_path(), 'data', 'audio', 'sample_long.wav')
audio_sample = r'D:\dev\GPT-SoVITS\output\jiacheng.wav'
output = asr_model.inference(audio_sample)
print(output)

# def get_file_paths(directory):
#     file_paths = []  # 用于存储文件路径的列表
#
#     # 遍历目录及其子目录中的所有文件
#     for root, directories, files in os.walk(directory):
#         for file in files:
#             # 构建文件的完整路径
#             file_path = os.path.join(root, file)
#             file_paths.append(file_path)
#
#     return file_paths
#
#
# root_path = r'D:\dev\GPT-SoVITS\output\slicer_opt'
#
# file_paths = get_file_paths(root_path)
#
# output_list = []
# # 打印文件路径
# for file_path in file_paths:
#     output = asr_model.inference(file_path)
#     output_list.append(output)
#
# print('\n'.join(output_list))

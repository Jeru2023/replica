import subprocess
import os

def get_root_path():
    # get current file absolute path
    current_file = os.path.abspath(__file__)
    
    # 获取当前脚本所在目录的上一级目录，即项目的根路径
    root_path = os.path.dirname(os.path.dirname(current_file))
    
    return root_path

def convert_mp4_to_wav(input_file, output_file):
    try:
        # use FFmpeg command to convert MP4 to WAV format 
        subprocess.call(['ffmpeg', '-i', input_file, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', output_file])
        print("convertion completed.")
    except Exception as e:
        print("convertion error:", str(e))



root_path = get_root_path()
print(root_path)

input_path = os.path.join(root_path, "data\\video\zhiling.mp4")
output_path = os.path.join(root_path, "data\\audio\zhiling.wav")

print(input_path)
convert_mp4_to_wav(input_path, output_path)        

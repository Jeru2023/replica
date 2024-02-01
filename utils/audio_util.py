import subprocess

def convert_mp4_to_wav(input_file, output_file):
    try:
        # use FFmpeg command to convert MP4 to WAV format 
        subprocess.call(['ffmpeg', '-i', input_file, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', output_file])
        print("convertion completed.")
    except Exception as e:
        print("convertion error:", str(e))
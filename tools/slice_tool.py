import librosa  # Optional. Use any library you like to read audio files.
import soundfile
from tools.slicer import Slicer
import os


def slice_audio(in_path, out_folder, out_file_prefix, threshold=-40, min_length=5000, min_interval=300, hop_size=10, max_sil_kept=500):
    """
    :param in_path:
    :param out_folder:
    :param out_file_prefix:
    :param threshold: key param that decides where to cut the audio, the smaller threshold get more chunks.
    :param min_length:
    :param min_interval:
    :param hop_size:
    :param max_sil_kept:
    :return:
    """
    audio, sr = librosa.load(in_path, sr=None, mono=False)  # Load an audio file with librosa.
    slicer = Slicer(
        sr=sr,
        threshold=threshold,
        min_length=min_length,
        min_interval=min_interval,
        hop_size=hop_size,
        max_sil_kept=max_sil_kept
    )
    chunks = slicer.slice(audio)

    for i, chunk in enumerate(chunks):
        if len(chunk.shape) > 1:
            chunk = chunk.T  # Swap axes if the audio is stereo.
        out_file_path = os.path.join(out_folder, f'{out_file_prefix}_{i}.wav')
        soundfile.write(out_file_path, chunk, sr)
import librosa  # Optional. Use any library you like to read audio files.
import soundfile
from tools.slicer import Slicer


def slice_audio(in_path, out_path, threshold=-40, min_length=5000, min_interval=300, hop_size=10, max_sil_kept=500):
    """
    :param in_path:
    :param out_path:
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
        soundfile.write(out_path, chunk, sr)
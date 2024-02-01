import utils
from tools import slice_tool

in_path = utils.get_root_path() + '/data/audio/sample_long.wav'
out_folder = utils.get_root_path() + '/data/audio/slice_trunks'
out_file_prefix = 'sample'
slice_tool.slice_audio(in_path, out_folder, out_file_prefix, threshold=-40)

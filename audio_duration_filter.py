import os
import shutil
from pydub import AudioSegment

def process_audio_file(src_path, dst_path, audio_file):
    duration = AudioSegment.from_wav(src_path).duration_seconds
    
    if 3 <= duration <= 10:
        shutil.move(src_path, dst_path)
        print(f"Moved {audio_file} to {dst_path}")
    else:
        os.remove(src_path)
        print(f"Deleted {audio_file} (duration: {duration:.2f}s)")

def process_audio_files(src_folder, dst_folder):
    # 遍历源文件夹中的所有音频文件
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file.endswith('.wav'):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(dst_folder, file)
                process_audio_file(src_path, dst_path, file)

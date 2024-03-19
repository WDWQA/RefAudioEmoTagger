import os
import logging
import argparse
from pydub import AudioSegment
from pydub.playback import play
from shutil import copy2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def filter_audio(src_folder, dst_folder=None, min_duration=3, max_duration=10, disable_filter=False):
    if not os.path.exists(src_folder):
        logging.error(f"源文件夹不存在: {src_folder}")
        return

    for root, _, files in os.walk(src_folder):
        for file in filter(lambda f: f.endswith('.wav'), files):
            src_path = os.path.join(root, file)
            
            if disable_filter:
                dst_path = src_path
            else:
                if dst_folder is None:
                    dst_path = os.path.join(root, f"filtered_{file}")
                else:
                    rel_path = os.path.relpath(root, src_folder)
                    dst_root = os.path.join(dst_folder, rel_path)
                    os.makedirs(dst_root, exist_ok=True)
                    dst_path = os.path.join(dst_root, f"filtered_{file}")
                
                duration = AudioSegment.from_wav(src_path).duration_seconds
                if min_duration <= duration <= max_duration:
                    copy2(src_path, dst_path)
                    logging.info(f"已复制 {src_path} 到 {dst_path}")
                    
                    # 仅在启用筛选时播放音频
                    audio = AudioSegment.from_wav(dst_path)
                    play(audio)
                else:
                    logging.warning(f"已跳过 {src_path} (时长: {duration:.2f}秒)")

def rename_wav_with_txt(directory):
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.lab'):
                basename = os.path.splitext(filename)[0]
                wav_path = os.path.join(root, f"{basename}.wav")
                txt_path = os.path.join(root, filename)
                
                if os.path.exists(wav_path):
                    with open(txt_path, 'r', encoding='utf-8') as txt_file:
                        new_name = txt_file.read().strip()
                        
                        if os.path.isfile(new_name) or os.path.isdir(new_name) or not os.path.basename(new_name):
                            print(f"跳过无效的文件名: {new_name}")
                        else:
                            try:
                                os.rename(wav_path, os.path.join(root, f"{new_name}.wav"))
                                print(f"已重命名 {wav_path} 为 {new_name}.wav")
                            except OSError as e:
                                print(f"重命名 {wav_path} 时出错: {e}")
                else:
                    print(f"找不到与 {filename} 对应的WAV文件")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='筛选并重命名音频文件')
    parser.add_argument('src_folder', help='源文件夹路径')
    parser.add_argument('-dst', '--dst_folder', help='目标文件夹路径')
    parser.add_argument('-min', '--min_duration', type=float, default=3, help='最小时长(秒),默认为3秒')
    parser.add_argument('-max', '--max_duration', type=float, default=10, help='最大时长(秒),默认为10秒')
    parser.add_argument('-d', '--disable_filter', action='store_true', help='禁用音频筛选')

    args = parser.parse_args()

    if args.disable_filter:
        filter_audio(args.src_folder, min_duration=args.min_duration, max_duration=args.max_duration, disable_filter=True)
        rename_wav_with_txt(args.src_folder)
    else:
        filter_audio(args.src_folder, args.dst_folder, args.min_duration, args.max_duration)
        rename_wav_with_txt(args.dst_folder if args.dst_folder else args.src_folder)

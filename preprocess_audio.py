import os
import logging
import argparse
from pydub import AudioSegment
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def filter_audio(src_folder, dst_folder=None, min_duration=3, max_duration=10, disable_filter=False, copy_parent_folder=False):
    if not os.path.exists(src_folder):
        logging.error(f"源文件夹不存在: {src_folder}")
        return

    audio_files = glob.glob(os.path.join(src_folder, "**", "*.wav"), recursive=True)
    total_files = len(audio_files)
    
    for index, src_path in enumerate(audio_files, start=1):
        if disable_filter:
            dst_path = src_path
        else:
            if dst_folder is None:
                dst_path = os.path.join(os.path.dirname(src_path), f"filtered_{os.path.basename(src_path)}")
            else:
                rel_path = os.path.relpath(src_path, src_folder)
                if copy_parent_folder:
                    parent_folder = os.path.basename(src_folder)
                    dst_parent_folder = os.path.join(dst_folder, parent_folder)
                    dst_path = os.path.join(dst_parent_folder, rel_path)
                else:
                    dst_path = os.path.join(dst_folder, rel_path)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            duration = AudioSegment.from_wav(src_path).duration_seconds  
            if min_duration <= duration <= max_duration:
                AudioSegment.from_wav(src_path).export(dst_path, format="wav")
                logging.info(f"[{index}/{total_files}] 已复制 {src_path} 到 {dst_path}")
            else:
                logging.warning(f"[{index}/{total_files}] 已跳过 {src_path} (时长: {duration:.2f}秒)")

    if not disable_filter:
        logging.info(f"过滤后的音频已保存在 {dst_folder}")

def rename_wav_with_lab(directory):
    renamed_count = 0
    txt_files = glob.glob(os.path.join(directory, "**", "*.lab"), recursive=True)
    total_files = len(txt_files)
    
    for index, txt_file in enumerate(txt_files, start=1):
        wav_file = os.path.splitext(txt_file)[0] + ".wav"

        if os.path.exists(wav_file):
            with open(txt_file, 'r', encoding='utf-8') as f:
                new_name = f.read().strip()

            new_wav_file = os.path.join(os.path.dirname(wav_file), f"{new_name}.wav")
            
            if new_wav_file != wav_file:
                try:
                    os.rename(wav_file, new_wav_file)
                    renamed_count += 1
                    logging.info(f"[{index}/{total_files}] 已重命名 {wav_file} 为 {new_wav_file}")
                except OSError as e:
                    logging.error(f"[{index}/{total_files}] 重命名 {wav_file} 时出错: {e}")
            
            try:
                os.remove(txt_file)
                logging.info(f"[{index}/{total_files}] 已删除 {txt_file}")
            except OSError as e:
                logging.error(f"[{index}/{total_files}] 删除 {txt_file} 时出错: {e}")
        else:
            logging.warning(f"[{index}/{total_files}] 找不到与 {txt_file} 对应的WAV文件")

    logging.info(f"共重命名了 {renamed_count} 个文件")
            
    return renamed_count

def rename_wav_with_list(list_file, wav_folder):
    renamed_count = 0
    
    with open(list_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        total_files = len(lines)
        
        for index, line in enumerate(lines, start=1):
            parts = line.strip().split('|')
            if len(parts) >= 4:
                wav_name = os.path.basename(parts[0])
                new_name = parts[-1]

                wav_files = glob.glob(os.path.join(wav_folder, "**", wav_name), recursive=True)

                if wav_files:
                    wav_file = wav_files[0]
                    new_name = new_name.replace('/', '_').replace('\\', '_')  # 替换文件名中的特殊字符
                    new_wav_file = os.path.join(os.path.dirname(wav_file), f"{new_name}.wav")
                    
                    if new_wav_file != wav_file:
                        try:
                            os.rename(wav_file, new_wav_file)
                            renamed_count += 1
                            logging.info(f"[{index}/{total_files}] 已重命名 {wav_file} 为 {new_wav_file}")
                        except OSError as e:
                            logging.error(f"[{index}/{total_files}] 重命名 {wav_file} 时出错: {e}")
                else:
                    logging.warning(f"[{index}/{total_files}] 找不到音频文件 {wav_name}")
            else:
                logging.warning(f"[{index}/{total_files}] 格式错误: {line}")

    logging.info(f"共重命名了 {renamed_count} 个文件")
    return renamed_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='筛选并重命名音频文件')
    parser.add_argument('src_folder', help='源文件夹路径')  
    parser.add_argument('-dst', '--dst_folder', help='目标文件夹路径')
    parser.add_argument('-min', '--min_duration', type=float, default=3, help='最小时长(秒),默认为3秒')
    parser.add_argument('-max', '--max_duration', type=float, default=10, help='最大时长(秒),默认为10秒') 
    parser.add_argument('-d', '--disable_filter', action='store_true', help='禁用音频筛选')

    args = parser.parse_args()

    src_items = len(os.listdir(args.src_folder))
    copy_parent_folder = src_items > 5
    
    if args.disable_filter:
        filter_audio(args.src_folder, disable_filter=True, copy_parent_folder=copy_parent_folder)
    else:
        os.makedirs(args.dst_folder, exist_ok=True)  
        filter_audio(args.src_folder, args.dst_folder, args.min_duration, args.max_duration, copy_parent_folder=copy_parent_folder)

    target_folder = args.src_folder if args.disable_filter else args.dst_folder        
    renamed_count = rename_wav_with_lab(target_folder)
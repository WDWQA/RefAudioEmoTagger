import os
import csv
import shutil
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_audio_file(audio_file, character, audio_emotion, text_emotion, output_path):
    src_path = Path(audio_file)
    
    if not src_path.exists():
        logging.warning(f"源文件不存在: {src_path}")
        return
    
    if text_emotion is not None:
        if audio_emotion != text_emotion and audio_emotion != '中立' and text_emotion:
            logging.info(f"跳过 {src_path},情感不匹配: AudioEmotion={audio_emotion}, TextEmotion={text_emotion}")
            return
    
    emotion_folder = Path(output_path) / character / audio_emotion
    emotion_folder.mkdir(parents=True, exist_ok=True)

    audio_name = src_path.name
    new_audio_name = f"【{audio_emotion}】{audio_name}"
    dst_path = emotion_folder / new_audio_name

    if not dst_path.exists():
        try:
            shutil.copy(str(src_path), str(dst_path))
            logging.info(f"已复制 {src_path} 到 {dst_path}")
        except OSError as e:
            logging.error(f"复制文件时出错: {e}")
    else:
        logging.warning(f"文件已存在: {dst_path}")

async def classify_audio_emotion(log_file, output_path, max_workers=4):
    log_path = Path(log_file)
    
    if not log_path.exists():
        logging.error(f"日志文件不存在: {log_path}")
        return
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(log_path, 'r', encoding='utf-8') as f_in:
        reader = csv.reader(f_in, delimiter='|')
        header = next(reader)
        
        text_emotion_index = None
        if "TextEmotion" in header:
            text_emotion_index = header.index("TextEmotion")
        
        audio_path_index = header.index("AudioPath")
        audio_emotion_index = header.index("AudioEmotion")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for row in reader:
                audio_path = row[audio_path_index]
                audio_emotion = row[audio_emotion_index]
                character = row[3] if len(row) > 3 else "Unknown"
                text_emotion = row[text_emotion_index] if text_emotion_index is not None else None
                future = executor.submit(process_audio_file, audio_path, character, audio_emotion, text_emotion, output_path)
                futures.append(future)

            for future in futures:
                await asyncio.wrap_future(future)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='按情感分类音频文件')
    parser.add_argument('--log_file', type=str, required=True, help='日志文件路径')
    parser.add_argument('--output_path', type=str, required=True, help='输出目录路径')
    parser.add_argument('--max_workers', type=int, default=4, help='最大工作线程数')

    args = parser.parse_args()

    asyncio.run(classify_audio_emotion(args.log_file, args.output_path, args.max_workers))
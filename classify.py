import os
import csv
import shutil
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_audio_file(audio_file, character, emotion, dataset_path):
    src_path = Path(dataset_path) / audio_file
    character_folder = Path(dataset_path) / character
    emotion_folder = character_folder / emotion
    emotion_folder.mkdir(parents=True, exist_ok=True)

    audio_name = audio_file.name
    new_audio_name = f"【{emotion}】{audio_name}"
    dst_path = emotion_folder / new_audio_name

    try:
        if src_path.exists():
            shutil.move(str(src_path), str(dst_path))
            logging.info(f"Moved {audio_file} to {dst_path}")
        else:
            logging.warning(f"File not found: {src_path}")
    except (shutil.Error, OSError) as e:
        logging.error(f"Error moving file: {e}")

def classify_audio_emotion(log_file, dataset_path, max_workers=4):
    log_path = Path(log_file).with_suffix('.log')
    file_handler = logging.FileHandler(str(log_path), mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    with open(log_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='|')
        next(reader)  # 跳过标题行
        audio_data = ((audio_path, character, emotion) for audio_path, character, emotion, _ in reader)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for data in audio_data:
            logger = logging.getLogger(f'Task: {data[0]}')
            logger.addHandler(file_handler)
            executor.submit(process_audio_file, *data, dataset_path, logger=logger)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify audio files by emotion')
    parser.add_argument('--log_file', type=str, required=True, help='Path to the log file')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum number of worker threads')

    args = parser.parse_args()

    classify_audio_emotion(args.log_file, args.dataset_path, args.max_workers)

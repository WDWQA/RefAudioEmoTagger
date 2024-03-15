import os
import csv
import shutil
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_audio_file(audio_file, character, emotion, dataset_path):
    src_path = os.path.join(dataset_path, audio_file)
    character_folder = os.path.join(dataset_path, character)
    emotion_folder = os.path.join(character_folder, emotion)
    os.makedirs(emotion_folder, exist_ok=True)

    audio_name = os.path.basename(audio_file)
    new_audio_name = f"【{emotion}】{audio_name}"
    dst_path = os.path.join(emotion_folder, new_audio_name)

    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        logging.info(f"Moved {audio_file} to {dst_path}")
    else:
        logging.warning(f"File not found: {src_path}")

def classify_audio_emotion(log_file, dataset_path, max_workers=4):
    with open(log_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='|')
        next(reader)  # 跳过标题行
        audio_data = [(audio_path, character, emotion) for audio_path, character, emotion, _ in reader]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(lambda data: process_audio_file(*data, dataset_path), audio_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify audio files by emotion')
    parser.add_argument('--log_file', type=str, required=True, help='Path to the log file')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset directory')
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum number of worker threads')

    args = parser.parse_args()

    classify_audio_emotion(args.log_file, args.dataset_path, args.max_workers)

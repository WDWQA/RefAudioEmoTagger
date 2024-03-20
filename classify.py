import os
import csv
import shutil
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_audio_file(audio_file, character, audio_emotion, text_emotion, output_path):
    src_path = Path(audio_file)
    
    if not src_path.exists():
        logging.warning(f"Source file does not exist: {src_path}")
        return
    
    if text_emotion is not None:
        if audio_emotion != text_emotion and audio_emotion != '中立' and text_emotion:
            logging.info(f"Skipping {src_path} due to emotion mismatch: AudioEmotion={audio_emotion}, TextEmotion={text_emotion}")
            return
    
    emotion_folder = Path(output_path) / character / audio_emotion
    emotion_folder.mkdir(parents=True, exist_ok=True)

    audio_name = src_path.name
    new_audio_name = f"【{audio_emotion}】{audio_name}"
    dst_path = emotion_folder / new_audio_name

    if not dst_path.exists():
        try:
            shutil.copy(str(src_path), str(dst_path))
            logging.info(f"Copied {src_path} to {dst_path}")
        except OSError as e:
            logging.error(f"Error copying file: {e}")
    else:
        logging.warning(f"File already exists: {dst_path}")

def classify_audio_emotion(log_file, output_path, max_workers=4):
    log_path = Path(log_file)
    
    if not log_path.exists():
        logging.error(f"Log file does not exist: {log_path}")
        return
    
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(log_path, 'r', encoding='utf-8') as f_in:
        reader = csv.reader(f_in, delimiter='|')
        header = next(reader)
        
        text_emotion_index = None
        if "TextEmotion" in header:
            text_emotion_index = header.index("TextEmotion")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for row in reader:
                audio_path = row[0]
                audio_emotion = row[1]
                character = row[3] if len(row) > 3 else "Unknown"
                text_emotion = row[text_emotion_index] if text_emotion_index is not None else None
                future = executor.submit(process_audio_file, audio_path, character, audio_emotion, text_emotion, output_path)
                futures.append(future)

            for future in as_completed(futures):
                exception = future.exception()
                if exception:
                    logging.error(f"Exception occurred: {exception}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Classify audio files by emotion')
    parser.add_argument('--log_file', type=str, required=True, help='Path to the log file')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--max_workers', type=int, default=4, help='Maximum number of worker threads')

    args = parser.parse_args()

    classify_audio_emotion(args.log_file, args.output_path, args.max_workers)

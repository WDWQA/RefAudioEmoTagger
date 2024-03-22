import asyncio
import logging
import os
import time
from itertools import islice
from pathlib import Path

import aiofiles
import argparse
import glob
import pandas as pd
import torch
import torchaudio
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AudioDataset(Dataset):
    def __init__(self, audio_paths: list, target_sample_rate: int = 16000, device: str = 'cuda'):
        self.audio_paths = audio_paths
        self.target_sample_rate = target_sample_rate
        self.device = device

    def __len__(self) -> int:
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> tuple:
        audio_path = self.audio_paths[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = self._resample_waveform(waveform, sample_rate)
        return waveform, audio_path

    def _resample_waveform(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate).to(self.device)
            waveform = resampler(waveform.to(self.device))
        return waveform


def collate_fn(batch: list) -> tuple:
    waveforms, audio_paths = zip(*batch)
    return list(waveforms), list(audio_paths)


class EmotionRecognitionPipeline:
    def __init__(self, model_path: str = "model/emotion2vec", model_revision: str = "v2.0.4", device: str = 'cuda', target_sample_rate: int = 16000):
        self.device = device
        self.target_sample_rate = target_sample_rate
        self.pipeline = pipeline(
            task=Tasks.emotion_recognition,
            model=model_path,
            model_revision=model_revision,
            device=device
        )

    async def infer(self, waveform: torch.Tensor, audio_path: str, output_file: str):
        waveform = waveform.to(self.device)
        recognition_result = self.pipeline(waveform, sample_rate=self.target_sample_rate, granularity="utterance", extract_embedding=False)
        top_emotion, confidence = self._get_top_emotion_with_confidence(recognition_result[0])
        result_str = f"{audio_path}|{top_emotion}|{confidence}|{os.path.basename(os.path.dirname(audio_path))}\n"
        async with aiofiles.open(output_file, 'a', encoding='utf-8') as f:
            await f.write(result_str)
            await f.flush()
                    
    @staticmethod
    def _get_top_emotion_with_confidence(recognition_result: dict) -> tuple:
        scores = recognition_result['scores']
        labels = recognition_result['labels']
        top_index = scores.index(max(scores))
        return labels[top_index].split('/')[0], scores[top_index]
        

async def process_audio_files(folder_path: str, recognizer: EmotionRecognitionPipeline, output_file: str, num_workers: int = 4, batch_size: int = 10):
    folder_path = Path(folder_path)
    if not folder_path.exists():
        logging.error(f"目录不存在：{folder_path}")
        return None

    audio_paths = glob.glob(str(folder_path / '**' / '*.wav'), recursive=True)
    dataset = AudioDataset(audio_paths, target_sample_rate=recognizer.target_sample_rate, device=recognizer.device)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=num_workers, collate_fn=collate_fn)

    start_time = time.time()
    for i in range(0, len(dataset), batch_size):
        batch_tasks = []
        for waveform, audio_path in islice(dataloader, batch_size):
            batch_tasks.append(recognizer.infer(waveform[0], audio_path[0], output_file))
        await asyncio.gather(*batch_tasks)
    logging.info(f"Processed {len(audio_paths)} files in {folder_path}, total time: {time.time() - start_time:.2f} seconds")


def contains_chinese(text: str) -> bool:
    return any('\u4e00' <= char <= '\u9fff' for char in text)
    

def process_text_emotion(df: pd.DataFrame, text_classifier: pipeline) -> pd.DataFrame:
    emotion_mapping = {
        '恐惧': '恐惧',
        '愤怒': '生气', 
        '厌恶': '厌恶',
        '喜好': '开心',
        '悲伤': '难过',
        '高兴': '开心',
        '惊讶': '吃惊'
    }

    def get_chinese_text(text: str) -> str:
        return ''.join(char for char in text if contains_chinese(char))

    texts = df['AudioPath'].apply(lambda x: os.path.splitext(os.path.basename(x))[0]).tolist()

    mapped_emotions = []
    for text in texts:
        if not contains_chinese(text):
            mapped_emotions.append('')
        else:
            chinese_text = get_chinese_text(text)
            result = text_classifier([chinese_text])[0]
            scores = result['scores']
            labels = result['labels'] 
            max_index = scores.index(max(scores))
            original_emotion = labels[max_index]
            mapped_emotion = emotion_mapping.get(original_emotion, original_emotion)
            mapped_emotions.append(mapped_emotion)

    df['TextEmotion'] = mapped_emotions
    return df


async def run_recognition(audio_folder: str, output_file: str, model_revision: str, num_workers: int, disable_text_emotion: bool, batch_size: int = 10):
    emotion_recognizer = EmotionRecognitionPipeline(model_revision=model_revision)
    output_file = Path(output_file)
    
    if output_file.exists():
        output_file.unlink()
    
    await process_audio_files(audio_folder, emotion_recognizer, str(output_file), num_workers, batch_size)
    
    if output_file.exists():
        df = pd.read_csv(output_file, sep='|', header=None, names=['AudioPath', 'AudioEmotion', 'Confidence', 'ParentFolder'])
    else:
        df = pd.DataFrame(columns=['AudioPath', 'AudioEmotion', 'Confidence', 'ParentFolder'])

    if not disable_text_emotion:
        text_classifier = pipeline(Tasks.text_classification, 'model/structbert_emotion', model_revision='v1.0.0')
        df = process_text_emotion(df, text_classifier)

    df.to_csv(output_file, sep='|', index=False, encoding='utf-8')
    logging.info(f"Results saved to {output_file}")

    await asyncio.sleep(0.1)  # 等待异步任务完成


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='识别音频文件中的情感')
    parser.add_argument('--folder_path', type=str, required=True, help='包含音频文件的文件夹路径')
    parser.add_argument('--output_file', type=str, required=True, help='输出文件的路径')
    parser.add_argument('--model_revision', type=str, default="v2.0.4", help='情感识别模型的修订版本')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载的工作进程数')
    parser.add_argument('--batch_size', type=int, default=10, help='批量处理的大小')
    parser.add_argument('--disable_text_emotion', action='store_true', help='是否禁用文本情感分类')
    args = parser.parse_args()

    asyncio.run(run_recognition(args.folder_path, args.output_file, args.model_revision, args.num_workers, args.disable_text_emotion, args.batch_size))
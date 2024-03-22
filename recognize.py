from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import os
import time
import logging
import torchaudio
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import asyncio
import aiofiles

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AudioDataset(Dataset):
    def __init__(self, audio_paths, target_sample_rate=16000, device='cuda'):
        self.audio_paths = audio_paths
        self.target_sample_rate = target_sample_rate
        self.device = device

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = self._resample_waveform(waveform, sample_rate)
        return waveform, audio_path

    def _resample_waveform(self, waveform, sample_rate):
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate).to(self.device)
            waveform = resampler(waveform.to(self.device))
        return waveform


def collate_fn(batch):
    waveforms, audio_paths = zip(*batch)
    return list(waveforms), list(audio_paths)


class EmotionRecognitionPipeline:
    def __init__(self, model_path="model/emotion2vec", model_revision="v2.0.4", device='cuda', target_sample_rate=16000):
        self.device = device
        self.target_sample_rate = target_sample_rate
        self.pipeline = pipeline(
            task=Tasks.emotion_recognition,
            model=model_path,
            model_revision=model_revision,
            device=device
        )

    async def batch_infer(self, dataloader, output_file):
        async with aiofiles.open(output_file, 'w', encoding='utf-8') as f:
            for waveforms, audio_paths in dataloader:
                waveforms = torch.stack(waveforms).to(self.device)  # 将列表转换为张量
                recognition_results = self.pipeline(waveforms, sample_rate=self.target_sample_rate, granularity="utterance", extract_embedding=False)
                for audio_path, result in zip(audio_paths, recognition_results):
                    top_emotion, confidence = self._get_top_emotion_with_confidence(result)
                    result_str = f"{audio_path}|{top_emotion}|{confidence}|{os.path.basename(os.path.dirname(audio_path))}\n"
                    await f.write(result_str)
                    
    @staticmethod
    def _get_top_emotion_with_confidence(recognition_result):
        scores = recognition_result['scores']
        labels = recognition_result['labels']
        top_index = scores.index(max(scores))
        return labels[top_index].split('/')[0], scores[top_index]
        

async def run_recognition(folder_path, output_file, model_revision="v2.0.4", num_workers=4, disable_text_emotion=True):
    if not os.path.exists(folder_path):
        logging.error(f"目录不存在：{folder_path}")
        return None

    audio_paths = glob.glob(os.path.join(folder_path, '**', '*.wav'), recursive=True)
    emotion_recognizer = EmotionRecognitionPipeline(model_revision=model_revision)
    dataset = AudioDataset(audio_paths, target_sample_rate=emotion_recognizer.target_sample_rate, device=emotion_recognizer.device)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=num_workers, collate_fn=collate_fn)

    start_time = time.time()
    await emotion_recognizer.batch_infer(dataloader, output_file)
    logging.info(f"Processed {len(audio_paths)} files in {folder_path}, total time: {time.time() - start_time:.2f} seconds")


def contains_chinese(text):
    return any('\u4e00' <= char <= '\u9fff' for char in text)
    

def process_text_emotion(df, text_classifier):
    emotion_mapping = {
        '恐惧': '恐惧',
        '愤怒': '生气', 
        '厌恶': '厌恶',
        '喜好': '开心',
        '悲伤': '难过',
        '高兴': '开心',
        '惊讶': '吃惊'
    }

    def get_chinese_text(text):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='识别音频文件中的情感')
    parser.add_argument('--folder_path', type=str, required=True, help='包含音频文件的文件夹路径')
    parser.add_argument('--output_file', type=str, required=True, help='输出文件的路径')
    parser.add_argument('--model_revision', type=str, default="v2.0.4", help='情感识别模型的修订版本')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载的工作进程数')
    parser.add_argument('--disable_text_emotion', action='store_true', help='是否禁用文本情感分类')
    args = parser.parse_args()
    
    asyncio.run(main(args))
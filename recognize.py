import os
import time
import logging
import argparse
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
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
        

async def process_audio_files(folder_path, recognizer, output_file, num_workers=4):
    if not os.path.exists(folder_path):
        logging.error(f"目录不存在：{folder_path}")
        return None

    audio_paths = glob.glob(os.path.join(folder_path, '**', '*.wav'), recursive=True)
    dataset = AudioDataset(audio_paths, target_sample_rate=recognizer.target_sample_rate, device=recognizer.device)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=num_workers, collate_fn=collate_fn)

    start_time = time.time()
    await recognizer.batch_infer(dataloader, output_file)
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


async def main(args):
    emotion_recognizer = EmotionRecognitionPipeline(model_revision=args.model_revision)
    output_file = args.output_file
    
    if os.path.exists(output_file):
        os.remove(output_file)
    
    await process_audio_files(args.folder_path, emotion_recognizer, output_file, args.num_workers)
    
    if os.path.exists(output_file):
        df = pd.read_csv(output_file, sep='|', header=None, names=['AudioPath', 'AudioEmotion', 'Confidence', 'ParentFolder'])
    else:
        df = pd.DataFrame(columns=['AudioPath', 'AudioEmotion', 'Confidence', 'ParentFolder'])

    if not args.disable_text_emotion:
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
    parser.add_argument('--disable_text_emotion', action='store_true', help='是否禁用文本情感分类')
    args = parser.parse_args()
    
    asyncio.run(main(args))
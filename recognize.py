import os
import time
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import torchaudio
import glob
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class EmotionRecognitionPipeline:
    def __init__(self, model_path="iic/emotion2vec_base_finetuned", model_revision="v2.0.4", device='cuda:0', target_sample_rate=16000):
        self.device = device
        self.target_sample_rate = target_sample_rate
        self.pipeline = pipeline(
            task=Tasks.emotion_recognition,
            model=model_path,
            model_revision=model_revision,
            device=device
        )

    def batch_infer(self, audio_paths):
        waveforms, sample_rates = zip(*map(torchaudio.load, audio_paths))
        resampled_waveforms = [self._resample_waveform(waveform, sample_rate) for waveform, sample_rate in zip(waveforms, sample_rates)]
        return self.pipeline(resampled_waveforms, sample_rate=self.target_sample_rate, granularity="utterance", extract_embedding=False)

    def _resample_waveform(self, waveform, sample_rate):
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sample_rate)
            waveform = resampler(waveform)
        return waveform.to(self.device)


def get_top_emotion_with_confidence(recognition_results):
    return [(result['labels'][result['scores'].index(max(result['scores']))].split('/')[0], max(result['scores'])) for result in recognition_results]


def process_batch(batch_audio_paths, recognizer):
    top_emotions_with_confidence = get_top_emotion_with_confidence(recognizer.batch_infer(batch_audio_paths))
    return [(audio_path, *top_emotion_confidence) for audio_path, top_emotion_confidence in zip(batch_audio_paths, top_emotions_with_confidence)]


def process_audio_files(folder_path, recognizer, batch_size=10, max_workers=4):
    if not os.path.exists(folder_path):
        logging.error(f"目录不存在：{folder_path}")
        return None

    audio_paths = glob.glob(os.path.join(folder_path, '**', '*.wav'), recursive=True)
    batches = [audio_paths[i:i + batch_size] for i in range(0, len(audio_paths), batch_size)]

    results = []
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch, batch, recognizer) for batch in batches]
        for future in as_completed(futures):
            results.extend(future.result())

    logging.info(f"Processed {len(audio_paths)} files in {folder_path}, total time: {time.time() - start_time:.2f} seconds")
    return results

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

def main(args):
    emotion_recognizer = EmotionRecognitionPipeline(model_revision=args.model_revision)
    audio_emotion_results = process_audio_files(args.folder_path, emotion_recognizer, args.batch_size, args.max_workers)

    if audio_emotion_results is None:
        return

    df = pd.DataFrame(audio_emotion_results, columns=['AudioPath', 'AudioEmotion', 'Confidence'])
    df['ParentFolder'] = df['AudioPath'].apply(lambda x: os.path.basename(os.path.dirname(x)))

    if not args.disable_text_emotion:
        text_classifier = pipeline(Tasks.text_classification, 'damo/nlp_structbert_emotion-classification_chinese-base', model_revision='v1.0.0')
        df = process_text_emotion(df, text_classifier)

    output_file = args.output_file
    df.to_csv(output_file, sep='|', index=False, encoding='utf-8')
    logging.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='识别音频文件中的情感')
    parser.add_argument('--folder_path', type=str, required=True, help='包含音频文件的文件夹路径')
    parser.add_argument('--output_file', type=str, required=True, help='输出文件的路径')
    parser.add_argument('--model_revision', type=str, default="v2.0.4", help='情感识别模型的修订版本')
    parser.add_argument('--batch_size', type=int, default=10, help='推理的批量大小')
    parser.add_argument('--max_workers', type=int, default=4, help='并行处理的最大工作线程数')
    parser.add_argument('--disable_text_emotion', action='store_true', help='是否禁用文本情感分类')
    args = parser.parse_args()
    main(args)

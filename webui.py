import os
import argparse
import logging
import gradio as gr
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir))

from preprocess_audio import filter_audio, rename_wav_with_txt
from recognize import run_recognition as run_recognition_func
from classify import classify_audio_emotion
import shutil

# 配置logging模块来关闭Gradio的输出
logging.getLogger("gradio").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("anyio").setLevel(logging.ERROR)

# 全局参数设置
INPUT_FOLDER = "input"  
PREPROCESS_OUTPUT_FOLDER = "referenceaudio"
CSV_OUTPUT_FOLDER = "csv_opt"
CLASSIFY_OUTPUT_FOLDER = "output"

MIN_DURATION = 3
MAX_DURATION = 10
DISABLE_TEXT_EMOTION = True

NUM_WORKERS = 4
MODEL_REVISION = "v2.0.4"

def create_folders(folders):
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)

def preprocess_and_rename_audio(input_folder, output_folder, min_duration, max_duration, disable_filter):
    input_folder = Path(input_folder)
    src_items = len(list(input_folder.glob('*')))
    copy_parent_folder = src_items > 5

    if disable_filter:
        filter_result = "跳过音频过滤步骤。"
        audio_folder = input_folder
    else:
        filter_audio(input_folder, output_folder, min_duration, max_duration, copy_parent_folder=copy_parent_folder)
        filter_result = f"音频过滤完成,结果保存在 {output_folder} 文件夹中。"
        audio_folder = Path(output_folder)

    renamed_files = rename_wav_with_txt(audio_folder)
    rename_result = f"音频重命名完成,共重命名 {renamed_files} 个文件。"

    return f"{filter_result}\n{rename_result}"

async def recognize_audio_emotions(audio_folder, num_workers, disable_text_emotion, output_file, model_revision=MODEL_REVISION):
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    await run_recognition_func(audio_folder, str(output_file), model_revision, num_workers, disable_text_emotion)
    
    return f"音频情感识别完成,结果保存在 {output_file} 文件中。"

async def classify_audio_emotions(log_file, num_workers, output_folder):
    log_file = Path(log_file)
    if not log_file.exists():
        return f"日志文件 {log_file} 不存在,无法进行音频情感分类。"
    
    classify_audio_emotion(str(log_file), output_folder, num_workers)
    return f"音频情感分类完成,结果保存在 {output_folder} 文件夹中。"

async def run_end_to_end_pipeline(input_folder, min_duration, max_duration, num_workers, disable_text_emotion, disable_filter):
    preprocess_output_folder = PREPROCESS_OUTPUT_FOLDER if not disable_filter else input_folder
    preprocess_result = preprocess_and_rename_audio(input_folder, preprocess_output_folder, min_duration, max_duration, disable_filter)
    
    output_file = Path(CSV_OUTPUT_FOLDER) / "recognition_result.csv"
    recognize_result = await recognize_audio_emotions(preprocess_output_folder, num_workers, disable_text_emotion, str(output_file), model_revision=MODEL_REVISION)
    
    classify_result = await classify_audio_emotions(str(output_file), num_workers, CLASSIFY_OUTPUT_FOLDER)
    return f"{preprocess_result}\n{recognize_result}\n{classify_result}"

def reset_folders():
    folders = [CSV_OUTPUT_FOLDER, CLASSIFY_OUTPUT_FOLDER, PREPROCESS_OUTPUT_FOLDER]
    for folder in folders:
        shutil.rmtree(folder, ignore_errors=True)
        Path(folder).mkdir(exist_ok=True)
    return f"{', '.join(folders)} 文件夹已重置。"

def launch_ui():
    create_folders([INPUT_FOLDER, PREPROCESS_OUTPUT_FOLDER, CSV_OUTPUT_FOLDER, CLASSIFY_OUTPUT_FOLDER])

    with gr.Blocks(theme=gr.themes.Base(
            primary_hue="teal",  
            secondary_hue="blue",  
            neutral_hue="gray", 
            text_size="md",  
            spacing_size="md",  
            radius_size="md",  
            font=["Source Sans Pro", "sans-serif"]
        ), title="音频情感识别与分类应用") as demo:

        gr.Markdown("# RefAudioEmoTagge\n本软件以GPL-3.0协议开源, 作者不对软件具备任何控制力。")

        with gr.Tab("一键推理"):
            with gr.Row():
                with gr.Column():  
                    one_click_input_folder = gr.Textbox(value=INPUT_FOLDER, label="输入文件夹")
                    one_click_min_duration = gr.Number(value=MIN_DURATION, label="最小时长(秒)")
                    one_click_max_duration = gr.Number(value=MAX_DURATION, label="最大时长(秒)")
                    one_click_disable_filter = gr.Checkbox(value=False, label="禁用参考音频筛选")
                with gr.Column():
                    one_click_num_workers = gr.Slider(1, 16, value=NUM_WORKERS, step=1, label="最大工作线程数") 
                    one_click_disable_text_emotion = gr.Checkbox(value=DISABLE_TEXT_EMOTION, label="禁用文本情感分类（效果不如预期，默认禁用）")

            with gr.Row():  
                one_click_button = gr.Button("一键推理", variant="primary")
                one_click_reset_button = gr.Button("一键重置")
            
            one_click_result = gr.Textbox(label="推理结果", lines=5)
            one_click_button.click(run_end_to_end_pipeline, [one_click_input_folder, one_click_min_duration, one_click_max_duration, one_click_num_workers, one_click_disable_text_emotion, one_click_disable_filter], one_click_result)
            one_click_reset_button.click(reset_folders, [], one_click_result)

        with gr.Tab("音频预处理"):
            with gr.Row():    
                preprocess_input_folder = gr.Textbox(value=INPUT_FOLDER, label="输入文件夹")    
                preprocess_output_folder = gr.Textbox(value=PREPROCESS_OUTPUT_FOLDER, label="输出文件夹")
                
            with gr.Row():
                preprocess_min_duration = gr.Number(value=MIN_DURATION, label="最小时长(秒)")  
                preprocess_max_duration = gr.Number(value=MAX_DURATION, label="最大时长(秒)")
                preprocess_disable_filter = gr.Checkbox(value=False, label="禁用参考音频筛选")

            preprocess_button = gr.Button("开始预处理", variant="primary")
            preprocess_result = gr.Textbox(label="预处理结果", lines=3)

            preprocess_button.click(preprocess_and_rename_audio, [preprocess_input_folder, preprocess_output_folder, preprocess_min_duration, preprocess_max_duration, preprocess_disable_filter], preprocess_result)

        with gr.Tab("音频情感识别"):    
            with gr.Row():
                recognize_folder = gr.Textbox(value=PREPROCESS_OUTPUT_FOLDER, label="音频文件夹")
                recognize_output_file = gr.Textbox(value=str(Path(CSV_OUTPUT_FOLDER) / "recognition_result.csv"), label="输出文件路径")

            with gr.Row():
                recognize_num_workers = gr.Slider(1, 16, value=NUM_WORKERS, step=1, label="最大工作线程数")
                recognize_disable_text_emotion = gr.Checkbox(value=DISABLE_TEXT_EMOTION, label="禁用文本情感分类（效果不如预期，默认禁用）")
                
            recognize_button = gr.Button("开始识别", variant="primary")
            recognize_result = gr.Textbox(label="识别结果", lines=3)

            recognize_button.click(recognize_audio_emotions, [recognize_folder, recognize_num_workers, recognize_disable_text_emotion, recognize_output_file], recognize_result)

        with gr.Tab("音频情感分类"):
            with gr.Row():
                classify_log_file = gr.Textbox(value=str(Path(CSV_OUTPUT_FOLDER) / "recognition_result.csv"), label="日志文件")
                classify_output = gr.Textbox(value=CLASSIFY_OUTPUT_FOLDER, label="输出文件夹")

            classify_num_workers = gr.Slider(1, 16, value=NUM_WORKERS, step=1, label="最大工作线程数")

            classify_button = gr.Button("开始分类", variant="primary")  
            classify_result = gr.Textbox(label="分类结果", lines=3)

            classify_button.click(classify_audio_emotions, [classify_log_file, classify_num_workers, classify_output], classify_result)
        
    demo.launch(inbrowser=True, server_name="0.0.0.0", server_port=9975, max_threads=100, share=False)

if __name__ == "__main__":
    launch_ui()
import os
import argparse
import logging
import gradio as gr
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from preprocess_audio import filter_audio, rename_wav_with_txt
from recognize import main as recognize_main
from classify import classify_audio_emotion
import shutil

# 配置logging模块来过滤掉特定的HTTP请求输出
logging.getLogger("gradio").setLevel(logging.WARNING)

# 全局参数设置
INPUT_FOLDER = "input"  
PREPROCESS_OUTPUT_FOLDER = "referenceaudio"
CSV_OUTPUT_FOLDER = "csv_opt"
CLASSIFY_OUTPUT_FOLDER = "output"

MIN_DURATION = 3
MAX_DURATION = 10
DISABLE_TEXT_EMOTION = True

MODEL_REVISION = "v2.0.4"
BATCH_SIZE = 10
MAX_WORKERS = 4

def create_folders(folders):
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

def preprocess_and_rename_audio(input_folder, output_folder, min_duration, max_duration, disable_filter):
    src_items = len(os.listdir(input_folder))
    copy_parent_folder = src_items > 5

    if disable_filter:
        filter_result = "跳过音频过滤步骤。"
        audio_folder = input_folder
    else:
        filter_audio(input_folder, output_folder, min_duration, max_duration, copy_parent_folder=copy_parent_folder)
        filter_result = f"音频过滤完成,结果保存在 {output_folder} 文件夹中。"
        audio_folder = output_folder

    renamed_files = rename_wav_with_txt(audio_folder)
    rename_result = f"音频重命名完成,共重命名 {renamed_files} 个文件。"

    return f"{filter_result}\n{rename_result}"

def recognize_audio_emotions(audio_folder, model_revision, batch_size, max_workers, disable_text_emotion, output_file):
    recognize_args = argparse.Namespace(
        folder_path=audio_folder,
        output_file=output_file,
        model_revision=model_revision,
        batch_size=batch_size,
        max_workers=max_workers,
        disable_text_emotion=disable_text_emotion
    )
    recognize_main(recognize_args)
    return f"音频情感识别完成,结果保存在 {output_file} 文件中。"

def classify_audio_emotions(log_file, max_workers, output_folder):
    classify_audio_emotion(log_file, output_folder, max_workers)
    return f"音频情感分类完成,结果保存在 {output_folder} 文件夹中。"

def run_end_to_end_pipeline(input_folder, min_duration, max_duration, model_revision, batch_size, max_workers, disable_text_emotion, disable_filter):
    preprocess_result = preprocess_and_rename_audio(input_folder, PREPROCESS_OUTPUT_FOLDER, min_duration, max_duration, disable_filter)
    output_file = os.path.join(CSV_OUTPUT_FOLDER, "recognition_result.csv")
    recognize_result = recognize_audio_emotions(PREPROCESS_OUTPUT_FOLDER, model_revision, batch_size, max_workers, disable_text_emotion, output_file)
    classify_result = classify_audio_emotions(output_file, max_workers, CLASSIFY_OUTPUT_FOLDER)
    return f"{preprocess_result}\n{recognize_result}\n{classify_result}"

def reset_folders():
    folders = [CSV_OUTPUT_FOLDER, CLASSIFY_OUTPUT_FOLDER, PREPROCESS_OUTPUT_FOLDER]
    for folder in folders:
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder)
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

        gr.Markdown("# 🎙️ 音频情感识别与分类\n对音频文件进行情感识别和分类。")

        with gr.Tab("一键推理"):
            with gr.Row():
                with gr.Column():  
                    one_click_input_folder = gr.Textbox(value=INPUT_FOLDER, label="输入文件夹")
                    one_click_min_duration = gr.Number(value=MIN_DURATION, label="最小时长(秒)")
                    one_click_max_duration = gr.Number(value=MAX_DURATION, label="最大时长(秒)")
                    one_click_disable_filter = gr.Checkbox(value=False, label="禁用参考音频筛选")
                with gr.Column():
                    one_click_model_revision = gr.Textbox(value=MODEL_REVISION, label="模型版本")
                    one_click_batch_size = gr.Slider(1, 100, value=BATCH_SIZE, step=1, label="批量大小")
                    one_click_max_workers = gr.Slider(1, 16, value=MAX_WORKERS, step=1, label="最大工作线程数") 
                    one_click_disable_text_emotion = gr.Checkbox(value=DISABLE_TEXT_EMOTION, label="禁用文本情感分类")

            with gr.Row():  
                one_click_button = gr.Button("一键推理", variant="primary")
                one_click_reset_button = gr.Button("一键重置")
            
            one_click_result = gr.Textbox(label="推理结果", lines=5)
            one_click_button.click(run_end_to_end_pipeline, [one_click_input_folder, one_click_min_duration, one_click_max_duration, one_click_model_revision, one_click_batch_size, one_click_max_workers, one_click_disable_text_emotion, one_click_disable_filter], one_click_result)
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
                recognize_output_file = gr.Textbox(value=os.path.join(CSV_OUTPUT_FOLDER, "recognition_result.csv"), label="输出文件路径")

            with gr.Row():
                recognize_model_revision = gr.Textbox(value=MODEL_REVISION, label="模型版本") 
                recognize_batch_size = gr.Slider(1, 100, value=BATCH_SIZE, step=1, label="批量大小")
                recognize_max_workers = gr.Slider(1, 16, value=MAX_WORKERS, step=1, label="最大工作线程数")
                recognize_disable_text_emotion = gr.Checkbox(value=DISABLE_TEXT_EMOTION, label="禁用文本情感分类")
                
            recognize_button = gr.Button("开始识别", variant="primary")
            recognize_result = gr.Textbox(label="识别结果", lines=3)

            recognize_button.click(recognize_audio_emotions, [recognize_folder, recognize_model_revision, recognize_batch_size, recognize_max_workers, recognize_disable_text_emotion, recognize_output_file], recognize_result)

        with gr.Tab("音频情感分类"):
            with gr.Row():
                classify_log_file = gr.Textbox(value=os.path.join(CSV_OUTPUT_FOLDER, "recognition_result.csv"), label="日志文件")
                classify_output = gr.Textbox(value=CLASSIFY_OUTPUT_FOLDER, label="输出文件夹")

            classify_max_workers = gr.Slider(1, 16, value=MAX_WORKERS, step=1, label="最大工作线程数")

            classify_button = gr.Button("开始分类", variant="primary")  
            classify_result = gr.Textbox(label="分类结果", lines=3)

            classify_button.click(classify_audio_emotions, [classify_log_file, classify_max_workers, classify_output], classify_result)
        
    demo.launch(inbrowser=True, server_name="0.0.0.0", server_port=9975, max_threads=100)

if __name__ == "__main__":
    launch_ui()

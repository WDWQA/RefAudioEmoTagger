import os
import argparse
import logging
import gradio as gr
from preprocess_audio import filter_audio, rename_wav_with_txt
from recognize import main as recognize_main
from classify import classify_audio_emotion

# 配置logging模块来过滤掉特定的HTTP请求输出
logging.getLogger("gradio").setLevel(logging.WARNING)

def create_folders(folders):
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

def preprocess_and_rename_audio(input_folder, output_folder, min_duration, max_duration, disable_filter):
    if disable_filter:
        filter_result = "跳过音频过滤步骤。"
        audio_folder = input_folder
    else:
        filtered_files = filter_audio(input_folder, output_folder, min_duration, max_duration)
        filter_result = f"音频过滤完成,过滤出 {filtered_files} 个文件,结果保存在 {output_folder} 文件夹中。"
        audio_folder = output_folder

    renamed_files = rename_wav_with_txt(audio_folder)
    rename_result = f"音频重命名完成,共重命名 {renamed_files} 个文件,结果保存在 {audio_folder} 文件夹中。"

    return f"{filter_result}\n{rename_result}", audio_folder

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
    preprocess_result, audio_folder = preprocess_and_rename_audio(input_folder, "referenceaudio", min_duration, max_duration, disable_filter)
    output_file = "csv_opt/recognition_result.csv"
    recognize_result = recognize_audio_emotions(audio_folder, model_revision, batch_size, max_workers, disable_text_emotion, output_file)
    output_folder = "output"
    classify_result = classify_audio_emotions(output_file, max_workers, output_folder)
    return f"{preprocess_result}\n{recognize_result}\n{classify_result}"

def launch_ui():
    create_folders(["input", "referenceaudio", "csv_opt", "output"])

    with gr.Blocks(theme=gr.themes.Default(
            primary_hue="blue",
            secondary_hue="blue",
            neutral_hue="blue", 
            text_size="md",
            spacing_size="md",
            radius_size="md",
            font=["sans-serif", "Arial", "Helvetica", "sans-serif"],
        ), title="音频情感识别与分类应用") as demo:

        gr.Markdown("# 🎙️ 音频情感识别与分类\n这个应用可以帮助你对音频文件进行情感识别和分类。")

        with gr.Tab("一键推理"):
            with gr.Row():
                one_click_input_folder = gr.Textbox(label="输入文件夹", value="input") 
                one_click_min_duration = gr.Number(label="最小时长(秒)", value=3)
                one_click_max_duration = gr.Number(label="最大时长(秒)", value=10)
            with gr.Row():
                one_click_model_revision = gr.Textbox(label="模型版本", value="v2.0.4")
                one_click_batch_size = gr.Slider(1, 100, 10, step=1, label="批量大小")
                one_click_max_workers = gr.Slider(1, 16, 4, step=1, label="最大工作线程数")
            one_click_disable_text_emotion = gr.Checkbox(label="禁用文本情感分类", value=False)
            one_click_disable_filter = gr.Checkbox(label="禁用参考音频筛选", value=False)
            one_click_button = gr.Button("一键推理")
            one_click_result = gr.Textbox(label="推理结果") 
            one_click_button.click(run_end_to_end_pipeline, [one_click_input_folder, one_click_min_duration, one_click_max_duration, one_click_model_revision, one_click_batch_size, one_click_max_workers, one_click_disable_text_emotion, one_click_disable_filter], one_click_result)
        
        with gr.Tab("音频预处理"):
            with gr.Row():
                preprocess_input_folder = gr.Textbox(label="输入文件夹", value="input")
                preprocess_output_folder = gr.Textbox(label="输出文件夹", value="referenceaudio")  
            with gr.Row():
                preprocess_min_duration = gr.Number(label="最小时长(秒)", value=3)
                preprocess_max_duration = gr.Number(label="最大时长(秒)", value=10)
            preprocess_disable_filter = gr.Checkbox(label="禁用参考音频筛选", value=False)
            preprocess_button = gr.Button("开始预处理")
            preprocess_result = gr.Textbox(label="预处理结果")
            preprocess_button.click(preprocess_and_rename_audio, [preprocess_input_folder, preprocess_output_folder, preprocess_min_duration, preprocess_max_duration, preprocess_disable_filter], preprocess_result)

        with gr.Tab("音频情感识别"):
            with gr.Row():  
                recognize_folder = gr.Textbox(label="音频文件夹", value="referenceaudio")
                model_revision = gr.Textbox(label="模型版本", value="v2.0.4")
            with gr.Row():
                batch_size = gr.Slider(1, 100, 10, step=1, label="批量大小") 
                recognize_max_workers = gr.Slider(1, 16, 4, step=1, label="最大工作线程数")
            disable_text_emotion = gr.Checkbox(label="禁用文本情感分类", value=False)
            output_file = gr.Textbox(label="输出文件路径", value="csv_opt/recognition_result.csv")
            recognize_button = gr.Button("开始识别")
            recognize_result = gr.Textbox(label="识别结果")
            recognize_button.click(recognize_audio_emotions, [recognize_folder, model_revision, batch_size, recognize_max_workers, disable_text_emotion, output_file], recognize_result)

        with gr.Tab("音频情感分类"):
            with gr.Row():
                log_file = gr.Textbox(label="日志文件", value="csv_opt/recognition_result.csv")
                classify_max_workers = gr.Slider(1, 16, 4, step=1, label="最大工作线程数")
            classify_output = gr.Textbox(label="输出文件夹", value="output")
            classify_button = gr.Button("开始分类")
            classify_result = gr.Textbox(label="分类结果")
            classify_button.click(classify_audio_emotions, [log_file, classify_max_workers, classify_output], classify_result)

    demo.launch(share=True, inbrowser=True, server_name="0.0.0.0", server_port=9975, max_threads=100)

if __name__ == "__main__":
    launch_ui()

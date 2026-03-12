import gradio as gr
import ollama
import torch
import os
import gc
import warnings
from TTS.api import TTS
from datetime import datetime
from docx import Document
from pydub import AudioSegment
import functools

# 强行禁用全局验证，放行XTTS模型
torch.load = functools.partial(torch.load, weights_only=False)

warnings.filterwarnings("ignore", category=FutureWarning)

# --- 环境配置 ---
TEMP_DIR = "temp"
if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_WAV_PATH = "styles/default.mp3"

# 全局变量：存入完整的记录对象，用于回放索引
conversation_history = []


def clear_vram():
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()


print("正在加载 XTTS v2...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# 文档读取逻辑
def read_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text_content = ""
    if ext == ".txt":
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
        except:
            with open(file_path, 'r', encoding='gbk') as f:
                text_content = f.read()
    elif ext == ".docx":
        doc = Document(file_path)
        text_content = "\n".join([para.text for para in doc.paragraphs])
    elif ext == ".doc":
        return "错误：系统暂不支持旧版 .doc 格式，请将其另存为 .docx 后上传。"
    return text_content.strip()


def get_history_data():
    """将历史记录格式化为表格显示的数据，最新的在上面"""
    # 这里反转显示，但保留原始 ID 对应 conversation_history 的真实索引
    display_list = []
    # 真实索引 i，从后往前遍历
    for i in range(len(conversation_history) - 1, -1, -1):
        m = conversation_history[i]
        display_list.append([i, m["mode"], m["content"], m["time"]])
    return display_list


# --- 核心处理逻辑 ---

def process_all(mode, input_text, upload_file, style, speed):
    clear_vram()
    try:
        final_text = ""
        if mode == "文档文件转语音 (File to Audio)":
            if upload_file is None: return "错误：请上传文件", None, gr.update()
            final_text = read_document(upload_file.name)
            if final_text.startswith("错误"): return final_text, None, gr.update()
        elif mode == "AI 角色对话":
            if not input_text: return "请输入内容", None, gr.update()
            response = ollama.chat(model='llama3.1:8b', messages=[{'role': 'user', 'content': input_text}])
            final_text = response['message']['content']
        else:
            final_text = input_text

        if not final_text: return "无效文本", None, gr.update()

        # TTS 合成
        styles_map = {"默认": "styles/default.mp3", "愤怒": "styles/angry.mp3", "温柔": "styles/gentle.mp3"}
        ref_audio = styles_map.get(style, DEFAULT_WAV_PATH)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_output_path = os.path.join(TEMP_DIR, f"raw_{timestamp}.wav")
        final_output_path = os.path.join(TEMP_DIR, f"tts_{timestamp}.wav")

        tts.tts_to_file(
            text=final_text,
            speaker_wav=ref_audio,
            language="zh-cn",
            file_path=raw_output_path
        )

        # 使用 Pydub 进行物理变速
        audio = AudioSegment.from_file(raw_output_path)

        if speed > 1.0:
            modified_audio = audio.speedup(playback_speed=speed, chunk_size=150, crossfade=25)
        elif speed < 1.0:
            new_sample_rate = int(audio.frame_rate * speed)
            modified_audio = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sample_rate})
            modified_audio = modified_audio.set_frame_rate(audio.frame_rate)
        else:
            modified_audio = audio

        modified_audio.export(final_output_path, format="wav")

        # 移除临时原始文件
        if os.path.exists(raw_output_path): os.remove(raw_output_path)

        # 写入历史
        conversation_history.append({
            "mode": mode,
            "content": final_text[:30] + "...",
            "full_content": final_text,  # 保存全文供回放显示
            "time": datetime.now().strftime("%H:%M:%S"),
            "audio_path": final_output_path
        })

        return final_text, final_output_path, get_history_data()

    except Exception as e:
        return f"执行失败: {str(e)}", None, gr.update()


def play_from_history(evt: gr.SelectData):
    """从表格点击事件中获取 ID 并提取音频"""
    # evt.value 是点击格子里的内容，evt.index 是 [行, 列]
    try:
        row_index = evt.index[0]
        # 因为表格是反转显示的，需要通过第一列的值（真实ID）来找数据
        table_data = get_history_data()
        real_id = table_data[row_index][0]

        record = conversation_history[real_id]
        return record["full_content"], record["audio_path"]
    except:
        return gr.update(), gr.update()


# --- Gradio 界面 ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🎙️ 多格式文档转语音系统 (含历史回放)")

    with gr.Row():
        with gr.Column(scale=1):
            mode_radio = gr.Radio(
                ["直接文字转语音", "文档文件转语音 (File to Audio)", "AI 角色对话"],
                label="功能模式", value="直接文字转语音"
            )

            text_input = gr.Textbox(label="输入文本", lines=5, visible=True)
            file_input = gr.File(label="上传文档 (.txt/.docx)", file_types=[".txt", ".docx"], visible=False)

            style_drop = gr.Dropdown(["默认", "愤怒", "温柔"], label="语音风格", value="默认")
            speed_slider = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="语速控制")
            run_btn = gr.Button("🚀 开始合成", variant="primary")

        with gr.Column(scale=1):
            text_output = gr.Textbox(label="解析出的文本内容", lines=8, interactive=False)
            audio_output = gr.Audio(label="合成语音", type="filepath", autoplay=True)

    gr.Markdown("### 🕒 历史记录 (点击下方任意行即可回放)")
    history_table = gr.Dataframe(
        headers=["ID", "模式", "预览", "时间"],
        interactive=False,
        label="历史记录表"
    )


    # 逻辑绑定
    def toggle_inputs(mode):
        return {
            text_input: gr.update(visible=mode != "文档文件转语音 (File to Audio)"),
            file_input: gr.update(visible=mode == "文档文件转语音 (File to Audio)")
        }


    mode_radio.change(toggle_inputs, inputs=[mode_radio], outputs=[text_input, file_input])

    run_btn.click(
        process_all,
        inputs=[mode_radio, text_input, file_input, style_drop, speed_slider],
        outputs=[text_output, audio_output, history_table]
    )

    # 表格点击联动
    history_table.select(play_from_history, outputs=[text_output, audio_output])


def cleanup():
    if os.path.exists(TEMP_DIR):
        print("\n正在清理临时文件...")
        for f in os.listdir(TEMP_DIR):
            try:
                os.remove(os.path.join(TEMP_DIR, f))
                print("成功清理临时文件")
            except:
                print("清理临时文件失败")
                pass


if __name__ == "__main__":
    cleanup()
    demo.launch()
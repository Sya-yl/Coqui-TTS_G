import gradio as gr
import ollama
import torch
import os
import gc
import warnings
import json
import re
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
DEFAULT_WAV_PATH = "styles/calm.wav"

# 全局变量：存入完整的记录对象，用于回放索引
conversation_history = []


def clear_vram():
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()


class EmotionalDirector:
    @staticmethod
    def analyze(text):
        # 1. 彻底清洗输入文本，防止干扰 JSON 结构
        clean_text = re.sub(r'["\'\n\r]', ' ', text[:100])

        prompt = f"""分析以下文本的情感风格。
                只能从以下列表中选择一个：["紧张", "平静", "难过", "兴奋", "开心", "生气", "惊讶", "好奇"] 

                输出格式必须是严格的 JSON，禁止包含任何额外文字或解释：
                {{"style": "风格名字", "reason": "简短理由"}}

                注意：reason字段内严禁使用双引号。

                文本内容：{clean_text}"""

        try:
            response = ollama.chat(model='llama3.1:8b', messages=[{'role': 'user', 'content': prompt}])
            content = response['message']['content'].strip()

            # 2. 容错逻辑：先尝试正则提取 JSON 部分
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group()

            # 3. 尝试标准解析
            try:
                data = json.loads(content)
                style = data.get("style", "平静")
            except json.JSONDecodeError:
                # 4. 深度容错：如果解析失败，直接用正则匹配 style 后的值
                style_match = re.search(r'"style"\s*:\s*"([^"]+)"', content)
                style = style_match.group(1) if style_match else "平静"

            allowed_styles = ["紧张", "平静", "难过", "兴奋", "开心", "生气", "惊讶", "好奇"]
            if style not in allowed_styles: style = "平静"

            print(f"DEBUG | 识别风格: {style}")
            return {"style": style}

        except Exception as e:
            print(f"AI解析彻底失败， fallback 到平静模式: {e}")
            return {"style": "平静"}


print("正在加载 XTTS v2...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


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
    return text_content.strip()


def get_history_data():
    display_list = []
    for i in range(len(conversation_history) - 1, -1, -1):
        m = conversation_history[i]
        display_list.append([i, m["mode"], m["content"][:50] + "...", m["time"]])
    return display_list


def split_text_smart(text):
    pattern = r'(“[^”]+”)'
    raw_segments = re.split(pattern, text)
    refined_parts = []
    for seg in raw_segments:
        if not seg or not seg.strip(): continue
        if seg.startswith('“') and seg.endswith('”'):
            refined_parts.append((True, seg))
        else:
            # 增加对末尾无标点句子的处理
            sub_segs = re.split(r'([。！？；!?; \n])', seg)
            for i in range(0, len(sub_segs) - 1, 2):
                combined = (sub_segs[i] + sub_segs[i + 1]).strip()
                if combined: refined_parts.append((False, combined))
            if len(sub_segs) % 2 != 0 and sub_segs[-1].strip():
                refined_parts.append((False, sub_segs[-1].strip()))
    return refined_parts


def process_all(mode, input_text, upload_file):
    clear_vram()
    try:
        if mode == "文档文件转语音 (File to Audio)":
            if upload_file is None: return "错误：请上传文件", None, gr.update()
            final_text = read_document(upload_file.name)
        elif mode == "AI 角色对话":
            if not input_text: return "请输入内容", None, gr.update()
            response = ollama.chat(model='llama3.1:8b', messages=[
                {'role': 'system', 'content': "你是一个感性的人类伙伴。注意标点符号的使用，不要有低级错误。"},
                {'role': 'user', 'content': input_text}
            ])
            final_text = response['message']['content']
        else:
            final_text = input_text

        if not final_text: return "无效文本", None, gr.update()

        has_quotes = '“' in final_text and '”' in final_text
        segments = split_text_smart(final_text)
        combined_audio = AudioSegment.empty()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        global_style = "平静"
        if not has_quotes:
            global_style = EmotionalDirector.analyze(final_text[:200])["style"]

        for idx, (is_dialogue, content) in enumerate(segments):
            if has_quotes and is_dialogue:
                analysis_text = content.replace('“', '').replace('”', '')
                active_style = EmotionalDirector.analyze(analysis_text)["style"]
            else:
                active_style = global_style

            styles_map = {
                "紧张": "styles/nervous.wav",
                "平静": "styles/calm.wav",
                "难过": "styles/sad.wav",
                "兴奋": "styles/excited.wav",
                "开心": "styles/happy.wav",
                "生气": "styles/angry.wav",
                "惊讶": "styles/surprised.wav",
                "好奇": "styles/curious.wav"
            }
            ref_audio = styles_map.get(active_style, DEFAULT_WAV_PATH)

            raw_path = os.path.join(TEMP_DIR, f"sent_{idx}_{timestamp}.wav")
            tts.tts_to_file(
                text=content,
                speaker_wav=ref_audio,
                language="zh-cn",
                file_path=raw_path
            )

            if os.path.exists(raw_path):
                segment = AudioSegment.from_file(raw_path).set_frame_rate(24000).set_channels(1).normalize()
                gap_ms = 400 if is_dialogue else 150
                combined_audio += segment + AudioSegment.silent(duration=gap_ms)
                os.remove(raw_path)

            if idx % 5 == 0: gc.collect()

        final_output_path = os.path.join(TEMP_DIR, f"final_{timestamp}.wav")
        combined_audio.export(final_output_path, format="wav")

        conversation_history.append({
            "mode": mode,
            "content": final_text,
            "audio_path": final_output_path,
            "time": datetime.now().strftime("%H:%M:%S")
        })

        return final_text, final_output_path, get_history_data()

    except Exception as e:
        return f"执行失败: {str(e)}", None, gr.update()


def play_from_history(evt: gr.SelectData):
    try:
        row_index = evt.index[0]
        table_data = get_history_data()
        real_id = table_data[row_index][0]
        record = conversation_history[real_id]
        return record["content"], record["audio_path"]
    except Exception as e:
        return gr.update(), gr.update()


# --- Gradio 界面 ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🎙️ 多功能文字转语音系统")

    with gr.Row():
        with gr.Column(scale=1):
            mode_radio = gr.Radio(
                ["文字转语音", "文档转语音", "AI 角色对话"],
                label="功能模式", value="文字转语音"
            )
            text_input = gr.Textbox(label="输入文本", lines=5, visible=True)
            file_input = gr.File(label="上传文档 (.txt/.docx)", file_types=[".txt", ".docx"], visible=False)
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


    def toggle_inputs(mode):
        return {
            text_input: gr.update(visible=mode != "文档文件转语音 (File to Audio)"),
            file_input: gr.update(visible=mode == "文档文件转语音 (File to Audio)")
        }


    mode_radio.change(toggle_inputs, inputs=[mode_radio], outputs=[text_input, file_input])

    run_btn.click(
        process_all,
        inputs=[mode_radio, text_input, file_input],
        outputs=[text_output, audio_output, history_table]
    )

    history_table.select(play_from_history, outputs=[text_output, audio_output])


def cleanup():
    if os.path.exists(TEMP_DIR):
        for f in os.listdir(TEMP_DIR):
            try:
                os.remove(os.path.join(TEMP_DIR, f))
            except:
                pass


if __name__ == "__main__":
    cleanup()
    demo.launch(share=False)
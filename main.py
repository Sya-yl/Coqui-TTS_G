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
        clean_text = re.sub(r'["\'\n\r]', ' ', text[:150])
        print(f"[Analyze Input] {clean_text}")
        prompt = f"""请深入分析以下文本的情感色彩，即使是看似中性的文本也要捕捉细微的情感倾向。
        尽可能参考前文的情感逻辑。

        输出严格的 JSON 格式：
        {{
          "style": "从[紧张, 平静, 难过, 兴奋, 开心, 生气, 惊讶, 好奇, 思考, 怀疑]中选最接近的一个",
          "speed": 0.7到1.4之间的语速浮点数,
          "intensity": 0.4到1.6之间的情感强度,
          "pause": 0到50之间的句尾停顿毫秒数（尽可能变化，默认为0）
        }}

        文本：{clean_text}

        请基于以下线索判断：
        - 关键词："然而"、"但是" → 可能有转折情绪

        直接返回JSON，不要其他内容："""

        try:
            response = ollama.chat(model='llama3.1:8b', messages=[{'role': 'user', 'content': prompt}], options={
                'temperature': 0.3,  # 降低随机性，加快生成
                'num_predict': 120,  # 增加到 120，避免 JSON 被截断
                'top_p': 0.9,  # 限制采样范围
                'num_thread': 8,
            })
            content = response['message']['content'].strip()
            print(f"[Raw Response] {content}")
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                result = {
                    "style": data.get("style", "平静"),
                    "speed": data.get("speed", 1.0),
                    "intensity": data.get("intensity", 1.0),
                    "pause": data.get("pause", 400)
                }
                print(f"[Parsed Result] {result}")
                return result
            else:
                print("[Error] No JSON found in response")
        except Exception as e:
            print(f"[Exception] {e}")
        return {"style": "平静", "speed": 1.0, "intensity": 1.0, "pause": 10}


print("正在加载 XTTS v2...")
model_path = "C:/Users/Sya/AppData/Local/tts/tts_models--multilingual--multi-dataset--xtts_v2"
config_path = os.path.join(model_path, "config.json")
tts = TTS(model_path=model_path, config_path=config_path).to(device)


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
    """
    智能分句模块
    1. 完全删除换行、空格等无意义符号
    2. 区分对话和叙述：引号内文本≥10字才算对话，否则去引号作叙述处理
    3. 按强分隔符切分
    4. 控制单句长度在 10-80 字，适配 Llama 3.1 8B
    """
    if not text or not text.strip():
        return []

    # 第一步：清洗文本 - 完全删除无意义符号
    text = text.replace('\n', '').replace('\r', '')
    text = text.replace(' ', '').replace('\t', '').replace('\v', '').replace('\f', '')
    text = re.sub(r'\s+', '', text)

    if not text:
        return []

    # 第二步：处理引号内容 - 短引号去引号，长引号保留
    # 先找出所有引号内容
    pattern = r'(“[^”]+”)'
    parts = re.split(pattern, text)

    temp_parts = []
    for part in parts:
        if not part:
            continue
        # 如果是引号内容且长度<5，去掉引号当作普通文本
        if part.startswith('“') and part.endswith('”') and len(part) < 5:
            temp_parts.append(part[1:-1])
        else:
            temp_parts.append(part)

    # 重新组合文本
    text = ''.join(temp_parts)

    # 第三步：重新分离对话和叙述（此时只剩≥10字的引号内容）
    pattern = r'(“[^”]+”)'
    parts = re.split(pattern, text)

    result = []

    for part in parts:
        if not part:
            continue

        # 判断是否为对话（此时引号内容必然≥10字）
        is_dialogue = part.startswith('“') and part.endswith('”')

        # 第四步：按强分隔符切分
        if is_dialogue:
            inner_text = part[1:-1]
            sentences = re.split(r'(?<=[。！？!?])(?![。！？!?])', inner_text)
            for s in sentences:
                if s:
                    result.append((True, '“' + s + '”'))
        else:
            sentences = re.split(r'(?<=[。！？!?])(?![。！？!?])', part)
            for s in sentences:
                if s:
                    result.append((False, s))

    # 第五步：处理超长句子（>80字按逗号切分）
    processed = []
    for is_dialogue, sentence in result:
        if len(sentence) <= 80:
            processed.append((is_dialogue, sentence))
        else:
            if is_dialogue:
                inner = sentence[1:-1]
                sub_parts = re.split(r'(?<=[，,；;])(?![，,；;])', inner)
                temp = ""
                for part in sub_parts:
                    if not part:
                        continue
                    if len(temp) + len(part) <= 60:
                        temp += part
                    else:
                        if temp:
                            processed.append((True, '“' + temp + '”'))
                        temp = part
                if temp:
                    processed.append((True, '“' + temp + '”'))
            else:
                sub_parts = re.split(r'(?<=[，,；;])(?![，,；;])', sentence)
                temp = ""
                for part in sub_parts:
                    if not part:
                        continue
                    if len(temp) + len(part) <= 60:
                        temp += part
                    else:
                        if temp:
                            processed.append((False, temp))
                        temp = part
                if temp:
                    processed.append((False, temp))

    # 第六步：合并过短片段（<10字）
    final = []
    for is_dialogue, sentence in processed:
        if len(sentence) < 10 and final and final[-1][0] == is_dialogue:
            prev_is_dialogue, prev_sentence = final[-1]
            final[-1] = (prev_is_dialogue, prev_sentence + sentence)
        else:
            final.append((is_dialogue, sentence))

    # 输出统计
    dialogue_count = sum(1 for d, _ in final if d)
    print(f"📊 分句: 原文{len(text)}字 → {len(final)}句 (对话{dialogue_count}句)")

    return final


def process_all(mode, input_text, upload_file):
    clear_vram()
    try:
        if mode == "文档转语音":
            if upload_file is None: return "错误：请上传文件", None, gr.update(),""
            final_text = read_document(upload_file.name)
        elif mode == "AI 角色对话":
            if not input_text: return "请输入内容", None, gr.update(),""
            response = ollama.chat(model='llama3.1:8b', messages=[
                {'role': 'system', 'content': "你是一个感性的人类伙伴。注意标点符号的使用，不要有低级错误。"},
                {'role': 'user', 'content': input_text}
            ])
            final_text = response['message']['content']
        else:
            final_text = input_text

        if not final_text: return "无效文本", None, gr.update(),""

        segments = split_text_smart(final_text)
        combined_audio = AudioSegment.empty()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        feature_logs = []
        tts_instance = tts

        for idx, (is_dialogue, content) in enumerate(segments):
            # ===== 每一句都触发情感分析 =====
            if is_dialogue:
                analysis_text = content.replace('“', '').replace('”', '')
            else:
                analysis_text = content

            # 无条件调用情感分析
            features = EmotionalDirector.analyze(analysis_text)

            active_style = features["style"]
            target_speed = features["speed"]
            suggested_pause = features["pause"]

            log_line = f"分句 {idx + 1} | 风格: {active_style} | 语速: {target_speed}x | 停顿: {suggested_pause}ms"
            feature_logs.append(log_line)

            styles_map = {
                "紧张": "styles/nervous.wav", "平静": "styles/calm.wav", "难过": "styles/sad.wav",
                "兴奋": "styles/excited.wav", "开心": "styles/happy.wav", "生气": "styles/angry.wav",
                "惊讶": "styles/surprised.wav", "好奇": "styles/curious.wav"
            }
            ref_audio = styles_map.get(active_style, DEFAULT_WAV_PATH)

            raw_path = os.path.join(TEMP_DIR, f"sent_{idx}_{timestamp}.wav")

            tts_instance.tts_to_file(
                text=content,
                speaker_wav=ref_audio,
                language="zh-cn",
                speed=target_speed,
                file_path=raw_path
            )

            if os.path.exists(raw_path):
                segment = AudioSegment.from_file(raw_path).set_frame_rate(24000).set_channels(1).normalize()
                gap_ms = suggested_pause
                combined_audio += segment + AudioSegment.silent(duration=gap_ms)
                os.remove(raw_path)

            clear_vram()

        final_output_path = os.path.join(TEMP_DIR, f"final_{timestamp}.wav")
        combined_audio.export(final_output_path, format="wav")

        conversation_history.append({
            "mode": mode,
            "content": final_text,
            "audio_path": final_output_path,
            "time": datetime.now().strftime("%H:%M:%S")
        })

        return final_text, final_output_path, get_history_data(), "\n".join(feature_logs)

    except Exception as e:
        return f"执行失败: {str(e)}", None, gr.update(),""


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
            param_display = gr.Textbox(label="🧠 Llama 语义控制特征流", lines=6, interactive=False,
                                       placeholder="等待参数注入...")

    gr.Markdown("### 🕒 历史记录 (点击下方任意行即可回放)")
    history_table = gr.Dataframe(
        headers=["ID", "模式", "预览", "时间"],
        interactive=False,
        label="历史记录表"
    )


    def toggle_inputs(mode):
        return {
            text_input: gr.update(visible=mode != "文档转语音"),
            file_input: gr.update(visible=mode == "文档转语音")
        }


    mode_radio.change(toggle_inputs, inputs=[mode_radio], outputs=[text_input, file_input])

    run_btn.click(
        process_all,
        inputs=[mode_radio, text_input, file_input],
        outputs=[text_output, audio_output, history_table, param_display]
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

import os
import glob
import re
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm
import soundfile as sf
import torch

# SenseVoice 所需导入
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# 初始化模型
print("正在加载SenseVoice模型...")
sensevoice_model = AutoModel(
    model="iic/SenseVoiceSmall",
    trust_remote_code=True,
    device='cuda',  # 根据你的硬件调整，'cpu' 或 'cuda'
    disable_update=True
)

# 初始化FSMN-VAD模型（全局初始化，避免重复加载）
print("正在加载FSMN-VAD模型...")
vad_model = AutoModel(
    model="fsmn-vad",  # 使用FSMN-VAD模型
    trust_remote_code=True,
    device='cuda' if torch.cuda.is_available() else 'cpu',  # 根据硬件调整
    disable_update=True
)

# 需要检测的关键词列表
TARGET_WORDS = ["牛", "强", "厉害", "卧槽", "准", "锁", "傻逼"]
# SenseVoice 中可能输出的事件标签
EVENT_TAGS = {
    "laughter": ["<|Laughter|>", "[笑声]", "laughter"]
}
TARGET_EVENTS = ["laughter"]


def extract_audio_from_video(video_path, audio_output_path):
    """使用pydub从视频提取音频并保存为wav格式"""
    try:
        # 根据视频格式调整导入方法
        if video_path.endswith('.mp4'):
            video = AudioSegment.from_file(video_path, format="mp4")
        else:
            # 可添加其他格式支持
            video = AudioSegment.from_file(video_path)

        # 设置为单声道、16kHz采样率（Whisper推荐格式）
        audio = video.set_channels(1).set_frame_rate(16000)
        audio.export(audio_output_path, format="wav")
        return True
    except Exception as e:
        print(f"提取音频失败: {e}")
        return False


def analyze_audio_with_sensevoice(audio_path):
    """使用SenseVoice分析音频并返回结果"""
    try:
        result = sensevoice_model.generate(
            input=audio_path,
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=60,
        )
        return result[0]  # 返回主要结果
    except Exception as e:
        print(f"SenseVoice分析失败: {e}")
        return None


def parse_sensevoice_output(sensevoice_result):
    """
    解析SenseVoice的输出，提取事件、文本信息
    """
    events = []
    full_text = ""

    if sensevoice_result and 'text' in sensevoice_result:
        full_text = sensevoice_result['text']
        print(f"SenseVoice 输出文本: {full_text}")

        # 解析事件标签
        for event_type, tags in EVENT_TAGS.items():
            for tag in tags:
                pattern = re.escape(tag)
                matches = re.finditer(pattern, full_text)
                for match in matches:
                    events.append({
                        'type': 'audio_event',
                        'subtype': event_type,
                        'content': tag,
                        'start_char': match.start(),
                        'end_char': match.end(),
                    })

    return events, full_text


def get_audio_segments_with_vad(audio_path):
    """
    使用FSMN-VAD模型获取音频中人声片段的开始和结束时间（毫秒）
    """
    try:
        # 使用VAD模型处理音频文件
        vad_res = vad_model.generate(input=audio_path)
        segments = []

        if vad_res and len(vad_res) > 0:
            for segment in vad_res[0]['value']:
                segments.append({
                    'start': segment[0],  # 假设第一个元素是开始时间
                    'end': segment[1]  # 假设第二个元素是结束时间
                })
        return segments
    except Exception as e:
        print(f"VAD处理失败: {e}")
        return []


def crop_audio(audio_path, start_time, end_time, output_path):
    """
    裁剪音频片段
    """
    try:
        # 读取音频
        audio_data, sample_rate = sf.read(audio_path)

        # 计算样本数
        start_sample = int(start_time * sample_rate / 1000)
        end_sample = int(end_time * sample_rate / 1000)

        # 裁剪音频
        cropped_audio = audio_data[start_sample:end_sample]

        # 保存裁剪后的音频
        sf.write(output_path, cropped_audio, sample_rate)
        return True
    except Exception as e:
        print(f"音频裁剪失败: {e}")
        return False


def find_highlight_timestamps(vad_segments, audio_duration):
    """
    根据VAD分段和SenseVoice识别结果找出高光时刻的时间戳
    """
    highlights = []

    # 处理每个VAD片段
    # 使用tqdm创建进度条，包装vad_segments这个可迭代对象
    # total参数设置总长度（总片段数），desc参数设置进度条前的描述文本
    for segment in tqdm(vad_segments, total=len(vad_segments), desc="处理音频片段"):
        start_time = segment['start'] / 1000.0  # 转换为秒
        end_time = segment['end'] / 1000.0  # 转换为秒

        # 创建临时音频文件
        temp_audio_path = f"temp_segment_{start_time}_{end_time}.wav"
        if not crop_audio("temp_audio.wav", segment['start'], segment['end'], temp_audio_path):
            continue

        # 使用SenseVoice分析裁剪后的音频
        sensevoice_result = analyze_audio_with_sensevoice(temp_audio_path)
        if sensevoice_result is None:
            continue

        # 解析SenseVoice输出
        sensevoice_events, full_text = parse_sensevoice_output(sensevoice_result)

        # 检查目标事件
        for event in sensevoice_events:
            if event['subtype'] in TARGET_EVENTS:
                highlights.append({
                    'time': start_time,
                    'type': 'audio_event',
                    'subtype': event['subtype'],
                    'content': full_text,
                    'confidence': 0.8
                })

        # 检查关键词
        for keyword in TARGET_WORDS:
            if keyword in full_text:
                # 找到关键词所在的句子
                sentence_start = full_text.rfind('。', 0, full_text.find(keyword)) + 1
                sentence_end = full_text.find('。', full_text.find(keyword))
                if sentence_end == -1:
                    sentence_end = len(full_text)

                context_sentence = full_text[sentence_start:sentence_end].strip()

                highlights.append({
                    'time': start_time,
                    'type': 'keyword',
                    'content': context_sentence,
                    'keyword': keyword,
                    'confidence': 0.8
                })

        # 清理临时文件
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    # 按时间排序
    highlights.sort(key=lambda x: x['time'])
    return highlights


def get_audio_duration(audio_path):
    """获取音频文件的持续时间（秒）"""
    try:
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0  # 转换为秒
    except Exception as e:
        print(f"获取音频时长失败: {e}")
        return 0


def process_single_video(video_path, output_txt_path, pbar=None):
    print(f"开始处理视频: {video_path}")
    temp_audio_path = "temp_audio.wav"

    # 提取音频
    if not extract_audio_from_video(video_path, temp_audio_path):
        if pbar: pbar.update(1)
        return False

    audio_duration = get_audio_duration(temp_audio_path)

    # 使用VAD获取人声时间段
    vad_segments = get_audio_segments_with_vad(temp_audio_path)
    if not vad_segments:
        print("警告：未检测到语音活动")
        if pbar: pbar.update(1)
        return False

    # 高光时刻检测（基于VAD分段）
    highlights = find_highlight_timestamps(vad_segments, audio_duration)

    # 写入结果文件
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"视频文件: {video_path}\n")
        f.write(f"音频时长: {audio_duration:.2f}秒\n")
        f.write(f"检测到的高光时刻数量: {len(highlights)}\n\n")
        for highlight in highlights:
            hours = int(highlight['time'] / 3600)
            minutes = int(highlight['time'] / 60 % 60)
            seconds = int(highlight['time'] % 60)
            f.write(
                f"时间: {hours:02d}:{minutes:02d}:{seconds:02d}:00 - 类型: {highlight['type']} - 内容: {highlight['content']} - 置信度: {highlight['confidence']:.4f}\n")

    # 清理临时文件
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)

    print(f"完成处理: {video_path}, 找到 {len(highlights)} 个高光时刻")
    if pbar:
        pbar.update(1)
        pbar.set_postfix_str(f"最新处理: {os.path.basename(video_path)}")
    return True


def main(video_directory):
    """主函数：遍历目录并处理所有MP4文件"""
    # 获取所有MP4文件
    video_files = glob.glob(os.path.join(video_directory, "**/*.mp4"), recursive=True)
    # 按修改时间从新到旧排序
    video_files.sort(key=os.path.getmtime, reverse=True)

    print(f"找到 {len(video_files)} 个MP4文件")

    # 创建输出目录
    output_dir = os.path.join(video_directory, "highlights")
    os.makedirs(output_dir, exist_ok=True)

    # 使用tqdm创建主进度条
    with tqdm(total=len(video_files), desc="处理视频", unit="个") as pbar:
        # 处理每个视频
        for video_path in video_files:
            # 生成输出txt文件名（与原视频同名）
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_txt_path = os.path.join(output_dir, f"{video_name}_highlights.txt")

            # 检查输出文件是否已存在
            if os.path.exists(output_txt_path):
                print(f"跳过已处理视频: {video_path}")
                pbar.update(1)
                continue

            # 处理视频
            success = process_single_video(video_path, output_txt_path, pbar)

            if not success:
                print(f"处理失败: {video_path}")
            else:
                print(f"处理完成: {video_path} -> {output_txt_path}")

    print("所有视频处理完成！")


if __name__ == "__main__":
    VIDEO_DIRECTORY = "Z:\\test\\"  # 替换为视频目录

    main(VIDEO_DIRECTORY)

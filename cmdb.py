import time, itertools, pprint, subprocess, sys
from collections import Counter
import unicodedata
from faster_whisper import WhisperModel
import demucs.separate
import streamlit as st
from pathlib import Path
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import yt_dlp
import ffmpeg
import re
whisper_models = {None: 'tiny'}  # 'he': 'ivrit-ai/faster-whisper-v2-d3-e3',
loaded_model_name = None
loaded_model = None

def youtube_to_wav(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'youtube_sound.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
    }
    yt_dlp.YoutubeDL(ydl_opts).download([url])
    stream = ffmpeg.input('output.m4a')
    stream = ffmpeg.output(stream, 'output.wav')
    return 'youtube_sound.wav'

def load_whisper(lang):
    global loaded_model_name, loaded_model
    model_name = whisper_models.get(lang, whisper_models[None])
    if model_name == loaded_model_name:
        return loaded_model, False

    loaded_model = WhisperModel(model_name, device="auto", compute_type="int8")
    print(f'loaded {model_name}')
    loaded_model_name = model_name
    return loaded_model, True

# loaded_separator = None

# def load_separator():
#     global loaded_separator
#     if loaded_separator is not None:
#         return load_separator
#     loaded_separator = demucs.api.Separator(model='htdemucs')
#     return loaded_separator

def segment(result):
    result = [segment.words for segment in result]
    word_durations = [word.end - word.start for segment in result for word in segment]
    avg_word_duration = sum(word_durations) / len(word_durations)
    word_speed = 1 / avg_word_duration
    print(word_speed)
    max_characters_per_line = 30
    max_lines = max(2, int(word_speed - 1))
    if word_speed >= 3.5:
        result = [[word for segment in result for word in segment]]

    all_new_segments = []
    for segment in result:
        current_segment = []
        current_line = []
        current_line_len = 0
        for word in itertools.chain(segment, [None]):
            if word is None or (current_line_len + len(
                    word.word.strip()) > max_characters_per_line and current_line_len > 0):  # really long words
                # flush line
                if len(current_segment) >= max_lines:
                    # flush segment
                    all_new_segments.append(current_segment)
                    current_segment = []
                if current_line:
                    current_segment.append(current_line)
                current_line = []
                current_line_len = 0
                if word is None:
                    # flush line again
                    if current_segment:
                        all_new_segments.append(current_segment)
                    current_segment = []
            if word is not None:
                current_line.append(word)
                current_line_len += len(word.word.strip())
    return all_new_segments

def dominant_strong_direction(s):
    count = Counter([unicodedata.bidirectional(c) for c in list(s)])
    rtl_count = count['R'] + count['AL'] + count['RLE'] + count["RLI"]
    ltr_count = count['L'] + count['LRE'] + count["LRI"]
    return "rtl" if rtl_count > ltr_count else "ltr"

def ass_time(seconds):
    return f'{int(seconds // 3600)}:{int(seconds % 3600 // 60):02d}:{int(seconds % 60):02d}.{int(seconds * 100 % 100):02d}'

def output_line(words, selected_word):
    text_wrap = r'{{\c&H{color}&}}{text}{{\r}}'
    marked_color = '0000FF'
    unmarked_colors = ['FFFFFF', 'FEFEFE']
    direction = dominant_strong_direction(''.join(w.word for w in words))
    wrapped = [text_wrap.format(text=w.word, color=marked_color if w == selected_word else unmarked_colors[i % 2]) for
               i, w in enumerate(words)]
    return ''.join(reversed(wrapped) if direction == 'rtl' else wrapped)

def make_ass(segments, prepare_time_seconds=5, offset=0.1):
    header = '''[Script Info]
PlayResX: 800
PlayResY: 800
WrapStyle: 1

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Alignment, Encoding
Style: S1, Arial,80,&HFFFFFF,&HFFFFFF,&H202020,&H000000,5,0

[Events]
Format: Layer, Start, End, Style, Text
'''
    # Format: Layer, Start, End, Style, Text
    # Dialogue: 0,0:00:00.00,0:00:10.00,S1,{\pos(260,320)}The quick brown fox jumps over a lazy dog.\NSphinx of black quartz, judge my vow.
    newline = r'\N {\fs20} \h {\r} \N'
    mintime = 0.0
    ass_lines = []
    last_segment_end = None
    for segment in segments:
        preferred_start = segment[0][0].start - prepare_time_seconds
        prev_time = max(preferred_start, last_segment_end if last_segment_end is not None else 0)
        for line in segment:
            for word in line:
                direction = dominant_strong_direction('\n'.join(''.join(w.word for w in line) for line in segment))
                if word.start - prev_time > mintime:
                    # output uncolored line
                    ass_lines.append(
                        f'Dialogue: 0,{ass_time(prev_time + offset)},{ass_time(word.start + offset)},S1,{{\\pos(400,400)}}{newline.join(output_line(line, None) for line in segment)}')
                ass_lines.append(
                    f'Dialogue: 0,{ass_time(word.start + offset)},{ass_time(word.end + offset)},S1,{{\\pos(400,400)}}{newline.join(output_line(line, word) for line in segment)}')
                prev_time = word.end
        last_segment_end = segment[-1][-1].end
    open('sub.ass', 'w', encoding='utf8').write(header + '\n'.join(ass_lines))
    return 'sub.ass'

def instrumental(input, output):
    base_name = os.path.basename(input)
    file_name, _ = os.path.splitext(base_name)
    demucs.separate.main(["--two-stems=vocals", "-o", ".", input])
    print("FINISHED DEMUCS")
    inst = f'./htdemucs/{file_name}/no_vocals.wav'
    # demucs.api.save_audio(inst, output, samplerate=separator.samplerate)
    return inst

def work(audiopath, outputpath):
    model, _ = load_whisper(None)
    # separator = load_separator()
    t1 = time.time()
    result, info = model.transcribe(audiopath, word_timestamps=True)
    model, reloaded = load_whisper(info.language)
    if reloaded:
        result, info = model.transcribe(audiopath, word_timestamps=True)
    segments = segment(result)
    asspath = make_ass(segments)
    soundpath = instrumental(audiopath, f'{audiopath}_inst.mp3')
    try:
        process = subprocess.run(
            ['ffmpeg', '-y', '-f', 'lavfi', '-i', 'color=c=black:s=1280x720', '-i', soundpath, '-shortest', '-fflags',
             '+shortest', '-vf', f'subtitles={asspath}', '-vcodec', 'h264', outputpath], capture_output=True)
        if process.returncode != 0:
            raise RuntimeError('ffmpeg failed')
    finally:
        print("AFTER FFMPEG")
        pass
        os.remove(asspath)
        os.remove(soundpath)
    t2 = time.time()
    print(f't: {t2 - t1}')

def main():
    st.title("Karaoke Generator")
    input_type = st.radio("Input Type", ["Local File", "YouTube Link"])
    if input_type == "Local File":
        uploaded_file = st.file_uploader("Choose an audio file (mp3 or wav)", type=["mp3", "wav"])
        if uploaded_file is not None:
            file = Path(f'./uploads/{uploaded_file.name}')
            first_arg = f'./uploads/{uploaded_file.name}'
            with open(file, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File uploaded: {file}")
    else:
        youtube_url = st.text_input("YouTube Link")
        if youtube_url:
            first_arg = youtube_to_wav(youtube_url)
            print(youtube_url)
            st.success(f"Downloaded YouTube audio: {first_arg}")
    output_path = st.text_input("Output MP4 File Path (Optional)", 'RESULT.mp4')
    if st.button("Generate Karaoke Video"):
        with st.spinner('Processing...'):
            work(first_arg, output_path)
        st.success("Karaoke video generated!")
        st.video(output_path)

if __name__ == '__main__':
    main()

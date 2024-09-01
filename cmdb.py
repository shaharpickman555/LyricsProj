import time, itertools, pprint, subprocess, sys, math, os, collections, re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import torch
from faster_whisper import WhisperModel
import demucs.api
import yt_dlp
import streamlit as st

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


whisper_models = {None: 'tiny'}  # 'he': 'ivrit-ai/faster-whisper-v2-d3-e3',
loaded_model_name = None
loaded_model = None

def youtube_to_mp3(url):
    ydl_opts = {
        'extract_audio': True,
        'format': 'bestaudio',
        'outtmpl': '%(id)s.mp3',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        outfile = f'{info['id']}.mp3'
    return outfile, info['title']

def load_whisper(lang):
    global loaded_model_name, loaded_model
    model_name = whisper_models.get(lang, whisper_models[None])
    if model_name == loaded_model_name:
        return loaded_model, False

    loaded_model = WhisperModel(model_name, device="auto", compute_type="int8")
    print(f'loaded {model_name}')
    loaded_model_name = model_name
    return loaded_model, True

loaded_separator = None
def load_separator():
    global loaded_separator
    if loaded_separator is not None:
        return load_separator
    loaded_separator = demucs.api.Separator(model='htdemucs')
    return loaded_separator

def segment(result):
    result = [segment.words for segment in result]
    
    word_durations = [word.end - word.start for segment in result for word in segment]
    words_per_spoken_second = len(word_durations) / sum(word_durations)
    
    max_characters_per_line = 30
    max_lines = max(2, int(words_per_spoken_second - 1))

    #merge all?
    if words_per_spoken_second >= 3.5:
        result = [[word for segment in result for word in segment]]

    all_new_segments = []
    for segment in result:
        current_segment = []
        current_line = []
        current_line_len = 0
        for word in itertools.chain(segment, [None]):
            if word is None or (current_line_len + len(word.word.strip()) > max_characters_per_line and current_line_len > 0): #really long words
                #flush line
                
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
    count = collections.Counter([unicodedata.bidirectional(c) for c in list(s)])
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

    wrapped = [text_wrap.format(text=w.word, color=marked_color if w == selected_word else unmarked_colors[i % 2]) for i,w in enumerate(words)]
    
    return ''.join(reversed(wrapped) if direction == 'rtl' else wrapped)

def ass_circle(start_layer, x, y, start_time, end_time, fadein_time):
    mid_time = (end_time + start_time) / 2
    return [rf'Dialogue: {start_layer},{ass_time(start_time - fadein_time)},{ass_time(mid_time)},S1,{{\pos({x},{y})}}{{\fad({int(fadein_time * 1000)},0)}}{{\p1}}m 0 0 b 20 0 20 50 0 50{{\p0}}',
            rf'Dialogue: {start_layer + 2},{ass_time(start_time - fadein_time)},{ass_time(mid_time)},S1,{{\pos({x},{y})}}{{\fad({int(fadein_time * 1000)},0)}}{{\p1}}m 0 50 b -20 50 -20 0 0 0{{\p0}}{{\r}}',
            rf'Dialogue: {start_layer + 1},{ass_time(start_time)},{ass_time(end_time)},B1,{{\pos({x + 2.5},{y})}}{{\org({x - 10},{y})}}{{\t(\frz-360)}}{{\p1}}m 0 55 b -25 55 -25 0 0 0{{\p0}}{{\r}}',
            rf'Dialogue: {start_layer},{ass_time(mid_time)},{ass_time(end_time)},S1,{{\pos({x},{y})}}{{\p1}}m 0 50 b -20 50 -20 0 0 0{{\p0}}{{\r}}']
    
    
def make_ass_swap(segments, offset, prepare_time_seconds=5):
    header = '''[Script Info]
PlayResX: 800
PlayResY: 800
WrapStyle: 1

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Alignment, Encoding
Style: S1, Arial,80,&HFFFFFF,&HFFFFFF,&H202020,&H000000,5,0
Style: B1, Arial,80,&H000000,&H000000,&H202020,&H000000,5,0

[Events]
Format: Layer, Start, End, Style, Text
'''
    
    #flatten and append 1 word lines
    new_segment = []
    for segment in segments:
        for line in segment:
            if len(line) == 1 and new_segment:
                new_segment[-1].extend(line)
            else:
                new_segment.append(line)
    
    segments = [new_segment]
    
    word_durations = [word.end - word.start for line in segments[0] for word in line]
    words_per_spoken_second = len(word_durations) / sum(word_durations)

    y_off = 120
    
    #always even
    num_lines = int(math.ceil(max(min(int(words_per_spoken_second - 1), 800 / y_off), 2) / 2) * 2)
    
    ystart = 400 - (y_off * num_lines / 2)
    
    switch_factor = 0.5
    
    num_batches = 2
    batch_size = int(math.ceil(num_lines / num_batches)) #last batch might be smaller
    
    line_to_first_line_in_batch = lambda l: ((l // batch_size) * batch_size)
    first_line_to_last_line = lambda l: (((l // num_lines) * num_lines) + min((l % num_lines) + batch_size - 1, num_lines - 1))
    
    ass_lines = []
    last_segment_end = None
    for segment in segments:
        for i, line in enumerate(segment):
            appear_batch_first_line = line_to_first_line_in_batch(i - num_lines + batch_size)
            appear_batch_last_line = first_line_to_last_line(appear_batch_first_line)
            
            disappear_batch_first_line = line_to_first_line_in_batch(appear_batch_first_line + num_lines)
            disappear_batch_last_line = first_line_to_last_line(disappear_batch_first_line)
            
            if appear_batch_first_line < 0:
                appear_time = max(segment[0][0].start - 1, 0)
            else:
                appear_time = segment[appear_batch_first_line][0].start * (1 - switch_factor) + segment[appear_batch_last_line][-1].end * switch_factor
                
            if disappear_batch_last_line > len(segment) - 1:
                disappear_time = segment[-1][-1].end + 1
            else:
                disappear_time = segment[disappear_batch_first_line][0].start * (1 - switch_factor) + segment[disappear_batch_last_line][-1].end * switch_factor

            actual_appear_time = max(appear_time, line[0].start - prepare_time_seconds)
            actual_disappear_time = min(disappear_time, line[-1].end + prepare_time_seconds)
            
            line_y_off = (i % num_lines) * y_off
            
            #output uncolored line as background
            ass_lines.append(f'Dialogue: 0,{ass_time(actual_appear_time + offset)},{ass_time(actual_disappear_time + offset)},S1,{{\\pos(400,{ystart + line_y_off})}}{{\\fad(1000,1000)}}{output_line(line, None)}')
            
            for word in line:
                #output colored lines on top
                ass_lines.append(f'Dialogue: 1,{ass_time(word.start + offset)},{ass_time(word.end + offset)},S1,{{\\pos(400,{ystart + line_y_off})}}{output_line(line, word)}')
            
            #counter
            if len(line) > 1:
                time_since_last_word = (line[0].start - segment[i - 1][-1].end) if i > 0 else (line[0].start - (last_segment_end or 0))
                if time_since_last_word >= prepare_time_seconds:
                    clock_duration = 4
                    
                    ass_lines.extend(ass_circle(2, 200, 100, line[0].start - clock_duration + offset, line[0].start + offset, 0.5))

        last_segment_end = segment[-1][-1].end
    
    return header + '\n'.join(ass_lines)

def instrumental(separator, input, output, start_silence, end_silence):
    origin, separated = separator.separate_audio_file(input)

    #inst = sum(data for name, data in separated.items() if name != 'vocals')
    inst = origin - separated['vocals']
    
    silence1 = torch.zeros([2, start_silence * separator.samplerate])
    silence2 = torch.zeros([2, end_silence * separator.samplerate])
    inst = torch.cat((silence1, inst, silence2), dim=-1)
    
    demucs.api.save_audio(inst, output, samplerate=separator.samplerate)
    return output
    
def work(audiopath, outputpath):
    model, _ = load_whisper(None)
    separator = load_separator()
    
    t1 = time.time()
    result, info = model.transcribe(audiopath, word_timestamps=True)

    model, reloaded = load_whisper(info.language)

    if reloaded:
        result, info = model.transcribe(audiopath, word_timestamps=True)

    segments = segment(result)
    start_silence = 1
    assdata = make_ass_swap(segments, offset=start_silence + 0.1)
    asspath = f'{audiopath}.ass'
    
    soundpath = instrumental(separator, audiopath, f'{audiopath}_inst.mp3', start_silence=start_silence, end_silence=start_silence)
    
    open(asspath, 'w', encoding='utf8').write(assdata)

    try:
        process = subprocess.run(['ffmpeg', '-y', '-f', 'lavfi', '-i', 'color=c=black:s=1280x720', '-i', soundpath, '-shortest', '-fflags', '+shortest', '-vf', f'subtitles={asspath}', '-vcodec', 'h264', outputpath], capture_output=True)
        if process.returncode != 0:
            raise RuntimeError(f'ffmpeg failed: {process.stderr}')
    finally:
        pass
        #os.remove(asspath)
        #os.remove(soundpath)
        
    t2 = time.time()

    print(f't: {t2 - t1}')

def main(argv):
    if len(argv) >= 2 and argv[1] == '-cli':
        if len(argv) == 2:
            print(f'Usage {argv[0]} -cli <input> [output]')
            return
            
        input = argv[2]
        
        if not os.path.isfile(input):
            #maybe youtube?
            input, title = youtube_to_mp3(input)
            
        if len(argv) == 3:
            last_dot = input.rfind('.') if '.' in input else None
            output_path = f'{input[:last_dot]}.mp4'
        else:
            output_path = argv[3]

        return work(input, output_path)
    
    st.title("Karaoke Generator")
    input_type = st.radio("Input Type", ["Local File", "YouTube Link"])
    if input_type == "Local File":
        uploaded_file = st.file_uploader("Choose an audio file (mp3 or wav)", type=["mp3", "wav"])
        if uploaded_file is not None:
            file = Path(f'./uploads/{uploaded_file.name}')
            local_path = f'./uploads/{uploaded_file.name}'
            with open(file, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File uploaded: {file}")
    else:
        youtube_url = st.text_input("YouTube Link")
        if youtube_url:
            local_path, title = youtube_to_mp3(youtube_url)
            print(youtube_url)
            st.success(f"Downloaded YouTube audio: {local_path}")
    output_path = st.text_input("Output MP4 File Path (Optional)", 'RESULT.mp4')
    if st.button("Generate Karaoke Video"):
        with st.spinner('Processing...'):
            work(local_path, output_path)
        st.success("Karaoke video generated!")
        st.video(output_path)

if __name__ == '__main__':
    main(sys.argv)


import time, itertools, pprint, subprocess, sys, math, os, collections, re
import unicodedata
from pathlib import Path
import argparse
import torch
from faster_whisper import WhisperModel, vad
import whisperx
import demucs.api
import yt_dlp
import streamlit as st
from collections import namedtuple
Word = namedtuple('Word', ['word', 'start', 'end'])
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="unhashable type: 'list'", category=UserWarning)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
whisper_models = {None: 'tiny','he': 'ivrit-ai/faster-whisper-v2-d3-e3'}
loaded_model_name = None
loaded_model = None

def replace_ext(path, ext):
    if '.' not in path:
        return f'{path}{ext}'
    return f"{path[:path.rfind('.')]}{ext}"

def youtube_download(url, audio_only=True):
    ext = 'mp3' if audio_only else 'mp4'
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': f'%(id)s.{ext}',
    }
    if audio_only:
        ydl_opts['extract_audio'] = True
        ydl_opts['format'] = 'bestaudio'
        
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        outfile = f'{info["id"]}.{ext}'
    return outfile, info['title']

def whisperX_load(model,audio):
    result = model.transcribe(audio, batch_size=16, print_progress=True)
    language = result['language']
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device='cpu')
    return whisperx.align(result["segments"], model_a, metadata, audio, return_char_alignments=False,
                            device='cpu'), language

def load_model_type(type='faster',model_name='tiny'):
    if type=='faster':
        return WhisperModel(model_name, device="auto", compute_type="int8")
    else: #whisperX
        return whisperx.load_model(model_name, device="cpu", compute_type="int8", vad_options={"vad_onset": 0.05, "vad_offset": 0.05})

def load_whisper(lang,model_type):
    global loaded_model_name, loaded_model
    model_name = whisper_models.get(lang, whisper_models[None])
    if model_name == loaded_model_name:
        return loaded_model, False

    loaded_model = load_model_type(type=model_type,model_name=model_name)

    print(f'loaded {model_name}')
    loaded_model_name = model_name
    return loaded_model, True

loaded_separator = None
def load_separator():
    global loaded_separator
    if loaded_separator is not None:
        return loaded_separator
    loaded_separator = demucs.api.Separator(model='htdemucs')
    return loaded_separator

def segment(result,model_type):
    if model_type=='faster':
        result = [segment.words for segment in result]
    else: #whisperX
        for i, segment in enumerate(result['segments']):
            result['segments'][i] = [Word(word=f" {w['word']}", start=w['start'], end=w['end']) for w in segment['words']]
        result = result['segments']

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

def instrumental(input, output_inst, output_vocals=None, start_silence=0, end_silence=0):
    separator = load_separator()
    
    origin, separated = separator.separate_audio_file(input)
    inst = origin - separated['vocals']
    
    silence1 = torch.zeros([2, start_silence * separator.samplerate])
    silence2 = torch.zeros([2, end_silence * separator.samplerate])
    
    inst = torch.cat((silence1, inst, silence2), dim=-1)
    
    if output_vocals:
        vocals = torch.cat((silence1, separated['vocals'], silence2), dim=-1)
        demucs.api.save_audio(vocals, output_vocals, samplerate=separator.samplerate)
        
    demucs.api.save_audio(inst, output_inst, samplerate=separator.samplerate)
    
def make_lyrics_video(audiopath, outputpath, model_type,transcribe_using_vocals=True):
    #preload for timing purposes
    load_separator()
    load_whisper(None,model_type)
    
    asspath = replace_ext(audiopath, '.ass')
    instpath = replace_ext(audiopath, '_inst.mp3')
    vocalspath = replace_ext(audiopath, '_vocals.mp3') if transcribe_using_vocals else None
    
    t1 = time.time()
    
    model, _ = load_whisper(None,model_type)
    
    silence = 1
    instrumental(audiopath, instpath, output_vocals=vocalspath, start_silence=silence, end_silence=silence)
    
    if transcribe_using_vocals:
        silence = 0
        #use vocals only
        audiopath = vocalspath
        transcribe_params = {} #dict(vad_filter=True, vad_parameters=vad.VadOptions(speech_pad_ms=50))
    else:
        transcribe_params = {}
    if model_type=='faster':
        result, info = model.transcribe(audiopath, word_timestamps=True, **transcribe_params)
        language = info.language
    else: #whisperX
        audio = whisperx.load_audio(audiopath)
        result, language = whisperX_load(model,audio)

    model, reloaded = load_whisper(language,model_type)

    if reloaded:
        if model_type=='x': #whisperX
            result, language = whisperX_load(model,audio)
        else: #faster
            result, info = model.transcribe(audiopath, word_timestamps=True)

    segments = segment(result,model_type)
    assdata = make_ass_swap(segments, offset=silence + 0.0)
 
    
    open(asspath, 'w', encoding='utf8').write(assdata)

    try:
        process = subprocess.run(['ffmpeg', '-y', '-f', 'lavfi', '-i', 'color=c=black:s=1280x720', '-i', instpath, '-shortest', '-fflags', '+shortest', '-vf', f'subtitles={asspath}', '-vcodec', 'h264', outputpath], capture_output=True)
        if process.returncode != 0:
            raise RuntimeError(f'ffmpeg failed: {process.stderr}')
    finally:
        pass
        #os.remove(asspath)
        #os.remove(instpath)
        
    t2 = time.time()

    print(f't: {t2 - t1}')

def remove_vocals_from_video(mp4_input, output_path):
    instpath = replace_ext(mp4_input, '_inst.mp3')
    instrumental(mp4_input, instpath)
    mp4_input_name, mp4_input_ext = os.path.splitext(mp4_input)
    new_mp4_input = f"{mp4_input_name}-original{mp4_input_ext}"
    os.rename(mp4_input,new_mp4_input)
    try:
        process = subprocess.run(['ffmpeg', '-y', '-i', new_mp4_input, '-i', instpath, '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', '-map', '0:v:0', '-map', '1:a:0','-shortest', output_path], capture_output=True)
        if process.returncode != 0:
            raise RuntimeError(f'ffmpeg failed: {process.stderr}')
    finally:
        pass
        os.remove(instpath)

def main(argv):
    parser = argparse.ArgumentParser(
        description="A karaoke tool to process input songs and create karaoke videos."
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        default=None,
        help="Your song for karaoke (local file or youtube link)"
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help="Your mp4 karaoke song output file"
    )
    parser.add_argument(
        '-ahl', '--already-has-lyrics',
        action='store_true',
        default=False,
        help="Add if its already a lyric/Karaoke video"
    )
    parser.add_argument(
        '-t', '--model-type',
        type=str,
        default='faster',
        help="change between faster_whisper (faster) & whisperX (x)"
    )
    args = parser.parse_args()

    if len(argv) >= 2: # CLI Mode
        input = args.input

        if not os.path.isfile(input):  # YT Link
            input, title = youtube_download(input, audio_only=not args.already_has_lyrics)

        if args.output != None:
            output_path = args.output
        else:
            output_path = replace_ext(input, '.mp4')

        if args.already_has_lyrics:
            return remove_vocals_from_video(input, output_path)
        else:
            return make_lyrics_video(input, output_path,args.model_type)

    st.title("Karaoke Generator")
    input_type = st.radio("Input Type", ["Local File", "YouTube Link Without Lyrics", "YouTube Link With Lyrics"])
    if input_type == "Local File":
        uploaded_file = st.file_uploader("Choose an audio file (mp3 or wav)", type=["mp3", "wav"])
        if uploaded_file is not None:
            file = Path(f'./uploads/{uploaded_file.name}')
            local_path = f'./uploads/{uploaded_file.name}'
            output_path = replace_ext(local_path, '.mp4')
            with open(file, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File uploaded: {file}")
    elif input_type == "YouTube Link Without Lyrics":
        youtube_url = st.text_input("YouTube Link Without Lyrics")
        if youtube_url:
            local_path, title = youtube_download(youtube_url, audio_only=True)
            output_path = replace_ext(local_path, '.mp4')
            print(youtube_url)
            st.success(f"Downloaded YouTube audio: {local_path}")
    else:
        youtube_url = st.text_input("YouTube Link With Lyrics")
        if youtube_url:
            local_path, title = youtube_download(youtube_url, audio_only=False)
            output_path = replace_ext(local_path, '.mp4')
            print(youtube_url)
            st.success(f"Downloaded YouTube video: {local_path}")
    if st.button("Generate Karaoke Video"):
        with st.spinner('Processing...'):
            if input_type == 'YouTube Link With Lyrics':
                remove_vocals_from_video(local_path, output_path)
            else:
                make_lyrics_video(local_path, output_path)
        st.success("Karaoke video generated!")
        st.video(output_path)

if __name__ == '__main__':
    main(sys.argv)


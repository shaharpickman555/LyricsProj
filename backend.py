import time, itertools, pprint, subprocess, sys, math, os, collections, re, hashlib, base64, shutil, threading, ctypes, traceback
from dataclasses import dataclass
import unicodedata
from pathlib import Path
import argparse
import torch
import demucs.api
import yt_dlp
import streamlit as st
from collections import namedtuple
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

has_juice = torch.cuda.device_count() > 0
device_str = 'cuda' if has_juice else 'cpu'
model_size_str = 'large-v3' if has_juice else 'tiny'

Word = namedtuple('Word', ['word', 'start', 'end'])

whisper_model_frameworks = {
                    'faster':
                    {
                        None: (model_size_str, 'faster'),
                        #'he': ('ivrit-ai/faster-whisper-v2-d3-e3', 'faster'),
                    },
                    'whisperx':
                    {
                        None: (model_size_str, 'whisperx'),
                        #whisperx he?
                    },
                 }

whisper_models = None
def set_model_framework(model_framework):
    global whisper_models
    whisper_models = whisper_model_frameworks[model_framework]
    
    if model_framework == 'faster':
        from faster_whisper import WhisperModel
    elif model_framework == 'whisperx':
        import whisperx
    else:
        raise ValueError(f'Unknown framework: {model_framework}')
    
def import_model_module():
    global whisper_models
    
    

loaded_model_desc = None
loaded_model = None
def transcribe_audio(audio, lang_hint=None):
    global loaded_model_desc, loaded_model
    model_desc = whisper_models.get(lang_hint, whisper_models[None])
    
    model_name, model_framework = model_desc
    
    if model_desc != loaded_model_desc:
        if model_framework == 'faster':
            loaded_model = WhisperModel(model_name, device=device_str, compute_type='int8')
            transcribe_options = {'word_timestamps': True}
        elif model_framework == 'whisperx':
            loaded_model = whisperx.load_model(model_name, device=device_str, compute_type='int8', vad_options={'vad_onset': 0.05, 'vad_offset': 0.05}, asr_options={'multilingual': True, 'hotwords': None})
            transcribe_options = {}
        else:
            raise ValueError(f'unknown model {model_desc}')
        print(f'loaded model {model_framework} {model_name} to {device_str}')
        
    if model_framework == 'whisperx' and isinstance(audio, str):
        #path
        audio = whisperx.load_audio(audio)
    
    result = loaded_model.transcribe(audio, **transcribe_options)
    language = result.get('language') if isinstance(result, dict) else result[1].language
    
    if model_desc != whisper_models.get(language, whisper_models[None]):
        #try again with correct language hint
        return transcribe_audio(audio, lang_hint=language)
    
    return result, audio
    
loaded_align_model_lang = None
loaded_align_model = None
def align_audio(transcribe_result):
    global loaded_align_model_lang, loaded_align_model
    
    model_result, audio = transcribe_result
    
    need_alignment = isinstance(model_result, dict)
    
    if need_alignment:
        #whisperX
        language = model_result['language']
        if loaded_align_model_lang != language:
            loaded_align_model = whisperx.load_align_model(language_code=language, device=device_str)
            loaded_align_model_lang = language
            
        model_result = whisperx.align(model_result['segments'], *loaded_align_model, audio, return_char_alignments=False, device=device_str)
        return [[Word(word=f' {w['word']}', start=w['start'], end=w['end']) for w in segment.get('words', segment['word_segments'])] for segment in model_result['segments']]
    else:
        #faster
        result, info = model_result
        return [segment.words for segment in result]

loaded_separator = None
def load_separator():
    global loaded_separator
    if loaded_separator is not None:
        return loaded_separator
    loaded_separator = demucs.api.Separator(model='htdemucs')
    return loaded_separator

local_upload_dir = 'uploads/'
local_cache_dir = 'songs/'

def getitem(l, i, default=None):
    try:
        return l[i]
    except IndexError:
        return default

def replace_ext(path, ext):
    if '.' not in path:
        return f'{path}{ext}'
    return f'{path[:path.rfind('.')]}{ext}'
    
def youtube_info(url):
    with yt_dlp.YoutubeDL() as ydl:
        info = ydl.extract_info(url, download=False)
        
    return info['id'], info['title']

def youtube_download(url, local_dir, audio_only=True, dont_cache=False):
    ext = 'mp3' if audio_only else 'mp4'
    ydl_opts = {
        'format': 'mp4',
        'outtmpl': os.path.join(local_dir, f'%(id)s.{ext}'),
        'nooverwrites': True,
    }
    
    if audio_only:
        ydl_opts['extract_audio'] = True
        ydl_opts['format'] = 'bestaudio'

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=dont_cache)
        outfile = os.path.join(local_dir, f'{info['id']}.{ext}')
        if not os.path.exists(outfile) and not dont_cache:
            ydl.extract_info(url, download=True)
    return outfile, info['title']

def segment(result):
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
    rtl_count = count['R'] + count['AL'] + count['RLE'] + count['RLI']
    ltr_count = count['L'] + count['LRE'] + count['LRI']
    return 'rtl' if rtl_count > ltr_count else 'ltr'

def ass_time(seconds):
    return f'{int(seconds // 3600)}:{int(seconds % 3600 // 60):02d}:{int(seconds % 60):02d}.{int(seconds * 100 % 100):02d}'

def output_line(words, selected_word):
    text_wrap = r'{{\c&H{color}&}}{text}{{\r}}'
    marked_color = '0000FF'
    unmarked_colors = ['FFFFFF', 'FEFEFE']

    direction = dominant_strong_direction(''.join(w.word for w in words))

    wrapped = [text_wrap.format(text=w.word, color=marked_color if w == selected_word else unmarked_colors[i % 2]) for i, w in enumerate(words)]

    return ''.join(reversed(wrapped) if direction == 'rtl' else wrapped)

def ass_circle(start_layer, x, y, start_time, end_time, fadein_time):
    mid_time = (end_time + start_time) / 2
    return [rf'Dialogue: {start_layer}, {ass_time(start_time - fadein_time)}, {ass_time(mid_time)}, S1, {{\pos({x}, {y})}}{{\fad({int(fadein_time * 1000)}, 0)}}{{\p1}}m 0 0 b 20 0 20 50 0 50{{\p0}}',
            rf'Dialogue: {start_layer + 2}, {ass_time(start_time - fadein_time)}, {ass_time(mid_time)}, S1, {{\pos({x}, {y})}}{{\fad({int(fadein_time * 1000)}, 0)}}{{\p1}}m 0 50 b -20 50 -20 0 0 0{{\p0}}{{\r}}',
            rf'Dialogue: {start_layer + 1}, {ass_time(start_time)}, {ass_time(end_time)}, B1, {{\pos({x + 2.5}, {y})}}{{\org({x - 10}, {y})}}{{\t(\frz-360)}}{{\p1}}m 0 55 b -25 55 -25 0 0 0{{\p0}}{{\r}}',
            rf'Dialogue: {start_layer}, {ass_time(mid_time)}, {ass_time(end_time)}, S1, {{\pos({x}, {y})}}{{\p1}}m 0 50 b -20 50 -20 0 0 0{{\p0}}{{\r}}']


def make_ass_swap(segments, offset, prepare_time_seconds=5):
    header = '''[Script Info]
PlayResX: 800
PlayResY: 800
WrapStyle: 1

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Alignment, Encoding
Style: S1, Arial, 80, &HFFFFFF, &HFFFFFF, &H202020, &H000000, 5, 0
Style: B1, Arial, 80, &H000000, &H000000, &H202020, &H000000, 5, 0

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
            ass_lines.append(f'Dialogue: 0, {ass_time(actual_appear_time + offset)}, {ass_time(actual_disappear_time + offset)}, S1, {{\\pos(400, {ystart + line_y_off})}}{{\\fad(1000, 1000)}}{output_line(line, None)}')

            for word in line:
                #output colored lines on top
                ass_lines.append(f'Dialogue: 1, {ass_time(word.start + offset)}, {ass_time(word.end + offset)}, S1, {{\\pos(400, {ystart + line_y_off})}}{output_line(line, word)}')

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

def make_lyrics_video(audiopath, outputpath, transcribe_using_vocals=True, remove_intermediates=True):
    asspath = replace_ext(audiopath, '.ass')
    instpath = replace_ext(audiopath, '_inst.mp3')
    vocalspath = replace_ext(audiopath, '_vocals.mp3') if transcribe_using_vocals else None

    t1 = time.time()

    silence = 1
    instrumental(audiopath, instpath, output_vocals=vocalspath, start_silence=silence, end_silence=silence)

    if transcribe_using_vocals:
        silence = 0
        #use vocals only
        audiopath = vocalspath
        
    result = transcribe_audio(audiopath)
    segments = align_audio(result)

    segments = segment(segments)
    assdata = make_ass_swap(segments, offset=silence + 0.0)


    open(asspath, 'w', encoding='utf8').write(assdata)

    try:
        process = subprocess.run(['ffmpeg', '-y', '-f', 'lavfi', '-i', 'color=c=black:s=1280x720', '-i', instpath, '-shortest', '-fflags', '+shortest', '-vf', f'subtitles={asspath}', '-vcodec', 'h264', outputpath], capture_output=True)
        if process.returncode != 0:
            raise RuntimeError(f'ffmpeg failed: {process.stderr}')
    finally:
        if remove_intermediates:
            os.remove(asspath)
            os.remove(instpath)
            os.remove(vocalspath)

    t2 = time.time()

    print(f't: {t2 - t1}')

def remove_vocals_from_video(mp4_input, output_path):
    instpath = replace_ext(mp4_input, '_inst.mp3')
    instrumental(mp4_input, instpath)
    mp4_input_name, mp4_input_ext = os.path.splitext(mp4_input)
    new_mp4_input = f'{mp4_input_name}-original{mp4_input_ext}'
    os.rename(mp4_input, new_mp4_input)
    try:
        process = subprocess.run(['ffmpeg', '-y', '-i', new_mp4_input, '-i', instpath, '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', '-map', '0:v:0', '-map', '1:a:0', '-shortest', output_path], capture_output=True)
        if process.returncode != 0:
            raise RuntimeError(f'ffmpeg failed: {process.stderr}')
    finally:
        pass
        os.remove(instpath)
        
def passthrough(input, output):
    shutil.copy(input, output)
        
def digest(path=None, content=None):
    return base64.b64encode(hashlib.sha256(open(path, 'rb').read() if path else content).digest()[:15], altchars=b'+-').decode()

def generate_with_cache(f, local_path, selectors, dont_cache=False, **kwargs):
    name = replace_ext(os.path.basename(local_path), '')
    out_path = os.path.join(local_cache_dir, f'{'_'.join(f'{str(k)}={str(v)}' for k,v in selectors.items())}_{name}.mp4')
    
    if dont_cache or not os.path.exists(out_path):
        f(local_path, out_path, **kwargs)
        
    return out_path
    
def has_cache(local_path, selectors):
    name = replace_ext(os.path.basename(local_path), '')
    out_path = os.path.join(local_cache_dir, f'{'_'.join(f'{str(k)}={str(v)}' for k,v in selectors.items())}_{name}.mp4')
    
    return os.path.exists(out_path)
    
def canonify_input_file(path=None, content=None):
    h = digest(path=path) if path else digest(content=content)

    canon = os.path.join(local_cache_dir, f'{h}.input')
    if path is None:
        with open(canon, 'wb') as fh:
            fh.write(content)
    elif path != canon:
        shutil.move(path, canon)
    return canon
    
#########################threading#####################

class StopException(Exception):
    pass
    
@dataclass(frozen=True, eq=False)
class Job:
    tid : int = None
    title : str = None
    path : str = None
    url : str = None
    data : bytes = None
    keep : str = 'nothing'
    model_type : str = ''
    no_cache : bool = False
    arg : dict = None
    
    #can change
    status : str = 'idle'
    out_path : str = None
    
    def __post_init__(self):
        if self.keep not in ('nothing', 'video', 'all'):
            raise ValueError('job.keep must be one of: nothing, video, all')
            
        not_nones = int(self.url is not None) + int(self.path is not None) + int(self.data is not None)
        if not_nones != 1:
            raise ValueError('must supply: job.url, job.path, job.data')
            
        if self.tid is None:
            object.__setattr__(self, 'tid', generate_tid())
        if self.arg is None:
            object.__setattr__(self, 'arg', {})
            
        if self.title is None:
            if self.url:
                id, title = youtube_info(self.url)
            elif self.path:
                title = os.path.splitext(os.path.basename(self.path))[0]
            elif self.data:
                title = digest(content=self.data)
            
            object.__setattr__(self, 'title', title)
            
    def __eq__(self, other):
        return self.tid == other.tid
        
    def update_status_locked(self, status, out_path=None):
        object.__setattr__(self, 'status', status)
        if out_path:
            object.__setattr__(self, 'out_path', out_path)


def raise_exception_in_thread(thread, e):
    ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.native_id, ctypes.py_object(e))
    
lock = None
event = None
worker_thread = None
should_stop = False
job_queue = None
job_done_cb = None
max_job_history = 1000
default_model_type = 'faster'

def work_loop():
    global should_stop, job_queue, lock, event, job_done_cb
    try:
        while not should_stop:
            job = None
            with lock:
                event.clear()
                if job_queue:
                    job = job_queue[0]
                    job_queue = job_queue[1:]
                    if job.status != 'idle':
                        continue
                    job.update_status_locked('processing')
            
            if job is None:
                event.wait()
                continue
                
            try:
                output = None
                
                #work
                actions = {'nothing': make_lyrics_video, 'video': remove_vocals_from_video, 'all': passthrough}
                
                #TODO cache youtube download?
                
                if job.url is not None:
                    download_path, title = youtube_download(job.url, local_upload_dir, audio_only=job.keep == 'nothing', dont_cache=job.no_cache)
                    path = canonify_input_file(content=open(download_path, 'rb').read()) #don't delete youtube video file
                elif job.path is not None:
                    path = canonify_input_file(path=job.path)
                else:
                    path = canonify_input_file(content=job.data)
                
                set_model_framework(job.model_type or default_model_type)
                output = generate_with_cache(actions[job.keep], path, selectors=dict(keep=job.keep), dont_cache=job.no_cache)
                
                #TODO remove canon input file?
                
                with lock:
                    job.update_status_locked('done', output)
                    
                job_done_cb(job, None)
            except Exception as e:
                if isinstance(e, (StopException, KeyboardInterrupt)):
                    raise
                    
                with lock:
                    job.update_status_locked('error')

                job_done_cb(job, e)
    except StopException:
        pass


def init_thread(pop_cb):
    global worker_thread, job_queue, should_stop, lock, event, job_done_cb
    
    os.makedirs(local_upload_dir, exist_ok=True)
    os.makedirs(local_cache_dir, exist_ok=True)
    
    stop_thread()
        
    should_stop = False
    lock = threading.Lock()
    event = threading.Event()
    job_queue = tuple()
    job_done_cb = pop_cb
    worker_thread = threading.Thread(target=work_loop)
    worker_thread.start()
    
    
def stop_thread():
    global worker_thread, should_stop, event
    
    if worker_thread is not None:
        should_stop = True
        event.set()
        raise_exception_in_thread(worker_thread, StopException)
        worker_thread.join()
        worker_thread = None
    
def job_in_queue(job):
    global job_queue, lock
    with lock:
        return job in job_queue

tid_counter = 0
def generate_tid():
    global tid_counter
    tid_counter += 1
    return tid_counter
    
def set_queue(jobs : list[Job]):
    global lock, event, job_queue
    
    with lock:
        job_queue = tuple(j for j in jobs)
        event.set()
        

def thread_test():
    def cb(job, error):
        if error:
            print(f'{job.tid} error: ', traceback.format_exc(error))
        else:
            print(f'{job.tid} is available at {job.out_path} ({job.status})')
            
    init_thread(cb)
    #job4 = Job(path=r'C:\projects\pick\LyricsProj\songs\allstar.wav')
    job1 = Job(url='https://www.youtube.com/watch?v=L_jWHffIx5E')
    job2 = Job(url='https://www.youtube.com/watch?v=L_jWHffIx5E', keep='video')
    job3 = Job(url='https://www.youtube.com/watch?v=L_jWHffIx5E', keep='all')
    jobs = [job1, job2, job3]

    set_queue(jobs)

    try:
        while True:
            print(job1.status, job2.status, job3.status)
            time.sleep(1)
    finally:
        stop_thread()
        
#####################################################################################

def main(argv):
    parser = argparse.ArgumentParser(
        description='A karaoke tool to process input songs and create karaoke videos.'
    )
    parser.add_argument(
        '-i', '--input',
        type=str,
        default=None,
        help='Your song for karaoke (local file or youtube link)'
    )
    parser.add_argument(
        '-k', '--keep-video',
        action='store_true',
        default=False,
        help='Add if its already a lyric/Karaoke video'
    )
    parser.add_argument(
        '-t', '--model-type',
        type=str,
        default='whisperx',
        help='change between faster whisper (faster) & whisperX (whisperx)'
    )
    parser.add_argument(
        '-d', '--dont-use-cache',
        action='store_true',
        default=False,
        help='Dont use local video if already generated'
    )
    args = parser.parse_args()
    
    os.makedirs(local_upload_dir, exist_ok=True)
    os.makedirs(local_cache_dir, exist_ok=True)
    
    set_model_framework(args.model_type)

    if len(argv) >= 2: # CLI Mode
        input = args.input

        if not os.path.isfile(input):  # YT Link
            input, title = youtube_download(input, local_upload_dir, audio_only=not args.keep_video)
            
        input = canonify_input_file(content=open(input, 'rb').read()) #dont move input file
        generate_with_cache(remove_vocals_from_video if args.keep_video else make_lyrics_video, input, selectors=dict(keep='video' if args.keep_video else 'nothing'), dont_cache=args.dont_use_cache)
        return
        
    st.title('Karaoke Generator')
    input_type = st.radio('Input Type', ['Local File', 'YouTube Link Without Lyrics', 'YouTube Link With Lyrics'])
    if input_type == 'Local File':
        uploaded_file = st.file_uploader('Choose an audio file (mp3 or wav)', type=['mp3', 'wav'])
        if uploaded_file is not None:
            input = canonify_input_file(content=uploaded_file.getbuffer())
            st.success(f'File uploaded: {uploaded_file.name}')
    else:
        audio_only = input_type == 'YouTube Link Without Lyrics'
        youtube_url = st.text_input('YouTube Link Without Lyrics')
        if youtube_url:
            local_path, title = youtube_download(youtube_url, local_upload_dir, audio_only=audio_only)
            input = canonify_input_file(path=local_path)
            st.success(f'Downloaded YouTube audio: {youtube_url}')
            
    if st.button('Generate Karaoke Video'):
        with st.spinner('Processing...'):
            if input_type == 'YouTube Link With Lyrics':
                output = generate_with_cache(remove_vocals_from_video, input, selectors=dict(keep='video'), dont_cache=False)
            else:
                output = generate_with_cache(make_lyrics_video, input, selectors=dict(keep='nothing'), dont_cache=False)
        st.success('Karaoke video generated!')
        st.video(output)

if __name__ == '__main__':
    #main(sys.argv)
    thread_test()

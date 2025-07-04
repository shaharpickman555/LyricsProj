import time, itertools, pprint, subprocess, sys
import math, os, collections, re, hashlib, base64
import shutil, threading, ctypes, traceback, inspect
import stat, logging, signal, argparse, gc, datetime
from dataclasses import dataclass
from collections import namedtuple
from pathlib import Path

import requests
import unicodedata
import torch
import demucs.api
import yt_dlp
import torch
import whisperx
import faster_whisper

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

has_juice = torch.cuda.device_count() > 0
device_str = 'cuda' if has_juice else 'cpu'
compute_type_str = 'int8' if has_juice else 'int8'

detection_model_name = 'large-v3' if has_juice else 'tiny'

Word = namedtuple('Word', ['word', 'start', 'end'])

ffmpeg_path = os.path.join(os.path.dirname(__file__), 'ffmpeg')
if not os.path.exists(ffmpeg_path):
    logger.warning('Falling back to system ffmpeg')
    ffmpeg_path = shutil.which('ffmpeg')

local_upload_dir = 'uploads/'
local_cache_dir = 'songs/'
max_job_history = 1000
max_job_duration = 20 * 60
max_job_filesize = 100 * 1024 * 1024
default_model_type = 'faster'

max_local_dir_size = 1024 ** 3
max_local_dir_num_files = 10

YOUTUBE_DOWNLOAD_PROGRESS = 0.1 #10% download, 90% everything else

DETECT_LANGUAGE_PROGRESS = 0.1
LOAD_TRANSCRIBE_MODEL_PROGRESS = 0.1 #10% detection, 10% load, 80% process

LOAD_SEPARATOR_MODEL_PROGRESS = 0.1
SEPARATOR_MODEL_PROGRESS = 0.75 #10% load, 75% process, 15% save

LYRICS_VIDEO_SEPARATOR_PROGRESS = 0.5
LYRICS_VIDEO_TRANSCRIBE_PROGRESS = 0.3
LYRICS_VIDEO_ALIGN_PROGRESS = 0.05  #50% separator, 30% transcribe, 5% align, 15% ffmpeg

REMOVE_VOCALS_SEPARATOR_PROGRESS = 5/7

default_model_name = 'large-v3' if has_juice else 'tiny'
heb_model_name = 'ivrit-ai/whisper-large-v3-ct2' if has_juice else 'tiny'

whisper_model_frameworks = {
                    'faster':
                    {
                        None: (default_model_name, 'faster', {}, dict(vad_filter=True, vad_parameters=faster_whisper.vad.VadOptions(threshold=0.5, max_speech_duration_s=2))),
                        'he': (heb_model_name, 'faster', {}, dict(patience=2, beam_size=5, vad_filter=True, vad_parameters=faster_whisper.vad.VadOptions(threshold=0.5, max_speech_duration_s=2))),
                    },
                    'whisperx':
                    {
                        None: (default_model_name, 'whisperx', {}, {}),
                        'he': (heb_model_name, 'whisperx', dict(patience=2, beam_size=5, multilingual=True), {}),
                    },
                 }
                 
alignment_model_override = {
    #'he': 'imvladikon/wav2vec2-xls-r-1b-hebrew', #"imvladikon/wav2vec2-xls-r-300m-hebrew", #'imvladikon/wav2vec2-xls-r-300m-lm-hebrew'
}

def set_debug(debug):
    if not debug and not has_juice:
        raise RuntimeError('No GPU on release')
    if debug:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

whisper_models = None
whisper_model_framework = None
def set_model_framework(model_framework):
    global whisper_models, whisper_model_framework
    whisper_models = whisper_model_frameworks[model_framework]
    whisper_model_framework = model_framework
    
    if model_framework == 'faster':
        global WhisperModel
        from faster_whisper import WhisperModel
    elif model_framework == 'whisperx':
        global whisperx
        import whisperx
    else:
        raise ValueError(f'Unknown framework: {model_framework}')
        
loaded_detection_model = None
def detect_audio(audio):
    global loaded_detection_model, WhisperModel
    
    from faster_whisper.audio import decode_audio

    if loaded_detection_model is None:
        from faster_whisper import WhisperModel
        loaded_detection_model = WhisperModel(detection_model_name, device=device_str, compute_type='int8')
    
    audio = decode_audio(audio, sampling_rate=loaded_detection_model.feature_extractor.sampling_rate)
    
    lang, conf, scores = loaded_detection_model.detect_language(audio, language_detection_segments=1000, language_detection_threshold=1.0)
    
    return lang
    
class WhisperXProgressHook:
    def __init__(self, cb):
        self.cb = cb
        
    def __bool__(self):
        frames = inspect.stack()
        loc = frames[1].frame.f_locals
        if 'idx' in loc and 'total_segments' in loc:
            self.cb((loc['idx'] + 1) / loc['total_segments'])
        return False

loaded_model_desc = None
loaded_model = None
def transcribe_audio(audio, progress_cb=None, lang_hint=None):
    global loaded_model_desc, loaded_model
    
    if progress_cb:
        progress_cb(0.0)
    
    if not lang_hint:
        lang = detect_audio(audio)
        logger.info(f'detected: {lang}')
    else:
        lang = lang_hint
    
    if progress_cb:
        progress_cb(DETECT_LANGUAGE_PROGRESS)
    
    model_desc = whisper_models.get(lang, whisper_models[None])
    
    model_name, model_framework, model_options, transcribe_options = model_desc
    
    if model_desc != loaded_model_desc:
        logger.info(f'Replacing loaded model -> {model_desc}')
        loaded_model = None
        gc.collect()
        torch.cuda.empty_cache()
        
        if model_framework == 'faster':
            loaded_model = WhisperModel(model_name, device=device_str, compute_type=compute_type_str, **model_options)
            transcribe_options['word_timestamps'] = True
        elif model_framework == 'whisperx':
            loaded_model = whisperx.load_model(model_name, device=device_str, compute_type=compute_type_str, vad_options={'vad_onset': 0.05, 'vad_offset': 0.05}, asr_options={'multilingual': True, 'hotwords': None, **model_options})
        else:
            raise ValueError(f'unknown model {model_desc}')
        loaded_model_desc = model_desc
        logger.info(f'loaded model {model_framework} {model_name} to {device_str}')
    
    current_progress = DETECT_LANGUAGE_PROGRESS + LOAD_TRANSCRIBE_MODEL_PROGRESS
    
    if progress_cb:
        progress_cb(current_progress)
        
    def hook(p):
        if progress_cb:
            progress_cb(current_progress + (p * (1 - current_progress)))
        
    if model_framework == 'whisperx':
        transcribe_options['print_progress'] = WhisperXProgressHook(hook)
        
        if isinstance(audio, str):
            #path
            audio = whisperx.load_audio(audio)

    result = loaded_model.transcribe(audio, language=lang, **transcribe_options)
    
    if model_framework == 'faster':
        #apply generator
        segment_gen, info = result
        segments = []
        
        for segment in segment_gen:
            segments.append(segment)
            hook(segment.end / info.duration)
        
        result = segments, info
        
    if progress_cb:
        progress_cb(1.0)
        
    return result, audio
    
loaded_align_model_lang = None
loaded_align_model = None
def align_audio(transcribe_result, progress_cb=None):
    global loaded_align_model_lang, loaded_align_model
    
    if progress_cb:
        progress_cb(0.0)
    
    model_result, audio = transcribe_result
    
    if isinstance(model_result, dict):
        #whisperx
        segments = model_result['segments']
        language = model_result['language']
        need_alignment = True
    else:
        #faster
        segments, info = model_result
        segments = [dict(words=segment.words, start=segment.start, end=segment.end, text=segment.text) for segment in segments]
        language = info.language
        need_alignment = False #alignment does not work properly here (missing dictionary)
        
    if need_alignment:
        # no meaningful progress to report

        if loaded_align_model_lang != language:
            loaded_align_model = whisperx.load_align_model(language_code=language, device=device_str, model_name=alignment_model_override.get(language))
            loaded_align_model_lang = language

        align_result = whisperx.align(segments, *loaded_align_model, audio, return_char_alignments=False, device=device_str)

        segments = [[Word(word=f' {w["word"]}', start=w['start'], end=w['end']) for w in segment.get('words', segment.get('word_segments'))] for segment in align_result['segments']]
    else:
        segments = [segment['words'] for segment in segments]
    
    if progress_cb:
        progress_cb(1.0)
    return segments

loaded_separator = None
def load_separator():
    global loaded_separator
    if loaded_separator is not None:
        return loaded_separator
    loaded_separator = demucs.api.Separator(model='htdemucs')
    return loaded_separator

def getitem(l, i, default=None):
    try:
        return l[i]
    except IndexError:
        return default

def replace_ext(path, ext):
    if '.' not in path:
        return f'{path}{ext}'
    return f'{path[:path.rfind(".")]}{ext}'
    
def youtube_info(url):
    yt_opts = {
        'skip_download': True,
        'force_noplaylist': True,
        'extract_flat': 'in_playlist',
    }
    
    with yt_dlp.YoutubeDL(yt_opts) as ydl:
        info = ydl.extract_info(url)

    try:
        thumbnail = min(
            (t for t in info['thumbnails'] if 'height' in t),
            key=lambda t: t['height']
        )['url']
    except ValueError:
        thumbnail = info['thumbnails'][0]['url']

    return info['id'], info['title'], dict(duration=info['duration'],thumbnail=thumbnail)

def youtube_download(url, local_dir, audio_only=True, dont_cache=False, progress_cb=None):
    if progress_cb:
        progress_cb(0.0)
        
    ext = 'mp3' if audio_only else 'mp4'
    ydl_opts = {
        'outtmpl': os.path.join(local_dir, f'%(id)s.{ext}'),
        'nooverwrites': False,
        'format': 'bv*[height<2000]+ba/b[height<2000]',
        'merge_output_format': ext,
        'force_noplaylist': True,
    }
    
    if audio_only:
        ydl_opts['extract_audio'] = True
        ydl_opts['format'] = 'bestaudio'

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=dont_cache)
        outfile = os.path.join(local_dir, f'{info["id"]}.{ext}')
        if not os.path.exists(outfile) and not dont_cache:
            ydl.extract_info(url, download=True)
            
    if progress_cb:
        progress_cb(1.0)
    return outfile, info['title']

def segment(result):
    word_durations = [word.end - word.start for segment in result for word in segment]
    words_per_spoken_second = len(word_durations) / sum(word_durations)

    max_characters_per_line = 20
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

def output_word(word):
    if dominant_strong_direction(word) == 'rtl':
        punc = '.,;:/\\?!'
        word = word.strip()
        
        startswith = any(word.startswith(p) for p in punc)
        endswith = any(word.endswith(p) for p in punc)
        
        if startswith and endswith:
            word = word[-1] + word[1:-1] + word[0]
        elif startswith:
            word = word[1:] + word[0]
        elif endswith:
            word = word[-1] + word[:-1]
            
        return word
    else:
        return word.strip()
    
def output_line(words, selected_word):
    text_wrap = r'{{\c&H{color}&}}{text}{{\r}}'
    marked_color = '0000FF'
    unmarked_colors = ['FFFFFF', 'FEFEFE']

    direction = dominant_strong_direction(''.join(w.word for w in words))

    wrapped = [text_wrap.format(text=output_word(w.word), color=marked_color if w == selected_word else unmarked_colors[i % 2]) for i, w in enumerate(words)]

    return r'{\q2}' + ' '.join(reversed(wrapped) if direction == 'rtl' else wrapped)

def ass_circle(start_layer, x, y, start_time, end_time, fadein_time):
    mid_time = (end_time + start_time) / 2
    return [rf'Dialogue: {start_layer + 1}, {ass_time(start_time - fadein_time)}, {ass_time(mid_time)}, CW1, {{\pos({x}, {y})}}{{\fad({int(fadein_time * 1000)}, 0)}}{{\p1}}m 0 0 b 20 0 20 50 0 50{{\p0}}',
            rf'Dialogue: {start_layer + 3}, {ass_time(start_time - fadein_time)}, {ass_time(mid_time)}, CW1, {{\pos({x}, {y})}}{{\fad({int(fadein_time * 1000)}, 0)}}{{\p1}}m 0 50 b -20 50 -20 0 0 0{{\p0}}{{\r}}',
            rf'Dialogue: {start_layer + 2}, {ass_time(start_time)}, {ass_time(end_time)}, CB1, {{\pos({x + 2.5}, {y})}}{{\org({x - 10}, {y})}}{{\t(\frz-360)}}{{\p1}}m 0 55 b -25 55 -25 0 0 0{{\p0}}{{\r}}',
            rf'Dialogue: {start_layer + 1}, {ass_time(mid_time)}, {ass_time(end_time)}, CW1, {{\pos({x}, {y})}}{{\p1}}m 0 50 b -20 50 -20 0 0 0{{\p0}}{{\r}}',
            rf'Dialogue: {start_layer}, {ass_time(start_time - fadein_time)}, {ass_time(end_time)}, CB1, {{\pos({x}, {y})}}{{\fad({int(fadein_time * 1000)}, 0)}}{{\p1}}m 30 0 b 70 0 70 100 30 100 b -10 100 -10 0 30 0{{\p0}}']


def make_ass_swap(segments, offset, prepare_time_seconds=5):
    header = '''[Script Info]
PlayResX: 800
PlayResY: 800
WrapStyle: 1

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Alignment, Encoding, BorderStyle, Outline, Shadow, MarginL, MarginR, MarginV
Style: CW1, Arial, 80, &HFFFFFF, &HFFFFFF, &H000000, &H000000, 5, 0, 0, 0, 0, 0, 0, 0
Style: CB1, Arial, 80, &H000000, &H000000, &H000000, &H000000, 5, 0, 0, 0, 0, 0, 0, 0
Style: W1, Assistant, 80, &HFFFFFF, &HFFFFFF, &H000000, &H000000, 5, 0, 0, 15, 1, 30, 30, 30
Style: DW1, KlokanTech Noto Sans, 80, &HFFFFFF, &HFFFFFF, &H000000, &H000000, 5, 0, 0, 15, 1, 30, 30, 30

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
    
    #TODO check if assistant can present

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
            ass_lines.append(f'Dialogue: 0, {ass_time(actual_appear_time + offset)}, {ass_time(actual_disappear_time + offset)}, W1, {{\\pos(400, {ystart + line_y_off})}}{{\\fad(1000, 1000)}}{output_line(line, None)}')

            for word in line:
                #output colored lines on top
                ass_lines.append(f'Dialogue: 1, {ass_time(word.start + offset)}, {ass_time(word.end + offset)}, W1, {{\\pos(400, {ystart + line_y_off})}}{output_line(line, word)}')

            #counter
            if len(line) > 1:
                time_since_last_word = (line[0].start - segment[i - 1][-1].end) if i > 0 else (line[0].start - (last_segment_end or 0))
                if time_since_last_word >= prepare_time_seconds:
                    clock_duration = 4

                    ass_lines.extend(ass_circle(2, 200, 100, line[0].start - clock_duration + offset, line[0].start + offset, 0.5))

        last_segment_end = segment[-1][-1].end

    return header + '\n'.join(ass_lines)

def instrumental(audioinput, output_inst, output_vocals=None, start_silence=0, end_silence=0, progress_cb=None):
    if progress_cb:
        progress_cb(0.0)
        
    separator = load_separator()
    
    if progress_cb:
        progress_cb(LOAD_SEPARATOR_MODEL_PROGRESS)
         
    def separator_cb(data):
        if data['state'] != 'start':
            return
        progress = data['segment_offset'] / data['audio_length']
        if progress_cb:
            progress_cb(LOAD_SEPARATOR_MODEL_PROGRESS + (progress * SEPARATOR_MODEL_PROGRESS))
    
    separator.update_parameter(callback=separator_cb)

    origin, separated = separator.separate_audio_file(audioinput)
    inst = origin - separated['vocals']

    silence1 = torch.zeros([2, start_silence * separator.samplerate])
    silence2 = torch.zeros([2, end_silence * separator.samplerate])

    inst = torch.cat((silence1, inst, silence2), dim=-1)
    
    if progress_cb:
        progress_cb(LOAD_SEPARATOR_MODEL_PROGRESS + SEPARATOR_MODEL_PROGRESS)

    if output_vocals:
        vocals = torch.cat((silence1, separated['vocals'], silence2), dim=-1)
        demucs.api.save_audio(vocals, output_vocals, samplerate=separator.samplerate)

    demucs.api.save_audio(inst, output_inst, samplerate=separator.samplerate)
    
    if progress_cb:
        progress_cb(1.0)
        
def run_process(*args):
    process = subprocess.run(args, capture_output=True)
    if process.returncode != 0:
        raise RuntimeError(f'process failed: {process.args}\n\n{process.stderr.decode("utf8", errors="ignore")}')
    return process.stdout.decode("utf8", errors="ignore")
        
def is_video_audio(input):
    output_video = run_process('ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', input)
    output_audio = run_process('ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', input)
    return 'video' in output_video, 'audio' in output_audio
        
def extract_audio(input, output_audio):
    run_process(ffmpeg_path, '-y', '-i', input, '-vn', output_audio)
    
def audio_with_blank(audiopath, outputpath, subtitles_path=None):
    run_process(ffmpeg_path, '-y', '-f', 'lavfi', '-i', 'color=c=black:s=1280x720', '-i', audiopath, '-shortest', '-fflags', '+shortest', *(['-vf', f'subtitles={subtitles_path}:fontsdir=fonts'] if subtitles_path else []), '-vcodec', 'h264', outputpath)

def video_with_audio(videopath, audiopath, outputpath, subtitles_path=None):
    codec = 'libx264'
    run_process(ffmpeg_path, '-y', '-i', videopath, '-i', audiopath, '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', '-shortest', '-fflags', '+shortest', '-filter_complex', f"[0:v]scale='max(720, iw)':-2[v0]; [v0]subtitles={subtitles_path}:fontsdir=fonts[v]" if subtitles_path else "scale='max(720, iw)':-2[v]", '-map', '[v]:v', '-map', '1:a', '-vcodec', codec, outputpath)

def try_remove(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

def make_lyrics_video(inputpath, outputpath, transcribe_using_vocals=True, remove_intermediates=True, progress_cb=None, lang_hint=None, blank_video=False, **_):
    global whisper_model_framework
    
    if progress_cb:
        progress_cb(0.0)
        
    def instrumental_progress_cb(progress):
        if progress_cb:
            progress_cb(progress * LYRICS_VIDEO_SEPARATOR_PROGRESS)
            
    def transcribe_progress_cb(progress):
        if progress_cb:
            progress_cb(LYRICS_VIDEO_SEPARATOR_PROGRESS + (progress * LYRICS_VIDEO_TRANSCRIBE_PROGRESS))
            
    def align_progress_cb(progress):
        if progress_cb:
            progress_cb(LYRICS_VIDEO_SEPARATOR_PROGRESS + LYRICS_VIDEO_TRANSCRIBE_PROGRESS + (progress * LYRICS_VIDEO_ALIGN_PROGRESS))
    
    try:
        audiopath = replace_ext(inputpath, '_audio.wav')
        
        asspath = replace_ext(inputpath, '.ass')
        instpath = replace_ext(inputpath, '_inst.mp3')
        vocalspath = replace_ext(inputpath, '_vocals.mp3') if transcribe_using_vocals else None
        
        extract_audio(inputpath, audiopath)
        
        silence = 0 # TODO video title
        instrumental(audiopath, instpath, output_vocals=vocalspath, start_silence=silence, end_silence=silence, progress_cb=instrumental_progress_cb)

        if transcribe_using_vocals:
            #vocals already have silence accounted for
            silence = 0
            #use vocals only
            transcribepath = vocalspath
        else:
            transcribepath = audiopath
            
        result = transcribe_audio(transcribepath, progress_cb=transcribe_progress_cb, lang_hint=lang_hint)
        segments = align_audio(result, progress_cb=align_progress_cb)
        segments = segment(segments)
        assdata = make_ass_swap(segments, offset=silence + 0.0)
        
        with open(asspath, 'w', encoding='utf8') as fh:
            fh.write(assdata)
        
        if blank_video:
            audio_with_blank(instpath, outputpath, asspath)
        else:
            video_with_audio(inputpath, instpath, outputpath, asspath)
    finally:
        if remove_intermediates:
            try_remove(audiopath)
            try_remove(asspath)
            try_remove(instpath)
            if vocalspath:
                try_remove(vocalspath)
    
    if progress_cb:
        progress_cb(1.0)

def remove_vocals_from_video(mp4_input, output_path, remove_intermediates=True, progress_cb=None, blank_video=False, **_):
    if progress_cb:
        progress_cb(0.0)
        
    def instrumental_progress_cb(progress):
        if progress_cb:
            progress_cb(progress * REMOVE_VOCALS_SEPARATOR_PROGRESS)
        
    try:
        audiopath = replace_ext(mp4_input, '_audio.wav')
        instpath = replace_ext(mp4_input, '_inst.mp3')
        extract_audio(mp4_input, audiopath)
        
        instrumental(audiopath, instpath, progress_cb=instrumental_progress_cb)
        
        if blank_video:
            audio_with_blank(instpath, output_path)
        else:
            video_with_audio(mp4_input, instpath, output_path)
    finally:
        if remove_intermediates:
            try_remove(instpath)
            try_remove(audiopath)
        
    if progress_cb:
        progress_cb(1.0)
        
def passthrough(input, output, progress_cb=None, blank_video=False, **_):
    if progress_cb:
        progress_cb(0.0)
        
    if blank_video:
        audio_with_blank(input, output)
    else:
        run_process(ffmpeg_path, '-y', '-i', input, '-c:v', 'copy', '-vcodec', 'h264', output)
    
    if progress_cb:
        progress_cb(1.0)
        
def digest(path=None, content=None):
    return base64.b64encode(hashlib.sha256(open(path, 'rb').read() if path else content).digest()[:15], altchars=b'+-').decode()

def selectors_join(selectors):
    return '_'.join(f'{str(k)}={str(v)}' for k,v in selectors.items())
    
def generate_with_cache(f, local_path, selectors, dont_cache=False, **kwargs):
    name = replace_ext(os.path.basename(local_path), '')
    out_path = os.path.join(local_cache_dir, f'{selectors_join(selectors)}_{name}.mp4')
    
    if dont_cache or not os.path.exists(out_path):
        f(local_path, out_path, **kwargs)
        
    return out_path
    
def has_cache(local_path, selectors):
    name = replace_ext(os.path.basename(local_path), '')
    out_path = os.path.join(local_cache_dir, f'{selectors_join(selectors)}_{name}.mp4')
    
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

def dir_size_num_oldest(path):
    tot = 0
    num = 0
    oldest = None
    oldest_time = None
    for root, dirs, files in os.walk(path):
        for file in files:
            filepath = os.path.join(root, file)
            st = os.lstat(filepath)
            if stat.S_ISREG(st.st_mode):
                tot += st.st_size
                num += 1
                
                if oldest_time is None or st.st_atime < oldest_time:
                    oldest = filepath
                    oldest_time = st.st_atime
    return tot, num, oldest
    
def clean_cache():
    for path in (local_upload_dir, local_cache_dir):
        while True:
            size, num, oldest = dir_size_num_oldest(path)
            if size > max_local_dir_size and num > max_local_dir_num_files:
                logger.info(f'Removing oldest file {oldest}')
                os.remove(oldest)
            else:
                break
    
#########################threading#####################

class StopException(Exception):
    pass
    
class CancelJob(Exception):
    pass
    
@dataclass(frozen=True, eq=False)
class Job:
    tid : int = None
    title : str = None
    path : str = None
    url : str = None
    data : bytes = None
    info : dict = None
    keep : str = 'nothing'
    model_type : str = ''
    no_cache : bool = False
    lang_hint : str = None
    blank_video : bool = False
    arg : dict = None
    
    #can change
    progress : float = 0.0
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
                id, title, info = youtube_info(self.url)
            elif self.path:
                title = os.path.splitext(os.path.basename(self.path))[0]
                info = dict(size=os.path.getsize(self.path))
            elif self.data:
                title = digest(content=self.data)
                info = dict(size=len(self.data))
                
            if getattr(info, 'duration', 0) > max_job_duration or getattr(info, 'size', 0) > max_job_filesize:
                raise ValueError('Job too big')
            
            object.__setattr__(self, 'title', title)
            object.__setattr__(self, 'info', info)
            
    def __eq__(self, other):
        return self.tid == getattr(other, 'tid', None)
        
    def update_status_locked(self, status, out_path=None):
        object.__setattr__(self, 'status', status)
        if out_path:
            object.__setattr__(self, 'out_path', out_path)
            
    def update_progress_locked(self, progress):
        object.__setattr__(self, 'progress', progress)


def raise_exception_in_thread(thread, e):
    ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.ident, ctypes.py_object(e))
    
lock = None
event = None
worker_thread = None
should_stop = False
job_queue = None
job_status_cb = None
current_job = None

YTD_CHECK_INTERVAL = 3600 #once per hour
ytd_last_check = datetime.datetime.now()

def die():
    os.kill(os.getpid(), signal.SIGINT)
    sys.exit(0)
    
def youtube_downloader_check():
    global ytd_last_check
    
    try:
        available_version_data = requests.get('https://github.com/yt-dlp/yt-dlp/raw/refs/heads/master/yt_dlp/version.py').text
        lc = {}
        exec(available_version_data, None, lc)
        
        available_version = lc['__version__']
        current_version = yt_dlp.version.__version__
   
        logger.info(f'Checking ytd version: {current_version} vs {available_version}')
        
        if current_version != available_version:
            die()
            
    except Exception as e:
        logger.info(f"Couldn't complete version check: {e}")
    
    ytd_last_check = datetime.datetime.now()
    
def youtube_downloader_next_check():
    global ytd_last_check
    return max(0, ((ytd_last_check + datetime.timedelta(seconds=YTD_CHECK_INTERVAL)) - datetime.datetime.now()).total_seconds())

def work_loop():
    global should_stop, job_queue, lock, event, job_status_cb, current_job
    
    def progress_cb(progress):
        with lock:
            job.update_progress_locked(progress)
        job_status_cb(job, None)
    
    def youtube_progress_cb(progress):
        progress_cb(progress * YOUTUBE_DOWNLOAD_PROGRESS)
    
    def process_cb(progress):
        progress_cb(YOUTUBE_DOWNLOAD_PROGRESS + (progress * (1 - YOUTUBE_DOWNLOAD_PROGRESS)))
        
    try:
        while not should_stop:
            job = None
            current_job = None
            clean_cache()
            with lock:
                event.clear()
                if job_queue:
                    job = job_queue[0]
                    job_queue = job_queue[1:]
                    if job.status != 'idle':
                        continue
                    job.update_status_locked('processing')
            
            if job is None:
                next_ytd_check = youtube_downloader_next_check()
                if next_ytd_check < 1:
                    youtube_downloader_check()
                    
                event.wait(timeout=next_ytd_check)
                continue
                
            try:
                current_job = job
                
                job_status_cb(job, None)
                
                output = None
                
                #work
                
                if job.url is not None:
                    download_path, title = youtube_download(job.url, local_upload_dir, audio_only=(job.keep == 'nothing' and job.blank_video), dont_cache=job.no_cache, progress_cb=youtube_progress_cb)
                    path = canonify_input_file(content=open(download_path, 'rb').read()) #don't delete youtube video file
                elif job.path is not None:
                    path = canonify_input_file(path=job.path)
                else:
                    path = canonify_input_file(content=job.data)
                    
                is_video, is_audio = is_video_audio(path)
                if not is_video and not is_audio:
                    raise ValueError('Input is not a video or an audio file')
                    
                #override if only audio
                blank_video = job.blank_video if is_video else True
                
                set_model_framework(job.model_type or default_model_type)
                
                actions = {'nothing': (make_lyrics_video, ['keep', 'lang_hint', 'blank_video']), 'video': (remove_vocals_from_video, ['keep']), 'all': (passthrough, ['keep'])}
                func, available_selectors = actions[job.keep]
                selectors = {k:v for k,v in dict(keep=job.keep, lang_hint=job.lang_hint, blank_video=blank_video).items() if k in available_selectors}
                
                output = generate_with_cache(func, path, selectors=selectors, dont_cache=job.no_cache, lang_hint=job.lang_hint, blank_video=blank_video, progress_cb=process_cb)
                
                #TODO remove canon input file?
                
                current_job = None
                
                with lock:
                    job.update_status_locked('done', output)
                    
                job_status_cb(job, None)
            except Exception as e:
                if isinstance(e, (StopException, KeyboardInterrupt)):
                    raise
                    
                with lock:
                    job.update_status_locked('canceled' if isinstance(e, CancelJob) else 'error')

                job_status_cb(job, e)
            finally:
                current_job = None
    except StopException:
        pass


def init_thread(status_cb):
    global worker_thread, job_queue, should_stop, lock, event, job_status_cb
    
    os.makedirs(local_upload_dir, exist_ok=True)
    os.makedirs(local_cache_dir, exist_ok=True)
    
    stop_thread()
        
    should_stop = False
    lock = threading.Lock()
    event = threading.Event()
    job_queue = tuple()
    job_status_cb = status_cb
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
        
def cancel_job(job):
    with lock:
        if worker_thread is not None and current_job is not None and current_job == job:
            raise_exception_in_thread(worker_thread, CancelJob)

def thread_test():
    def cb(job, error):
        if error:
            logger.info(f'{job.tid} error: {traceback.format_exc(error)}')
        elif job.status == 'processing':
            logger.info(f'progress: {100*job.progress:.2f}%')
        elif job.status == 'done':
            logger.info(f'{job.tid} is available at {job.out_path} ({job.status})')
            
    init_thread(cb)
    #job4 = Job(path=r'C:\projects\pick\LyricsProj\songs\allstar.wav')
    job1 = Job(url='https://www.youtube.com/watch?v=L_jWHffIx5E')
    job2 = Job(url='https://www.youtube.com/watch?v=L_jWHffIx5E', keep='video')
    job3 = Job(url='https://www.youtube.com/watch?v=L_jWHffIx5E', keep='all')
    jobs = [job1, job2, job3]

    set_queue(jobs)

    try:
        while True:
            logger.info(f'{job1.status}, {job2.status}, {job3.status}')
            time.sleep(1)
    finally:
        stop_thread()
        
#####################################################################################

def main(argv):
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    
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
        '-p', '--passthrough',
        action='store_true',
        default=False,
        help='Just reencode video'
    )
    parser.add_argument(
        '-t', '--model-type',
        type=str,
        default=default_model_type,
        help='change between faster whisper (faster) & whisperX (whisperx)'
    )
    parser.add_argument(
        '-d', '--dont-use-cache',
        action='store_true',
        default=False,
        help='Dont use local video if already generated'
    )
    parser.add_argument(
        '-l', '--lang-hint',
        type=str,
        default=None,
        help='Language hint'
    )
    parser.add_argument(
        '-b', '--blank-video',
        action='store_true',
        default=False,
        help='Make a black subtitle video'
    )
    
    args = parser.parse_args()
    
    os.makedirs(local_upload_dir, exist_ok=True)
    os.makedirs(local_cache_dir, exist_ok=True)
    
    set_model_framework(args.model_type)

    input = args.input
    
    def progress_cb(progress):
        logger.info(f'progress: {100*progress:.2f}%')
    
    def youtube_progress_cb(progress):
        progress_cb(progress * YOUTUBE_DOWNLOAD_PROGRESS)
    
    def process_cb(progress):
        progress_cb(YOUTUBE_DOWNLOAD_PROGRESS + (progress * (1 - YOUTUBE_DOWNLOAD_PROGRESS)))
        
    if args.passthrough:
        keep = 'all'
    elif args.keep_video:
        keep = 'video'
    else:
        keep = 'nothing'

    if not os.path.isfile(input):  # YT Link
        input, title = youtube_download(input, local_upload_dir, audio_only=(keep == 'nothing' and args.blank_video), dont_cache=args.dont_use_cache, progress_cb=youtube_progress_cb)
    
    canon_input = canonify_input_file(content=open(input, 'rb').read()) #dont move input file
    
    is_video, is_audio = is_video_audio(canon_input)
    if not is_video and not is_audio:
        raise ValueError('Input is not a video or an audio file')
        
    #override if only audio
    blank_video = args.blank_video if is_video else True
    
    print(input, canon_input, is_video, is_audio)
    
    actions = {'nothing': (make_lyrics_video, ['keep', 'lang_hint', 'blank_video']), 'video': (remove_vocals_from_video, ['keep']), 'all': (passthrough, ['keep'])}
    func, available_selectors = actions[keep]
    selectors = {k:v for k,v in dict(keep=keep, lang_hint=args.lang_hint, blank_video=blank_video).items() if k in available_selectors}
                
    generate_with_cache(func, canon_input, selectors=selectors, dont_cache=args.dont_use_cache, progress_cb=process_cb, lang_hint=args.lang_hint, blank_video=blank_video)

if __name__ == '__main__':
    main(sys.argv)
    #thread_test()

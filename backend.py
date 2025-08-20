import time, itertools, sys
import os, hashlib, base64, threading, queue
import shutil, ctypes, traceback, inspect
import stat, logging, signal, argparse, gc, datetime
from dataclasses import dataclass
from collections import namedtuple
from pathlib import Path

import requests
import torch
import yt_dlp
import whisperx
import faster_whisper

import instrumental
import ass_utils
import video_utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

has_juice = torch.cuda.device_count() > 0
device_str = 'cuda' if has_juice else 'cpu'
compute_type_str = 'int8' if has_juice else 'int8'

detection_model_name = 'large-v3' if has_juice else 'tiny'

Word = namedtuple('Word', ['word', 'start', 'end'])

local_upload_dir = 'uploads/'
local_cache_dir = 'songs/'
max_job_history = 1000
max_job_duration = 20 * 60
max_job_filesize = 100 * 1024 * 1024
default_model_type = 'faster'

max_local_dir_size = 1024 ** 3
max_local_dir_num_files = 10
unimportant_keep_seconds = 3600 * 24 * 1

DOWNLOAD_FILE_USER_AGENT = 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:15.0) Gecko/20100101 Firefox/15.0.1'

YOUTUBE_DOWNLOAD_PROGRESS = 0.1 #10% download, 90% everything else

DETECT_LANGUAGE_PROGRESS = 0.1
LOAD_TRANSCRIBE_MODEL_PROGRESS = 0.1 #10% detection, 10% load, 80% process

LYRICS_VIDEO_SEPARATOR_PROGRESS = 0.5
LYRICS_VIDEO_TRANSCRIBE_PROGRESS = 0.3
LYRICS_VIDEO_ALIGN_PROGRESS = 0.05  #50% separator, 30% transcribe, 5% align, 15% ffmpeg

REMOVE_VOCALS_SEPARATOR_PROGRESS = 5/7

default_model_name = 'large-v3' if has_juice else 'tiny'
heb_model_name = 'ivrit-ai/whisper-large-v3-ct2' if has_juice else 'tiny'

whisper_model_frameworks = {
                    'faster':
                    {
                        None: (default_model_name, 'faster', {}, dict(vad_filter=False)),
                        'he': (heb_model_name, 'faster', {}, dict(patience=2, beam_size=5, vad_filter=False)),
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

def getitem(l, i, default=None):
    try:
        return l[i]
    except IndexError:
        return default

def download_file(url, path, timeout=None):
    resp = requests.get(url, headers={'User-Agent': DOWNLOAD_FILE_USER_AGENT}, stream=True, timeout=timeout)
    with open(path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    
def youtube_info(url, audio_only=True):
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
        
        thumbnail_hq = max(info['thumbnails'], key=lambda t: t['preference'])['url']
    except ValueError:
        thumbnail = info['thumbnails'][0]['url']
        thumbnail_hq = thumbnail

    return info['id'], info['title'], dict(duration=info['duration'], thumbnail=thumbnail, thumbnail_hq=thumbnail_hq), f'{info["id"]}.{"mp3" if audio_only else "mp4"}'

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
        ydl_opts['video_utils.extract_audio'] = True
        ydl_opts['format'] = 'bestaudio'

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=dont_cache)
        outfile = os.path.join(local_dir, f'{info["id"]}.{ext}')
        if not os.path.exists(outfile) and not dont_cache:
            ydl.extract_info(url, download=True)
            
    if progress_cb:
        progress_cb(1.0)
    return outfile

def do_segmentation(result):
    if not result:
        return []
    result = [segment for segment in result if len(segment) > 0]
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

def make_title_video(bg_path, width, height, video_timebase, audio_timebase, title, subtitle, output_path, remove_intermediates=True, timeout=None):
    ass, ass_duration = ass_utils.make_ass_title(title, subtitle)
    ass_path = video_utils.replace_ext(output_path, '_title.ass')
    
    end_wait = 1
    try:
        with open(ass_path, 'w', encoding='utf8') as fh:
            fh.write(ass)
        video_utils.bg_with_subtitles(bg_path, width, height, video_timebase, audio_timebase, ass_duration + end_wait, ass_path, output_path, timeout=timeout)
    finally:
        if remove_intermediates:
            video_utils.try_remove(ass_path)
    
def make_lyrics_video(inputpath, outputpath, transcribe_using_vocals=True, transcribe_with_backup_vocals=True, backup_vocals_in_inst=True, remove_intermediates=True, progress_cb=None, lang_hint=None, blank_video=False, original_audio=False, **_):
    global whisper_model_framework
    
    if not transcribe_using_vocals and transcribe_with_backup_vocals:
        raise ValueError('transcribe_with_backup_vocals implies transcribe_using_vocals')
    
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
        audiopath = video_utils.replace_ext(inputpath, '_audio.wav')
        
        asspath = video_utils.replace_ext(inputpath, '.ass')
        instpath = video_utils.replace_ext(inputpath, '_inst.mp3')
        vocalspath = video_utils.replace_ext(inputpath, '_vocals.mp3') if transcribe_using_vocals else None
        
        video_utils.extract_audio(inputpath, audiopath)
        
        silence = 0 # TODO video title
        silence_marks = instrumental.instrumental(audiopath,
                                                    instpath,
                                                    output_vocals=vocalspath,
                                                    output_inst_with_backup=backup_vocals_in_inst,
                                                    output_vocals_with_backup=transcribe_with_backup_vocals,
                                                    start_silence=silence,
                                                    end_silence=silence,
                                                    progress_cb=instrumental_progress_cb)

        if transcribe_using_vocals:
            #vocals already have silence accounted for
            #use vocals only
            transcribepath = vocalspath
        else:
            transcribepath = audiopath
            
        result = transcribe_audio(transcribepath, progress_cb=transcribe_progress_cb, lang_hint=lang_hint)
        segments = align_audio(result, progress_cb=align_progress_cb)
        segments = do_segmentation(segments)
        
        silence_until = lambda t: sum(d for s,d in silence_marks if s <= t)
        
        #fix with silence_marks
        for segment in segments:
            for line in segment:
                for word in line:
                    #use word.start silence for both to avoid having a word over a silence
                    s = silence_until(word.start)
                    word.start, word.end = s + word.start, s + word.end
        
        assdata = ass_utils.make_ass_swap(segments)
        
        with open(asspath, 'w', encoding='utf8') as fh:
            fh.write(assdata)
        
        output_audio_path = audiopath if original_audio else instpath

        if blank_video:
            video_utils.audio_with_blank(output_audio_path, outputpath, asspath)
        else:
            video_utils.video_with_audio_and_subtitles(inputpath, output_audio_path, outputpath, subtitles_path=asspath)

    finally:
        if remove_intermediates:
            video_utils.try_remove(audiopath)
            video_utils.try_remove(asspath)
            video_utils.try_remove(instpath)
            if vocalspath:
                video_utils.try_remove(vocalspath)
    
    if progress_cb:
        progress_cb(1.0)

def remove_vocals_from_video(mp4_input, output_path, remove_intermediates=True, progress_cb=None, blank_video=False, **_):
    if progress_cb:
        progress_cb(0.0)
        
    def instrumental_progress_cb(progress):
        if progress_cb:
            progress_cb(progress * REMOVE_VOCALS_SEPARATOR_PROGRESS)
        
    try:
        audiopath = video_utils.replace_ext(mp4_input, '_audio.wav')
        instpath = video_utils.replace_ext(mp4_input, '_inst.mp3')
        
        video_utils.extract_audio(mp4_input, audiopath)
        
        instrumental.instrumental(audiopath, instpath, output_inst_with_backup=True, progress_cb=instrumental_progress_cb)
        
        if blank_video:
            video_utils.audio_with_blank(instpath, output_path)
        else:
            video_utils.video_with_audio(mp4_input, instpath, output_path)
            
    finally:
        if remove_intermediates:
            video_utils.try_remove(instpath)
            video_utils.try_remove(audiopath)
        
    if progress_cb:
        progress_cb(1.0)
        
def passthrough(input, output, progress_cb=None, blank_video=False, **_):
    if progress_cb:
        progress_cb(0.0)
        
    if blank_video:
        video_utils.audio_with_blank(input, output)
    else:
        video_utils.reencode_video(input, output)
    
    if progress_cb:
        progress_cb(1.0)
        
def digest(path=None, content=None):
    return base64.b64encode(hashlib.sha256(open(path, 'rb').read() if path else content).digest()[:15], altchars=b'+-').decode()

def selectors_join(selectors):
    return '_'.join(f'{str(k)}={str(v)}' for k,v in selectors.items())

def cache_path(local_path, selectors):
    name = video_utils.replace_ext(os.path.basename(local_path), '')
    return os.path.join(local_cache_dir, f'{selectors_join(selectors)}_{name}.mp4')
    
def generate_with_cache(f, local_path, selectors, dont_cache=False, **kwargs):
    out_path = cache_path(local_path, selectors)
    
    if dont_cache or not os.path.exists(out_path):
        f(local_path, out_path, **kwargs)
        
    return out_path
    
def canonify_input_file(content):
    h = digest(content=content)

    canon = os.path.join(local_cache_dir, f'{h}.input')
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
    
def clean_unimportant():
    threshold = time.time() - unimportant_keep_seconds
    for path in (local_upload_dir, local_cache_dir):
        for root, dirs, files in os.walk(path):
            for file in files:
                if not os.path.splitext(file)[0].endswith('_'):
                    #important
                    continue
                    
                filepath = os.path.join(root, file)
                st = os.lstat(filepath)
                
                if stat.S_ISREG(st.st_mode) and st.st_atime < threshold:
                    logger.info(f'Removing unimportant file {filepath}')
                    os.remove(filepath)
    
def clean_cache():
    clean_unimportant()
    for path in (local_upload_dir, local_cache_dir):
        while True:
            size, num, oldest = dir_size_num_oldest(path)
            if size > max_local_dir_size and num > max_local_dir_num_files:
                logger.info(f'Removing oldest file {oldest}')
                os.remove(oldest)
            else:
                break
    
#########################threading#####################

actions = {'nothing': (make_lyrics_video, ['keep', 'lang_hint', 'blank_video']), 'video': (remove_vocals_from_video, ['keep']), 'all': (passthrough, ['keep'])}

class StopException(Exception):
    pass
    
class CancelJob(Exception):
    pass
    
statuses = ['idle', 'processing', 'canceled', 'done', 'error']
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
    uploader : str = ''
    arg : dict = None
    
    #can change
    progress : float = 0.0
    status : str = 'idle'
    out_path : str = None
    error : BaseException = None
    
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
            
        
        if self.url:
            id, title, info, download_path = youtube_info(self.url, audio_only=(self.keep == 'nothing' and self.blank_video))
            canon_path = canonify_input_file(download_path.encode('latin-1')) #don't delete youtube video file afterwards, but also just use path for hashing
        elif self.path:
            title = os.path.splitext(os.path.basename(self.path))[0]
            info = dict(size=os.path.getsize(self.path))
            canon_path = canonify_input_file(open(self.path, 'rb').read())
        elif self.data:
            title = digest(content=self.data)[:8]
            info = dict(size=len(self.data))
            canon_path = canonify_input_file(job.data)
            
        if getattr(info, 'duration', 0) > max_job_duration or getattr(info, 'size', 0) > max_job_filesize:
            raise ValueError('Job too big')
        
        if self.title is None:
            object.__setattr__(self, 'title', title)
        object.__setattr__(self, 'info', info)
        object.__setattr__(self, 'canon_path', canon_path)
            
    def __eq__(self, other):
        return self.tid == getattr(other, 'tid', None)
        
    def update_status_locked(self, status, out_path=None):
        if status not in statuses:
            raise ValueError(f'status change to {status} is not allowed')
        object.__setattr__(self, 'status', status)
        if out_path:
            object.__setattr__(self, 'out_path', out_path)
            
    def update_error_locked(self, e):
        object.__setattr__(self, 'error', e)
            
    def update_progress_locked(self, progress):
        object.__setattr__(self, 'progress', progress)
        
    def action(self):
        if self.url:
            # we don't have a file to process
            is_video, is_audio = not (self.keep == 'nothing' and self.blank_video), True
        else:
            if os.path.exists(self.canon_path):
                path = self.canon_path
            elif os.path.exists(self.path):
                path = self.path
            else:
                path = None
                is_video, is_audio = True, True
            if path:
                is_video, is_audio = video_utils.is_video_audio(path)
            if not is_video and not is_audio:
                raise ValueError('Input is not a video or an audio file')
            
        #override if only audio
        blank_video = self.blank_video if is_video else True
        
        func, available_selectors = actions[self.keep]
        selectors = {k:v for k,v in dict(keep=self.keep, lang_hint=self.lang_hint, blank_video=blank_video).items() if k in available_selectors}
        
        return func, selectors, blank_video
        
    def cached_path(self):
        _, selectors, _ = self.action()
        return cache_path(self.canon_path, selectors)


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
        job_status_cb(job)
    
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
                job_status_cb(job)
                
                output = None
                if job.url is not None:
                    download_path = youtube_download(job.url, local_upload_dir, audio_only=(job.keep == 'nothing' and job.blank_video), dont_cache=job.no_cache, progress_cb=youtube_progress_cb)
                    shutil.copy(download_path, job.canon_path)
                elif job.path is not None:
                    if job.path != job.canon_path:
                        shutil.move(job.path, job.canon_path)
                else:
                    with open(job.canon_path, 'wb') as fh:
                        fh.write(job.data)
                
                set_model_framework(job.model_type or default_model_type)
                func, selectors, blank_video = job.action()
                output = generate_with_cache(func, job.canon_path, selectors=selectors, dont_cache=job.no_cache, lang_hint=job.lang_hint, blank_video=blank_video, progress_cb=process_cb)
                
                #TODO remove canon input file?
                
                current_job = None
                with lock:
                    job.update_status_locked('done', output)
                    
                job_status_cb(job)
            except Exception as e:
                if isinstance(e, (StopException, KeyboardInterrupt)):
                    raise
                    
                with lock:
                    job.update_status_locked('canceled' if isinstance(e, CancelJob) else 'error')
                    job.update_error_locked(e)

                job_status_cb(job)
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
    
    jobs = tuple(j for j in jobs)
    
    #finish cached immediately
    for j in jobs:
        outpath = j.cached_path()
        if not j.no_cache and os.path.exists(outpath):
            with lock:
                j.update_status_locked('done', outpath)
            job_status_cb(j)
    
    with lock:
        job_queue = jobs
        event.set()
        
def cancel_job(job):
    with lock:
        if worker_thread is not None and current_job is not None and current_job == job:
            raise_exception_in_thread(worker_thread, CancelJob)

def thread_test():
    def cb(job):
        if job.status == 'error':
            logger.info(f'{job.tid} error: {job.error} {"".join(traceback.format_exception(job.error))}')
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
        
########################title thread#############################################################
title_thread = None
title_should_stop = True
title_thread_queue = None

def title_thread_loop():
    global title_should_stop, title_thread_queue
    
    while not title_should_stop:
        try:
            f = title_thread_queue.get(timeout=1)
            try:
                f()
            except BaseException as e:
                if not isinstance(e, StopException):
                    print('error in task', e)
        except queue.Empty:
            pass
        except StopException:
            break
    
def title_thread_init():
    global title_thread, title_should_stop, title_thread_queue
    
    title_thread_stop()
    
    title_should_stop = False
    title_thread_queue = queue.SimpleQueue()
    title_thread = threading.Thread(target=title_thread_loop)
    title_thread.start()
    
def title_thread_stop():
    global title_thread, title_should_stop, title_thread_queue
    if title_thread is not None:
        title_should_stop = True
        raise_exception_in_thread(title_thread, StopException)
        title_thread.join()
        title_thread = None
        title_thread_queue = None
        
def title_thread_put(f):
    global title_thread_queue
    title_thread_queue.put(f)
    
def add_title_async(cb, job, show_uploader, timeout=None):
    title_thread_put(lambda: add_title_work(cb, job, show_uploader, timeout=timeout))
    
def add_title_work(cb, job, show_uploader, remove_intermediates=True, timeout=None):
    bg = job.info.get('thumbnail_hq')
    title = job.title
    subtitle = job.uploader if show_uploader else None
    
    uniq = base64.b64encode(os.urandom(3), altchars=b'+-').decode()
    
    title_out_path = video_utils.replace_ext(job.out_path, f'_{uniq}_title.mp4')
    
    output_path = video_utils.replace_ext(job.out_path, f'_{uniq}_.mp4')
    
    try:
        if bg:
            thumbnail_out_path = video_utils.replace_ext(job.out_path, '_thumb.png')
            download_file(bg, thumbnail_out_path, timeout=timeout)
        else:
            thumbnail_out_path = None
        
        w, h = video_utils.video_resolution(job.out_path)
        video_timebase, audio_timebase = video_utils.find_time_rates(job.out_path)
        make_title_video(thumbnail_out_path, w, h, video_timebase, audio_timebase, title, subtitle, title_out_path, remove_intermediates=remove_intermediates, timeout=timeout)
        video_utils.video_concat([title_out_path, job.out_path], output_path, timeout=timeout)
        
        with lock:
            job.update_status_locked('done', output_path)
        cb(job)
    except BaseException as e:
        with lock:
            job.update_status_locked('error')
            job.update_error_locked(e)
        cb(job)
    finally:
        if remove_intermediates:
            if thumbnail_out_path:
                video_utils.try_remove(thumbnail_out_path)
            video_utils.try_remove(title_out_path)
        
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
    parser.add_argument(
        '-a', '--keep-audio',
        action='store_true',
        default=False,
        help='Keep the audio unprocessed. (for debugging)'
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
        input = youtube_download(input, local_upload_dir, audio_only=(keep == 'nothing' and args.blank_video), dont_cache=args.dont_use_cache, progress_cb=youtube_progress_cb)
    
    inputdata = open(input, 'rb').read()
    canon_input = canonify_input_file(inputdata)
    
    if input != canon_input:
        with open(canon_input, 'wb') as fh:
            fh.write(inputdata)
    
    is_video, is_audio = video_utils.is_video_audio(canon_input)
    if not is_video and not is_audio:
        raise ValueError('Input is not a video or an audio file')
        
    #override if only audio
    blank_video = args.blank_video if is_video else True
    
    print(input, canon_input, is_video, is_audio)
    
    actions = {'nothing': (make_lyrics_video, ['keep', 'lang_hint', 'blank_video']), 'video': (remove_vocals_from_video, ['keep']), 'all': (passthrough, ['keep'])}
    func, available_selectors = actions[keep]
    selectors = {k:v for k,v in dict(keep=keep, lang_hint=args.lang_hint, blank_video=blank_video).items() if k in available_selectors}
                
    output = generate_with_cache(func, canon_input, selectors=selectors, dont_cache=args.dont_use_cache, progress_cb=process_cb, lang_hint=args.lang_hint, blank_video=blank_video, original_audio=args.keep_audio)
    print(output)
    
    
if __name__ == '__main__':
    main(sys.argv)
    #thread_test()

import os, subprocess, logging, shutil

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ffmpeg_path = os.path.join(os.path.dirname(__file__), 'ffmpeg')
if not os.path.exists(ffmpeg_path):
    logger.warning('Falling back to system ffmpeg')
    ffmpeg_path = shutil.which('ffmpeg')
    
def replace_ext(path, ext):
    if '.' not in path:
        return f'{path}{ext}'
    return f'{path[:path.rfind(".")]}{ext}'
    
def escape_path(path):
    bad_chars = '''\\:'"_='''
    for c in bad_chars:
        path = path.replace(c, f'\\{c}')
    return path

def try_remove(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
        
def run_process(*args, timeout=None):
    process = subprocess.run(args, capture_output=True, timeout=timeout)
    if process.returncode != 0:
        raise RuntimeError(f'process failed: {process.args}\n\n{process.stderr.decode("utf8", errors="ignore")}')
    return process.stdout.decode("utf8", errors="ignore")

def extract_audio(input, output_audio):
    run_process(ffmpeg_path, '-y', '-i', input, '-vn', output_audio)
    
def find_time_rates(input):
    output = run_process('ffprobe', '-v', 'error', '-show_entries', 'stream=codec_type,time_base', '-of', 'compact=p=0', input)
    video_time = None
    audio_time = None
    video_prefix = 'codec_type=video|time_base=1/'
    audio_prefix = 'codec_type=audio|time_base=1/'
    for line in output.replace('\r', '').split('\n'):
        if line.startswith(video_prefix):
            video_time = int(line[len(video_prefix):])
        if line.startswith(audio_prefix):
            audio_time = int(line[len(audio_prefix):])
    return video_time, audio_time
    
def is_video_audio(input):
    output_video = run_process('ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', input)
    output_audio = run_process('ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', input)
    return 'video' in output_video, 'audio' in output_audio
    
def video_resolution(input):
    output_res = run_process('ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', input)
    return tuple(int(n) for n in output_res.split('x'))

def video_best_resolution(videopath, h_to_w_ratio=None, min_width=None):
    if h_to_w_ratio is None:
        h_to_w_ratio = 9/16
    if min_width is None:
        min_width = 1080
        
    w, h = video_resolution(videopath)
    
    #scale to min_width
    if w < min_width:
        h = min_width * h / w
        w = min_width
        
    #fit to ratio
    if (h / w) < h_to_w_ratio:
        h = h_to_w_ratio * w
    else:
        w = h / h_to_w_ratio
        
    w, h = round(w), round(h)
        
    #make even
    if h % 2 != 0:
        h += 1
    if w % 2 != 0:
        w += 1
        
    return w, h
    
    
def audio_with_blank(audiopath, outputpath, subtitles_path=None, timeout=None):
    run_process(ffmpeg_path, '-y', '-f', 'lavfi', '-i', 'color=c=black:s=1280x720', '-i', audiopath, '-shortest', *(['-vf', f'ass=\'{escape_path(subtitles_path)}\':fontsdir=fonts:shaping=complex'] if subtitles_path else []), '-vcodec', 'h264', outputpath, timeout=timeout)
    return 1280, 720
    
video_codec_options = ['-vcodec', 'h264_nvenc']
#video_codec_options = ['-vcodec', 'libx264', '-g', '30', '-preset', 'ultrafast', '-tune', 'fastdecode']
    
def video_with_audio(videopath, audiopath, outputpath, h_to_w_ratio=None, min_width=None, timeout=None):
    w, h = video_best_resolution(videopath, h_to_w_ratio, min_width)
        
    run_process(ffmpeg_path, '-y', '-i', videopath, '-i', audiopath, '-c:v', 'copy',
                '-c:a', 'aac', '-strict', 'experimental', '-shortest', '-filter_complex', 
                f'[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1[v];', '-map', '[v]:v', '-map', '1:a', '-movflags','+faststart',
                *video_codec_options, outputpath, timeout=timeout)
    return w, h
    
def video_with_audio_and_subtitles(videopath, audiopath, outputpath, subtitles_path, h_to_w_ratio=None, min_width=None, timeout=None):
    w, h = video_best_resolution(videopath, h_to_w_ratio, min_width)
        
    run_process(ffmpeg_path, '-y', '-i', videopath, '-i', audiopath, '-c:v', 'copy',
                '-c:a', 'aac', '-strict', 'experimental', '-shortest', '-filter_complex',
                f'[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1[v0]; [v0]ass=\'{escape_path(subtitles_path)}\':fontsdir=fonts:shaping=complex[v]',
                '-map', '[v]:v', '-map', '1:a', '-movflags','+faststart', *video_codec_options, outputpath, timeout=timeout)
    return w, h

def reencode_video(videopath, outputpath, timeout=None):
    w, h = video_resolution(videopath)
    run_process(ffmpeg_path, '-y', '-i', videopath, '-c:v', 'copy', *video_codec_options, outputpath, timeout=timeout)
    return w, h
    
def bg_with_subtitles(bg_path, width, height, video_timebase, audio_timebase, duration, subtitles_path, output_path, timeout=None):
    run_process(ffmpeg_path, '-y',
                *(['-loop', '1', '-i', bg_path, '-f', 'lavfi', '-i', f'anullsrc=cl=stereo:r={audio_timebase}'] if bg_path is not None else ['-f', 'lavfi', '-i', f'color=c=black:s={width}x{height}',
                '-f', 'lavfi', '-i', f'anullsrc=cl=stereo:r={audio_timebase}', '-loop', '1']), '-video_track_timescale', str(video_timebase), *video_codec_options,
                '-t', str(duration), '-pix_fmt', 'yuv420p', '-filter_complex',
                f'fps=30[v0]; [v0]scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1[v1]; [v1]ass=\'{escape_path(subtitles_path)}\':fontsdir=fonts:shaping=complex[v]',
                '-map', '[v]:v', '-map', '1:a', output_path, timeout=timeout)

def video_concat(video_paths, output_path, timeout=None):
    list_path = replace_ext(output_path, '_list.txt')
    try:
        with open(list_path, 'w', encoding='utf8') as fh:
            fh.write('\n'.join(f"file '{os.path.abspath(video_path)}'" for video_path in video_paths))
            
        run_process(ffmpeg_path, '-y', '-f', 'concat', '-safe', '0', '-i', list_path, '-c', 'copy', output_path, timeout=timeout)
    finally:
        try_remove(list_path)

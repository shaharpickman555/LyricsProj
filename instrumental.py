import os, sys, itertools, yaml
import time
from pathlib import Path
import torch
import torchaudio
import demucs.api
from mel_band_roformer import MelBandRoformer

LOAD_SEPARATOR_MODEL_PROGRESS = 0.1
SEPARATOR_MODEL_PROGRESS = 0.9 #10% load, 90% process

LOAD_ROFORMER_MODEL_PROGRESS = 0.1
ROFORMER_MODEL_PROGRESS = 0.9

DEMUCS_PROGRESS = 0.2
ROFORMER_PROGRESS = 0.7
SAVE_PROGRESS = 0.1

selfdir = Path(__file__).parent

device = torch.device('cuda')

####demucs####

loaded_separator = None
def load_separator():
    global loaded_separator
    if loaded_separator is not None:
        return loaded_separator
    loaded_separator = demucs.api.Separator(model='htdemucs')
    return loaded_separator
    
def demucs_demix(audioinput, progress_cb=None):
    if progress_cb:
        progress_cb(0.0)
           
    def separator_cb(data):
        if data['state'] != 'start':
            return
        progress = data['segment_offset'] / data['audio_length']
        if progress_cb:
            progress_cb(LOAD_SEPARATOR_MODEL_PROGRESS + (progress * SEPARATOR_MODEL_PROGRESS))
        
    separator = load_separator()
    
    if progress_cb:
        progress_cb(LOAD_SEPARATOR_MODEL_PROGRESS)
    
    separator.update_parameter(callback=separator_cb)
    
    origin, separated = separator.separate_audio_file(audioinput)
    
    vocals = separated['vocals']
    inst = origin - vocals
        
    if progress_cb:
        progress_cb(1.0)
    
    return inst, vocals, separator.samplerate
    
####roformer####

model_dir = selfdir / 'mel_band_roformer'
config_file = model_dir / 'config_karaoke_becruily.yaml'
model_ckpt = model_dir / 'mel_band_roformer_karaoke_becruily.ckpt'

kareoke_roformer_overlap = 1.2 #lower is faster, linearly

kareoke_roformer_model = None
kareoke_roformer_chunk_size = None
kareoke_roformer_instruments = None
kareoke_roformer_sample_rate = None
def load_kareoke_roformer():
    global kareoke_roformer_model, kareoke_roformer_chunk_size, kareoke_roformer_instruments, kareoke_roformer_sample_rate
    if kareoke_roformer_model is None:
        with open(config_file, 'r', encoding='utf8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        kareoke_roformer_chunk_size = config['audio']['chunk_size']
        kareoke_roformer_instruments = config['training']['instruments']
        kareoke_roformer_sample_rate = config['audio']['sample_rate']
        kareoke_roformer_model = MelBandRoformer(**(config['model'])).to(device)
        kareoke_roformer_model.load_state_dict(torch.load(model_ckpt, map_location=device, weights_only=True))
        
    return kareoke_roformer_model, kareoke_roformer_chunk_size, kareoke_roformer_instruments, kareoke_roformer_sample_rate, kareoke_roformer_overlap

def get_windowing_array(window_size, fade_size):
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window.to(device)

@torch.no_grad
def roformer_demix(path, load_func, num_overlap=None, progress_cb=None):
    if progress_cb:
        progress_cb(0.0)
        
    model, chunk_size, instruments, model_samplerate, default_overlap = load_func()
    
    if num_overlap is None:
        num_overlap = default_overlap
    
    if progress_cb:
        progress_cb(LOAD_ROFORMER_MODEL_PROGRESS)
    
    if path is None:
        return [None for _ in instruments]
    
    C = chunk_size
    step = int(C // num_overlap)
    fade_size = C // 10
    border = C - step
    
    mix, orig_samplerate = torchaudio.load(path)
    if orig_samplerate != model_samplerate:
        mix = torchaudio.functional.resample(mix, orig_freq=orig_samplerate, new_freq=model_samplerate)
    
    if mix.dim() not in (1, 2):
        raise ValueError('mix dims should be 1 (mono) or 2 (stereo)')
    
    if mix.dim() == 1:
        #mono to stereo
        mix = mix.unsqueeze(0).repeat(2, 1)

    if mix.shape[-1] > 2 * border and border > 0:
        mix = torch.nn.functional.pad(mix, (border, border), mode='reflect')

    windowing_array = get_windowing_array(C, fade_size)
    
    if mix.device != device:
        mix = mix.to(device)

    with torch.amp.autocast('cuda'):
        req_shape = tuple((len(instruments), *mix.shape))
        result = torch.zeros(req_shape, dtype=torch.float32, device=device)
        counter = torch.zeros(req_shape, dtype=torch.float32, device=device)

        total_length = mix.shape[-1]
        num_chunks = (total_length + step - 1) // step
        
        for i in range(0, total_length, step):
            part = mix[:, i:i + C]
            length = part.shape[-1]
            if length < C:
                if length > C // 2 + 1:
                    part = torch.nn.functional.pad(input=part, pad=(0, C - length), mode='reflect')
                else:
                    part = torch.nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)

            x = model(part.unsqueeze(0))[0]

            if i == 0 or i + C >= total_length:
                window = windowing_array.clone()
                if i == 0:
                    window[:fade_size] = 1
                else:
                    window[-fade_size:] = 1
            else:
                window = windowing_array

            result[..., i:i+length] += x[..., :length] * window[..., :length]
            counter[..., i:i+length] += window[..., :length]
            
            if progress_cb:
                progress_cb(LOAD_ROFORMER_MODEL_PROGRESS + (ROFORMER_MODEL_PROGRESS * i / total_length))

        estimated_sources = (result / counter).nan_to_num(0.0)

    if mix.shape[-1] > 2 * border and border > 0:
        estimated_sources = estimated_sources[..., border:-border]
    
    if progress_cb:
        progress_cb(1.0)
    
    results = estimated_sources.unbind(0)
    # vocals last
    if instruments.index('Vocals') == 0:
        results = results[::-1]
        
    return *results, model_samplerate

########

#data.shape is [n_channels, samples]
def filter_silence(data, samplerate, window=0.2, min_included=1, threshold=1/100):
    min_windows = int(min_included / window)
    
    windowsize = int(samplerate * window)
    
    #rms over all channels
    rms = torch.tensor([w.square().mean() ** 0.5 for w in torch.split(data, windowsize, dim=-1)])
    rms[rms.isnan()] = 0.0
    
    mask = rms >= (max(rms) * threshold)
    
    silences = []
    silence_start = None
    for i, w in enumerate(itertools.chain(mask, [1])):
        if not w:
            if silence_start is None:
                silence_start = i
        else:
            if silence_start is not None:
                if silence_start == 0 and i > min_windows:
                    silences.append((silence_start, i - min_windows))
                elif i == len(mask) and (i - silence_start) > min_windows:
                    silences.append((silence_start + min_windows, i))
                elif (i - silence_start) > (min_windows * 2):
                    silences.append((silence_start + min_windows, i - min_windows))
                silence_start = None
    
    silence_marks = []
    if silences:
        nonsilence = []
        nonsilence_start = 0
        #invert to nonsilence
        for s,e in silences:
            if s != nonsilence_start:
                nonsilence.append((nonsilence_start * windowsize, s * windowsize))
            nonsilence_start = e
        if nonsilence_start * windowsize < data.shape[-1]:
            nonsilence.append((nonsilence_start * windowsize, data.shape[-1]))

        data = torch.cat([data[:, s : e] for s, e in nonsilence], dim=-1)
        silence_off = 0
        
        for s,e in silences:
            silence_marks.append(((s - silence_off) * window, (e - s) * window))
            silence_off += e - s
            
    return data, silence_marks

def instrumental(audioinput, output_inst, output_vocals=None, output_inst_with_backup=False, output_vocals_with_backup=True, start_silence=0, end_silence=0, progress_cb=None):
    if progress_cb:
        progress_cb(0.0)
        
    def demucs_progress(progress):
        if progress_cb:
            progress_cb(progress * DEMUCS_PROGRESS)
    
    def roformer_progress(progress):
        if progress_cb:
            progress_cb(DEMUCS_PROGRESS + (progress * ROFORMER_PROGRESS))
    
    if not output_inst_with_backup or (output_vocals is not None and output_vocals_with_backup):
        print('doing demucs')
        inst_without_backing, vocals_with_backing, d_samplerate = demucs_demix(audioinput, progress_cb=demucs_progress)
    else:
        # not needed
        print('skipping demucs')
        inst_without_backing, vocals_with_backing, d_samplerate = None, None, None
        
    if output_inst_with_backup or (output_vocals is not None and not output_vocals_with_backup):
        print('doing roformer')
        inst_with_backing, vocals_without_backing, r_samplerate = roformer_demix(audioinput, load_func=load_kareoke_roformer, progress_cb=roformer_progress)
    else:
        # not needed
        print('skipping roformer')
        inst_with_backing, vocals_without_backing, r_samplerate = None, None, None
    
    if progress_cb:
        progress_cb(DEMUCS_PROGRESS + ROFORMER_PROGRESS)
        
    if d_samplerate is not None and r_samplerate is not None and d_samplerate != r_samplerate:
        # match sample rates
        inst_with_backing = torchaudio.functional.resample(inst_with_backing, orig_freq=r_samplerate, new_freq=d_samplerate)
        if output_vocals is not None:
            vocals_without_backing = torchaudio.functional.resample(vocals_without_backing, orig_freq=r_samplerate, new_freq=d_samplerate)
        samplerate = d_samplerate
    else:
        samplerate = d_samplerate or r_samplerate
        
    inst = inst_with_backing if output_inst_with_backup else inst_without_backing
    if output_vocals is not None:
        vocals = vocals_with_backing if output_vocals_with_backup else vocals_without_backing
    
    silence1 = torch.zeros([2, start_silence * samplerate])
    silence2 = torch.zeros([2, end_silence * samplerate])

    inst = torch.cat((silence1.to(inst.device), inst, silence2.to(inst.device)), dim=-1)

    if output_vocals:
        vocals = torch.cat((silence1.to(vocals.device), vocals, silence2.to(vocals.device)), dim=-1)
        vocals, silence_marks = filter_silence(vocals, samplerate)
        torchaudio.save(output_vocals, vocals.cpu(), samplerate)
    else:
        silence_marks = [(0, start_silence)]

    torchaudio.save(output_inst, inst.cpu(), samplerate)
    
    if progress_cb:
        progress_cb(1.0)
        
    return silence_marks

def main(argv):
    def prog(progress):
        print(f'progress: {progress}')
        
    load_kareoke_roformer()
    start = time.time()
    vocals, inst, samplerate = roformer_demix(argv[1], load_func=load_kareoke_roformer, progress_cb=prog)
    end = time.time()
    print(end - start)
    
    torchaudio.save(f'{argv[1]}.vocals.mp3', vocals.cpu(), samplerate)
    torchaudio.save(f'{argv[1]}.inst.mp3', inst.cpu(), samplerate)
    
if __name__ == '__main__':
    main(sys.argv)
    
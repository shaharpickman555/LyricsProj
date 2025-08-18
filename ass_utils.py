import collections, unicodedata, math

def dominant_strong_direction(s):
    count = collections.Counter([unicodedata.bidirectional(c) for c in list(s)])
    rtl_count = count['R'] + count['AL'] + count['RLE'] + count['RLI']
    ltr_count = count['L'] + count['LRE'] + count['LRI']
    return 'rtl' if rtl_count > ltr_count else 'ltr'

def ass_time(seconds):
    return f'{int(seconds // 3600)}:{int(seconds % 3600 // 60):02d}:{int(seconds % 60):02d}.{int(seconds * 100 % 100):02d}'

def output_word(word):
    if dominant_strong_direction(word) == 'rtl':
        punc = ['.', ',', ';', ':', '/', '\\', '?', '!', '"', "'", '...']
        word = word.strip()
        
        startswith = max(len(p) if word.startswith(p) else 0 for p in punc)
        endswith = max(len(p) if word.endswith(p) else 0 for p in punc)
        
        word = word[-endswith:] + word[startswith : -endswith] + word[:startswith]
            
        return word
    else:
        return word.strip()

def output_title(title):
    words = title.split(' ')
    direction = dominant_strong_direction(''.join(w for w in words))
    return r'{\q2}' + ' '.join(output_word(w) for w in words), direction
    
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


def make_ass_swap(segments, prepare_time_seconds=5):
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

    if not segments:
        #no text?
        return header + 'Dialogue: 0, 0:00:00.00,0:00:00.01,Default,'

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
    
    max_color_time = 5
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
            ass_lines.append(f'Dialogue: 0, {ass_time(actual_appear_time)}, {ass_time(actual_disappear_time)}, W1, {{\\pos(400, {ystart + line_y_off})}}{{\\fad(1000, 1000)}}{output_line(line, None)}')

            #output colored lines on top
            for l, word in enumerate(line):
                diff = word.end - word.start
                if diff < 0.02:
                    #dont color word if too short
                    continue
                if diff > max_color_time:
                    #cap coloring time
                    if l == 0:
                        #first word, change start
                        word.start = word.end - max_color_time
                    else:
                        #not first, change end
                        word.end = word.start + max_color_time
                ass_lines.append(f'Dialogue: 1, {ass_time(word.start)}, {ass_time(word.end)}, W1, {{\\pos(400, {ystart + line_y_off})}}{output_line(line, word)}')

            #counter
            if len(line) > 1:
                time_since_last_word = (line[0].start - segment[i - 1][-1].end) if i > 0 else (line[0].start - (last_segment_end or 0))
                if time_since_last_word >= prepare_time_seconds:
                    clock_duration = 4

                    ass_lines.extend(ass_circle(2, 200, 100, line[0].start - clock_duration, line[0].start, 0.5))

        last_segment_end = segment[-1][-1].end

    return header + '\n'.join(ass_lines)

def make_ass_title(title, subtitle):
    header = '''[Script Info]
PlayResX: 800
PlayResY: 800
WrapStyle: 1

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Alignment, Encoding, BorderStyle, Outline, Shadow, MarginL, MarginR, MarginV
Style: W1, Assistant, 80, &HFFFFFF, &HFFFFFF, &H000000, &H000000, 5, 0, 0, 15, 1, 30, 30, 30
Style: W2, Assistant, 60, &HFFFFFF, &HFFFFFF, &H000000, &H000000, 5, 0, 0, 15, 1, 30, 30, 30

[Events]
Format: Layer, Start, End, Style, Text
'''
    
    title_limit = 30
    subtitle_limit = 40
    
    if len(title) > title_limit:
        title = title[:title_limit-3] + '...'
    
    if subtitle:
        if len(subtitle) > subtitle_limit:
            subtitle = subtitle[:subtitle_limit-3] + '...'
        
    title_start = 0
    subtitle_start = 0.3
    title_stay = 4
    subtitle_stay = 4
    transition_time = 0.15
    
    title, direction = output_title(title)
    if subtitle:
        subtitle, _ = output_title(subtitle)
    
    if direction == 'rtl':
        start_pos = 1200
        mid_pos = 400
        end_pos = -400
        sign = -1
    else:
        start_pos = -400
        mid_pos = 400
        end_pos = 1200
        sign = 1

    lines = [
        f'Dialogue: 1, {ass_time(title_start)}, {ass_time(title_start + transition_time)}, W1, {{\\move({start_pos}, 350, {mid_pos}, 350, 0, {int(transition_time * 1000)})}}{title}',
        f'Dialogue: 1, {ass_time(title_start + transition_time)}, {ass_time(title_start + transition_time + title_stay)}, W1, {{\\move({mid_pos}, 350, {mid_pos + (sign * 30)}, 350, 0, {int(title_stay * 1000)})}}{title}',
        f'Dialogue: 1, {ass_time(title_start + transition_time + title_stay)}, {ass_time(title_start + (transition_time * 2) + title_stay)}, W1, {{\\move({mid_pos + (sign * 30)}, 350, {end_pos}, 350, 0, {int(transition_time * 1000)})}}{title}',
    ]
    if subtitle:
        lines.extend([
            f'Dialogue: 2, {ass_time(subtitle_start)}, {ass_time(subtitle_start + transition_time)}, W2, {{\\move({start_pos}, 450, {mid_pos}, 450, 0, {int(transition_time * 1000)})}}{subtitle}',
            f'Dialogue: 2, {ass_time(subtitle_start + transition_time)}, {ass_time(subtitle_start + transition_time + subtitle_stay)}, W2, {{\\move({mid_pos}, 450, {mid_pos + (sign * 30)}, 450, 0, {int(subtitle_stay * 1000)})}}{subtitle}',
            f'Dialogue: 2, {ass_time(subtitle_start + transition_time + subtitle_stay)}, {ass_time(subtitle_start + (transition_time * 2) + subtitle_stay)}, W2, {{\\move({mid_pos + (sign * 30)}, 450, {end_pos}, 450, 0, {int(transition_time * 1000)})}}{subtitle}',
        ])
    
    ass = header + '\n'.join(lines)
    return ass, (transition_time * 2) + ((subtitle_start + subtitle_stay) if subtitle else (title_start + title_stay))

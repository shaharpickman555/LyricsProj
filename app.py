from flask import Flask, request, jsonify, render_template
#import whisper
import whisperx
import os

app = Flask(__name__)
# model = whisper.load_model("large")
model = whisperx.load_model("large-v3", device='cuda', compute_type='int8', vad_options={"vad_onset": 0.05, "vad_offset": 0.05})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file:
        filepath = os.path.join("uploads", file.filename)
        file.save(filepath)

        audio = whisperx.load_audio(filepath)
        result = model.transcribe(audio, batch_size=16,print_progress=True)
        # print(result["segments"])  # before alignment
        # f = open('before.txt', 'w')
        # f.write('dict = ' + repr(result["segments"]) + '\n')
        # f.close()
        model_a, metadata = whisperx.load_align_model(language_code=result["language"],device='cpu')
        result = whisperx.align(result["segments"], model_a, metadata, audio, return_char_alignments=False,device='cpu')
        # print(result["segments"])  # after alignment
        # f = open('after.txt', 'w')
        # f.write('dict = ' + repr(result["segments"]) + '\n')
        # f.close()
        # result = model.transcribe(filepath)

        print(result)  # Debug: Log the result from Whisper
        return jsonify(result), 200

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)

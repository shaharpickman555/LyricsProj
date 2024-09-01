from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import os
import json
import random

app = Flask(__name__)
socketio = SocketIO(app)

VIDEO_DIR = 'static/videos'
PLAYLIST_FILE = 'playlist.json'


def load_playlist():
    if os.path.exists(PLAYLIST_FILE):
        with open(PLAYLIST_FILE, 'r') as f:
            playlist = json.load(f)
    else:
        playlist = initialize_playlist()
        save_playlist(playlist)

    return playlist


def save_playlist(playlist):
    with open(PLAYLIST_FILE, 'w') as f:
        json.dump(playlist, f)


def initialize_playlist():
    videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
    if not videos:
        return {'current': None, 'queue': []}

    current_video = random.choice(videos)
    videos.remove(current_video)
    random.shuffle(videos)

    return {
        'current': current_video,
        'queue': videos
    }


@app.route('/')
def index():
    playlist = load_playlist()
    current_video = playlist['current']
    return render_template('index.html', current_video=current_video)


@app.route('/playlist')
def playlist():
    playlist = load_playlist()
    current_video = playlist['current']
    queue = playlist['queue']
    return render_template('playlist.html', current_video=current_video, queue=queue)

@socketio.on('get_playlist')
def handle_get_playlist():
    playlist = load_playlist()
    emit('update_playlist', playlist['queue'])


@socketio.on('reorder_playlist')
def handle_reorder_playlist(data):
    playlist = load_playlist()
    playlist['queue'] = data
    save_playlist(playlist)
    emit('update_playlist', data, broadcast=True)


@socketio.on('next_video')
def handle_next_video():
    playlist = load_playlist()
    if playlist['queue']:
        playlist['current'] = playlist['queue'].pop(0)
        save_playlist(playlist)
        emit('update_video', playlist['current'], broadcast=True)
        emit('update_current_video', playlist['current'], broadcast=True)
        emit('update_playlist', playlist['queue'], broadcast=True)  # Broadcast the updated queue

if __name__ == '__main__':
    socketio.run(app, debug=True)

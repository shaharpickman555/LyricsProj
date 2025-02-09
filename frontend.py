from backend import Job, set_queue, init_thread, stop_thread
import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_socketio import SocketIO, emit
import traceback
from typing import List

UPLOAD_FOLDER = "uploads"
SONGS_FOLDER = "songs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SONGS_FOLDER, exist_ok=True)
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
playlist = []
current_song = None

def get_current_song():
    global current_song
    for job in playlist:
        if job.status == "done":
            if current_song is None or current_song.tid != job.tid:
                current_song = job
                socketio.emit("player_updated", {"current_song": job.out_path})
            return job
    current_song = None
    return None

@app.route("/next_song", methods=["POST"])
def next_song():
    global playlist, current_song
    if current_song in playlist:
        playlist.remove(current_song)
        set_queue(playlist)
    current_song = get_current_song()
    socketio.emit("playlist_updated", serialize_playlist())
    socketio.emit("player_updated", {"current_song": current_song.out_path if current_song else None})
    return "", 204
@app.route("/")
def index():
    return render_template("index.html", playlist=playlist, current_song=get_current_song())

@app.route("/player")
def player():
    current = get_current_song()
    return render_template("player.html", current_song=current)

@app.route("/add_song", methods=["POST"])
def add_song():
    youtube_url = request.form.get("youtube_url", "").strip()
    local_file = request.files.get("local_file")
    keep_val = request.form.get("keep", "nothing")
    if youtube_url:
        job = Job(url=youtube_url, keep=keep_val)
        playlist.append(job)
    elif local_file and local_file.filename:
        save_path = os.path.join(UPLOAD_FOLDER, local_file.filename)
        local_file.save(save_path)
        job = Job(path=save_path, keep=keep_val)
        playlist.append(job)
    else:
        return redirect(url_for("index"))
    set_queue(playlist)
    socketio.emit("playlist_updated", serialize_playlist())
    return redirect(url_for("index"))

@app.route("/songs/<path:filename>")
def serve_song_file(filename):
    return send_from_directory(SONGS_FOLDER, filename)

@socketio.on("connect")
def on_connect():
    emit("playlist_updated", serialize_playlist())
    emit("player_updated", {"current_song": current_song.out_path if current_song else None})


@socketio.on("remove_song")
def handle_remove_song(index):
    global playlist, current_song
    i = int(index)
    if 0 <= i < len(playlist):
        if playlist[i] != current_song:
            del playlist[i]
            set_queue(playlist)
            socketio.emit("playlist_updated", serialize_playlist())

@socketio.on("reorder_playlist")
def handle_reorder_playlist(data):
    global playlist
    old_index, new_index = data["oldIndex"], data["newIndex"]
    if 0 <= old_index < len(playlist) and 0 <= new_index < len(playlist):
        if playlist[old_index] != get_current_song():
            playlist.insert(new_index, playlist.pop(old_index))
            set_queue(playlist)
            socketio.emit("playlist_updated", serialize_playlist())

def serialize_playlist():
    current = get_current_song()
    return [{
        "title": j.title,
        "status": j.status,
        "is_playing": (current is not None and j.tid == current.tid),
        "out_path": getattr(j, "out_path", "")
    } for j in playlist]

def job_status_callback(updated_job):
    socketio.emit("playlist_updated", serialize_playlist())

def cb(job, error):
    job_status_callback(job)
    if error:
        print(f'{job.tid} error: ', traceback.format_exc(error))
    else:
        print(f'{job.tid} is available at {job.out_path} ({job.status})')

init_thread(cb)

if __name__ == "__main__":
    try:
        socketio.run(app, debug=True, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)
    finally:
        stop_thread()

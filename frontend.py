import subprocess
from backend import Job, set_queue, init_thread, stop_thread
import os, sys, argparse
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from flask_socketio import SocketIO, emit, join_room, leave_room
import traceback
import platform
from typing import List

UPLOAD_FOLDER = "uploads"
SONGS_FOLDER = "songs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SONGS_FOLDER, exist_ok=True)

app = Flask(__name__)
socketio = SocketIO(app)
rooms = {}

def get_room_data(room_id):
    if room_id not in rooms:
        rooms[room_id] = {"playlist": [], "current_song": None}
    return rooms[room_id]

def get_current_song(room_id):
    data = get_room_data(room_id)
    playlist = data["playlist"]
    current_song = data["current_song"]
    for job in playlist:
        if job.status == "done":
            if not current_song or current_song.tid != job.tid:
                data["current_song"] = job
                socketio.emit(
                    "player_updated",
                    {
                        "current_song": job.out_path,
                        "in_process": any(j.status == "processing" for j in playlist),
                    },
                    to=room_id
                )
            return job
    data["current_song"] = None
    return None

@app.route("/")
def select_room():
    return render_template("select_room.html")

@app.route("/create_room", methods=["POST"])
def create_room():
    room_id = request.form.get("room_id", "").strip()
    if not room_id:
        return redirect(url_for("select_room"))
    if room_id not in rooms:
        rooms[room_id] = {"playlist": [], "current_song": None}
    return redirect(url_for("index", room_id=room_id))


@app.route("/<room_id>")
def index(room_id):
    data = get_room_data(room_id)
    current = get_current_song(room_id)
    return render_template(
        "index.html",
        playlist=data["playlist"],
        current_song=current,
        room_id=room_id
    )

@app.route("/player/<room_id>")
def player(room_id):
    current = get_current_song(room_id)
    return render_template("player.html", current_song=current, room_id=room_id)

@app.route("/next_song/<room_id>", methods=["POST"])
def next_song(room_id):
    data = get_room_data(room_id)
    playlist = data["playlist"]
    current_song = data["current_song"]
    if current_song and current_song in playlist:
        playlist.remove(current_song)
        set_queue(playlist)
    data["current_song"] = get_current_song(room_id)
    socketio.emit("playlist_updated", serialize_playlist(room_id), to=room_id)
    new_current = data["current_song"]
    socketio.emit(
        "player_updated",
        {
            "current_song": new_current.out_path if new_current else None,
            "in_process": any(j.status == "processing" for j in playlist),
        },
        to=room_id
    )
    return "", 204


@app.route("/add_song/<room_id>", methods=["POST"])
def add_song(room_id):
    data = get_room_data(room_id)
    playlist = data["playlist"]

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
        return redirect(url_for("index", room_id=room_id))
    set_queue(playlist)
    socketio.emit("playlist_updated", serialize_playlist(room_id), to=room_id)
    return redirect(url_for("index", room_id=room_id))

@app.route("/songs/<path:filename>")
def serve_song_file(filename):
    return send_from_directory(SONGS_FOLDER, filename)

@socketio.on("connect")
def on_connect():
    pass

@socketio.on("join_room")
def handle_join(data):
    room_id = data["room_id"]
    join_room(room_id)
    emit("playlist_updated", serialize_playlist(room_id))
    rdata = get_room_data(room_id)
    c = rdata["current_song"]
    emit(
        "player_updated",
        {
            "current_song": c.out_path if c else None,
            "in_process": any(j.status == "processing" for j in rdata["playlist"])
        }
    )

@socketio.on("remove_song")
def handle_remove_song(data):
    room_id = data["room_id"]
    index = int(data["index"])
    rdata = get_room_data(room_id)
    playlist = rdata["playlist"]
    current_song = rdata["current_song"]
    if 0 <= index < len(playlist):
        if playlist[index] != get_current_song(room_id):
            del playlist[index]
            set_queue(playlist)
            socketio.emit("playlist_updated", serialize_playlist(room_id), to=room_id)

@socketio.on("reorder_playlist")
def handle_reorder_playlist(data):
    room_id = data["room_id"]
    old_index, new_index = data["oldIndex"], data["newIndex"]
    rdata = get_room_data(room_id)
    playlist = rdata["playlist"]
    if 0 <= old_index < len(playlist) and 0 <= new_index < len(playlist):
        if playlist[old_index] != get_current_song(room_id):
            playlist.insert(new_index, playlist.pop(old_index))
            set_queue(playlist)
            socketio.emit("playlist_updated", serialize_playlist(room_id), to=room_id)

def serialize_playlist(room_id):
    data = get_room_data(room_id)
    current = get_current_song(room_id)
    result = []
    for j in data["playlist"]:
        result.append({
            "title": j.title,
            "status": j.status,
            "is_playing": (current is not None and j.tid == current.tid),
            "out_path": getattr(j, "out_path", "")
        })
    return result

def job_status_callback(updated_job):
    # This is global, so we need to find which room has this job
    for r_id, r_data in rooms.items():
        if updated_job in r_data["playlist"]:
            socketio.emit("playlist_updated", serialize_playlist(r_id), to=r_id)
            if not r_data["current_song"]:
                socketio.emit(
                    "player_updated",
                    {
                        "current_song": None,
                        "in_process": any(j.status == "processing" for j in r_data["playlist"])
                    },
                    to=r_id
                )
            break

def cb(job, error):
    job_status_callback(job)
    if error:
        print(f'{job.tid} error: ', traceback.format_exc(error))
    else:
        print(f'{job.tid} is available at {job.out_path} ({job.status})')

def create_app():
    init_thread(cb)
    print("Done creating App")
    return app

if __name__ == "__main__":
    import platform
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", help="release mode", action="store_true")
    args = parser.parse_args()
    try:
        if args.release and platform.system() == 'Linux':
            subprocess.run(['gunicorn', '-w', '4', 'frontend:create_app()'])
        else:
            socketio.run(
                create_app(),
                debug=True,
                host="0.0.0.0",
                port=8000,
                allow_unsafe_werkzeug=True,
                use_reloader=False
            )
    finally:
        stop_thread()
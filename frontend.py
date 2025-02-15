import subprocess
from backend import Job, set_queue, init_thread, stop_thread
import os, sys, argparse
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, make_response
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

def get_or_create_room(room_id):
    if room_id not in rooms:
        rooms[room_id] = {
            "playlist": [],
            "current_song": None
        }
    return rooms[room_id]

def update_rooms_list():
    socketio.emit("rooms_list_updated", list(rooms.keys()))

@app.route("/favicon.ico")
def favicon_ignore():
    return make_response("", 204)
@app.route("/")
def select_room():
    print(rooms.keys())
    return render_template("select_room.html")

@app.route("/api/create_room", methods=["POST"])
def api_create_room():
    room_id = request.form.get("room_id", "").strip()
    if not room_id:
        return jsonify({"error": "Room ID cannot be empty"}), 400
    get_or_create_room(room_id)
    update_rooms_list()
    return jsonify({"room_id": room_id}), 200

@app.route("/api/remove_room", methods=["POST"])
def api_remove_room():
    room_id = request.form.get("room_id", "").strip()
    if room_id in rooms:
        del rooms[room_id]
        update_rooms_list()
        return jsonify({"removed": room_id}), 200
    return jsonify({"error": "Room not found"}), 404

@app.route("/<room_id>")
def index(room_id):
    data = get_or_create_room(room_id)
    current = get_current_song(room_id)
    return render_template("index.html", playlist=data["playlist"], current_song=current, room_id=room_id)

@app.route("/player/<room_id>")
def player(room_id):
    current = get_current_song(room_id)
    return render_template("player.html", current_song=current, room_id=room_id)

@app.route("/next_song/<room_id>", methods=["POST"])
def next_song(room_id):
    data = get_or_create_room(room_id)
    playlist = data["playlist"]
    current_song = data["current_song"]
    if current_song and current_song in playlist:
        playlist.remove(current_song)
        set_queue(playlist)
    data["current_song"] = get_current_song(room_id)
    socketio.emit("playlist_updated", serialize_playlist(room_id), to=room_id)
    new_current = data["current_song"]
    socketio.emit("player_updated", {
        "current_song": new_current.out_path if new_current else None,
        "in_process": any(j.status == "processing" for j in playlist)
    }, to=room_id)
    return "", 204

@app.route("/add_song/<room_id>", methods=["POST"])
def add_song(room_id):
    data = get_or_create_room(room_id)
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

def get_current_song(room_id):
    data = get_or_create_room(room_id)
    playlist = data["playlist"]
    current_song = data["current_song"]
    for job in playlist:
        if job.status == "done":
            if not current_song or current_song.tid != job.tid:
                data["current_song"] = job
                socketio.emit("player_updated", {
                    "current_song": job.out_path,
                    "in_process": any(j.status == "processing" for j in playlist)
                }, to=room_id)
            return job
    data["current_song"] = None
    return None

@socketio.on("connect")
def on_connect():
    emit("rooms_list_updated", list(rooms.keys()))

@socketio.on("join_room")
def handle_join_room(data):
    room_id = data.get("room_id")
    if not room_id:
        return
    join_room(room_id)
    emit("playlist_updated", serialize_playlist(room_id))
    current = get_or_create_room(room_id)["current_song"]
    emit("player_updated", {
        "current_song": current.out_path if current else None,
        "in_process": any(j.status == "processing" for j in get_or_create_room(room_id)["playlist"])
    })

@socketio.on("remove_song")
def handle_remove_song(data):
    room_id = data["room_id"]
    index = data["index"]
    rdata = get_or_create_room(room_id)
    playlist = rdata["playlist"]
    i = int(index)
    if 0 <= i < len(playlist):
        if playlist[i] != get_current_song(room_id):
            del playlist[i]
            set_queue(playlist)
            socketio.emit("playlist_updated", serialize_playlist(room_id), to=room_id)

@socketio.on("reorder_playlist")
def handle_reorder_playlist(data):
    room_id = data["room_id"]
    old_index, new_index = data["oldIndex"], data["newIndex"]
    rdata = get_or_create_room(room_id)
    playlist = rdata["playlist"]
    if 0 <= old_index < len(playlist) and 0 <= new_index < len(playlist):
        if playlist[old_index] != get_current_song(room_id):
            playlist.insert(new_index, playlist.pop(old_index))
            set_queue(playlist)
            socketio.emit("playlist_updated", serialize_playlist(room_id), to=room_id)

def serialize_playlist(room_id):
    data = get_or_create_room(room_id)
    current = data["current_song"]
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
    for rid, rdata in rooms.items():
        if updated_job in rdata["playlist"]:
            socketio.emit("playlist_updated", serialize_playlist(rid), to=rid)
            if not rdata["current_song"]:
                csong = get_current_song(rid)
                socketio.emit("player_updated", {
                    "current_song": csong.out_path if csong else None,
                    "in_process": any(j.status == "processing" for j in rdata["playlist"])
                }, to=rid)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", help="release mode",action="store_true")
    args = parser.parse_args()
    try:
        if args.release and platform.system() == 'Linux':
            subprocess.run(['gunicorn','-w','4','frontend:create_app()'])
        else:
            socketio.run(create_app(), debug=True, host="0.0.0.0", port=8000, allow_unsafe_werkzeug=True, use_reloader=False)
    finally:
        stop_thread()

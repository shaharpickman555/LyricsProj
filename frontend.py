import subprocess
import os, sys, argparse, logging
import random, string, time, pprint
import re, traceback
from io import BytesIO

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, make_response, send_file, flash
from flask_socketio import SocketIO, emit, join_room, leave_room
from werkzeug.utils import secure_filename
import qrcode
import yt_dlp

from backend import Job, set_queue, init_thread, stop_thread, set_debug, max_job_filesize, die

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

UPLOAD_FOLDER = "uploads"
SONGS_FOLDER = "songs"
DEFAULT_ROOM_ID_LENGTH = 6
MAX_YT_SEARCH_RESULTS = 6

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SONGS_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'dev'
app.config['MAX_CONTENT_LENGTH'] = max_job_filesize
socketio = SocketIO(app)
rooms = {}

REBOOT_PW_PATH = 'reboot_pw.txt'
VALID_ROOM_REGEX = re.compile(r'^[A-Za-z0-9_-]+$')

ALLOWED_LANGUAGE_HINTS = {
    'ar': 'Arabic', 'en': 'English', 'fr': 'French', 'de': 'German', 'he': 'Hebrew',
    'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'es': 'Spanish'
}

def validate_room_id(room_id: str) -> bool:
    return bool(VALID_ROOM_REGEX.match(room_id))

def get_room(room_id: str):
    return rooms.get(room_id)

def get_validated_room(room_id: str):
    return get_room(room_id) if validate_room_id(room_id) else None

def create_room_if_valid(room_id: str):
    if room_id not in rooms and validate_room_id(room_id):
        rooms[room_id] = {"playlist": [], "current_song": None, "previous_songs": []}

def update_rooms_list():
    socketio.emit("rooms_list_updated", list(rooms.keys()))

def generate_room_id(length=DEFAULT_ROOM_ID_LENGTH):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

@app.get("/landing")
def landing():
    return render_template("landing.html")

@app.route("/")
def auto_create_room():
    room_id = generate_room_id()
    create_room_if_valid(room_id)
    logger.info(f"Auto-created room: {room_id}")
    return redirect(url_for("player", room_id=room_id))

@app.route("/api/create_room", methods=["POST"])
def api_create_room():
    room_id = request.form.get("room_id", "").strip()
    if not room_id:
        return jsonify({"error": "Room ID cannot be empty"}), 400
    if not validate_room_id(room_id):
        return jsonify({"error": "Invalid room name. Use only letters, digits, underscore, dash."}), 400
    create_room_if_valid(room_id)
    update_rooms_list()
    return jsonify({"room_id": room_id}), 200

@app.route("/api/remove_room", methods=["POST"])
def api_remove_room():
    room_id = request.form.get("room_id", "").strip()
    if not validate_room_id(room_id):
        return jsonify({"error": "Invalid room name."}), 400
    if room_id in rooms:
        del rooms[room_id]
        update_rooms_list()
        return jsonify({"removed": room_id}), 200
    return jsonify({"error": "Room not found"}), 404

@app.route("/<room_id>")
def index(room_id):
    data = get_validated_room(room_id)
    if data is None:
        return custom_not_found()
    current = get_current_song(room_id)
    return render_template("index.html", playlist=data["playlist"], current_song=current, room_id=room_id, languages=ALLOWED_LANGUAGE_HINTS)

@app.route("/player/<room_id>")
def player(room_id):
    data = get_validated_room(room_id)
    if data is None:
        return custom_not_found()
    current = get_current_song(room_id)
    return render_template("player.html", current_song=current, room_id=room_id)

@app.route("/next_song/<room_id>", methods=["POST"])
def next_song(room_id):
    data = get_validated_room(room_id)
    if data is None:
        return custom_not_found()
    playlist = data["playlist"]
    current_song = data["current_song"]
    if current_song and current_song in playlist:
        playlist.remove(current_song)
        set_queue(playlist)
        data["previous_songs"].append(current_song)
    data["current_song"] = get_current_song(room_id)
    socketio.emit("playlist_updated", serialize_room(room_id), to=room_id)
    new_current = data["current_song"]
    socketio.emit("player_updated", {
        "current_song": new_current.out_path if new_current else None,
        "in_process": any(j.status == "processing" for j in playlist),
    }, to=room_id)
    return "", 204

@app.route("/add_song/<room_id>", methods=["POST"])
def add_song(room_id):
    data = get_validated_room(room_id)
    if data is None:
        return custom_not_found()
    youtube_url = request.form.get("youtube_url", "").strip().split('&')[0]
    local_file = request.files.get("local_file")
    keep_val = request.form.get("keep", "nothing")
    uploader = request.form.get("uploader", "")
    
    #TODO sanitize uploader
    uploader = re.sub(r'[<>]', '', uploader)
    if not uploader:
        uploader = 'unknown'
    
    job_params = dict(uploader=uploader, keep=keep_val)

    if request.form.get("no_cache"):
        job_params["no_cache"] = True
    if request.form.get("dont_overlay_video"):
        job_params["blank_video"] = True
    lang_hint = request.form.get("lang_hint", "")
    if lang_hint and lang_hint in ALLOWED_LANGUAGE_HINTS:
        job_params["lang_hint"] = lang_hint

    if youtube_url:
        job_params["url"] = youtube_url
    elif local_file and local_file.filename:
        path = os.path.join(UPLOAD_FOLDER, secure_filename(local_file.filename))
        local_file.save(path)
        job_params["path"] = path
    else:
        return make_response('Wrong data', 400)

    try:
        job = Job(**job_params)
    except Exception as e:
        flash(f"Error creating job: {str(e)}", "danger")
        return make_response('Error', 500)

    data["playlist"].append(job)
    set_queue(data["playlist"])
    socketio.emit("playlist_updated", serialize_room(room_id), to=room_id)
    return 'Ok', 200

@app.route("/restore_song/<room_id>", methods=["POST"])
def restore_song(room_id):
    data = get_validated_room(room_id)
    if data is None:
        return custom_not_found()
    index_str = request.form.get("index", "")
    if not index_str.isdigit():
        return jsonify({"error": "Invalid index"}), 400
    index = int(index_str)
    if 0 <= index < len(data["previous_songs"]):
        song = data["previous_songs"].pop(index)
        data["playlist"].append(song)
        set_queue(data["playlist"])
        if not data["current_song"]:
            data["current_song"] = get_current_song(room_id)
    else:
        return jsonify({"error": "Index out of range"}), 400
    socketio.emit("playlist_updated", serialize_room(room_id), to=room_id)
    return jsonify({"restored": True}), 200

@app.route("/songs/<path:filename>")
def serve_song_file(filename):
    return send_from_directory(SONGS_FOLDER, filename)

@app.route("/qr")
def qr_code():
    data = request.args.get("data", "")
    if not data:
        return make_response("No data provided", 400)
    img = qrcode.make(data)
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

@app.route("/qr_inv")
def qr_inv():
    data = request.args.get("data", "")
    if not data:
        return make_response("No data provided", 400)
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=8, border=2)
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="white", back_color="black")
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

@app.route("/search/yt/<int:many>", methods=["GET"])
def search_yt(many):
    many = min(max(many, 1), MAX_YT_SEARCH_RESULTS)
    q = request.args.get("q", "")
    if not q:
        return make_response("", 400)

    ydl_opts = {
        'default_search': f'ytsearch{many}',
        'skip_download': True,
        'force_noplaylist': True,
        'extract_flat': 'in_playlist'
    }

    q = q.replace(":", " ")
    results = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(q)
        for entry in info['entries']:
            results.append(dict(
                url=entry['url'],
                title=entry['title'],
                uploader=entry['uploader'],
                views=entry['view_count'],
                duration=entry['duration'],
                thumbnail=min(entry['thumbnails'], key=lambda t: t['height'])['url']
            ))
    return jsonify(results), 200

@app.route("/singlemode")
def singlemode():
    room_id = generate_room_id()
    return redirect(url_for("singlemode_room", room_id=room_id))

def get_current_song(room_id):
    data = get_room(room_id)
    if not data:
        return None
    playlist = data["playlist"]
    # choose the first complete job that is not canceled or error
    jobs = [job for job in playlist if job.status not in ("canceled", "error")]
    if jobs:
        job = jobs[0]
        if job.status == "done":
            if data["current_song"] != job:
                data["current_song"] = job
                socketio.emit("player_updated", {
                    "current_song": job.out_path,
                    "in_process": any(j.status == "processing" for j in playlist)
                }, to=room_id)
            return job
    data["current_song"] = None
    return None

@app.route("/singlemode/<room_id>")
def singlemode_room(room_id):
    create_room_if_valid(room_id)
    data = get_validated_room(room_id)
    if data is None:
        return custom_not_found()
    current = get_current_song(room_id)
    return render_template("singlemode.html",
                           playlist=data["playlist"],
                           previous_songs=data["previous_songs"],
                           current_song=current,
                           room_id=room_id,
                           languages=ALLOWED_LANGUAGE_HINTS)


def serialize_jobs(jobs, current=None):
    return [{
        "title": j.title,
        "uploader": j.uploader,
        "status": j.status,
        "progress": j.progress,
        "is_playing": (current and j.tid == current.tid),
        "out_path": getattr(j, "out_path", ""),
        "url": getattr(j, "url", ""),
        "info": getattr(j, "info", {}),
        "error": str(j.error) if j.status == "error" else "",
    } for j in jobs]

def serialize_room(room_id):
    rdata = rooms[room_id]
    current = rdata["current_song"]
    return {
        "playlist": serialize_jobs(rdata["playlist"], current),
        "previous_songs": serialize_jobs(rdata["previous_songs"])
    }

@socketio.on("connect")
def on_connect():
    emit("rooms_list_updated", list(rooms.keys()))

@socketio.on("join_room")
def handle_join_room(data):
    room_id = data.get("room_id")
    if not validate_room_id(room_id) or room_id not in rooms:
        return
    join_room(room_id)
    emit("playlist_updated", serialize_room(room_id))
    current = rooms[room_id]["current_song"]
    emit("player_updated", {
        "current_song": current.out_path if current else None,
        "in_process": any(j.status == "processing" for j in rooms[room_id]["playlist"])
    })

@socketio.on("remove_song")
def handle_remove_song(data):
    room_id = data["room_id"]
    i = int(data["index"])
    if not validate_room_id(room_id) or room_id not in rooms:
        return
    playlist = rooms[room_id]["playlist"]
    if 0 <= i < len(playlist) and playlist[i] != rooms[room_id]["current_song"]:
        del playlist[i]
        set_queue(playlist)
        socketio.emit("playlist_updated", serialize_room(room_id), to=room_id)

@socketio.on("reorder_playlist")
def handle_reorder_playlist(data):
    room_id = data["room_id"]
    old_index = data["oldIndex"]
    new_index = data["newIndex"]
    if not validate_room_id(room_id) or room_id not in rooms:
        return
    playlist = rooms[room_id]["playlist"]
    if 0 <= old_index < len(playlist) and 0 <= new_index < len(playlist) and playlist[old_index] != rooms[room_id]["current_song"]:
        playlist.insert(new_index, playlist.pop(old_index))
        set_queue(playlist)
        socketio.emit("playlist_updated", serialize_room(room_id), to=room_id)

@app.route('/reboot', methods=['GET', 'POST'])
def reboot():
    time.sleep(0.2 + random.uniform(0.0, 0.3))
    pw = request.form.get('pw')
    if pw == open(REBOOT_PW_PATH, 'r').read().strip():
        die()
    return make_response('''
    <!doctype html>
    <html><head><meta name="viewport" content="width=device-width, initial-scale=1.0" /></head>
    <body><h4>Request Reboot</h4>
    <form action="" method="post"><input type="text" name="pw" />
    <input type="submit" value="request" /></form></body></html>''')

@app.errorhandler(404)
def handle_404(e):
    return custom_not_found()

def custom_not_found():
    return make_response(
        """<h1>Page not found</h1>
           <p>This page doesn't exist or the URL is incorrect.</p>
           <p>You can <a href='/'>go back to the main page</a> to create a new room.</p>""",
        404,
    )

def create_app():
    init_thread(cb)
    create_room_if_valid("temproom")
    logger.info("Done creating App")
    return app

def job_status_callback(updated_job):
    for rid, rdata in rooms.items():
        if updated_job in rdata["playlist"]:
            socketio.emit("playlist_updated", serialize_room(rid), to=rid)
            if not rdata["current_song"]:
                csong = get_current_song(rid)
                socketio.emit("player_updated", {
                    "current_song": csong.out_path if csong else None,
                    "in_process": any(j.status == "processing" for j in rdata["playlist"]),
                }, to=rid)
            break

def cb(job):
    job_status_callback(job)
    if job.status == 'error':
        logger.info(f'{job.tid} error: {job.error}')
    elif job.status == 'processing':
        logger.info(f'progress: {100*job.progress:.2f}%')
    elif job.status == 'done':
        logger.info(f'{job.tid} is available at {job.out_path} ({job.status})')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", help="release mode", action="store_true")
    args = parser.parse_args()
    set_debug(not args.release)
    try:
        socketio.run(create_app(), debug=True, host="0.0.0.0", port=8000, allow_unsafe_werkzeug=True, use_reloader=False)
    finally:
        stop_thread()

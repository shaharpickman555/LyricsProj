import os
import string

from flask import Flask, render_template, redirect, url_for, request, send_from_directory
from flask_socketio import SocketIO, emit
from backend import Song
import threading
import time, random

def _get_song_by_tid(tid):
    """Return the Song object matching the TID, or None if not found."""
    for s in song_playlist:
        if s.tid == tid:
            return s
    return None

def _get_first_done_song():
    """Return the first Song in the playlist with state='done', or None if none exist."""
    for s in song_playlist:
        if s.state == 'done':
            return s
    return None

def _remove_song_by_tid(tid):
    """Remove and return the Song from the playlist if found, else None."""
    for i, s in enumerate(song_playlist):
        if s.tid == tid:
            return song_playlist.pop(i)
    return None

def _song_to_dict_plus_playing(song):
    """
    Convert the song to a dictionary, plus add 'is_playing' if
    it is the global currently_playing_tid.
    """
    d = song.to_dict()
    d["is_playing"] = (song.tid == currently_playing_tid)
    return d
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
next_tid = 10
currently_playing_tid = None
song_playlist = [Song(tid=1, path='keep=video_89eusrJpdJACDWxhAxzG.mp4', title='Song1',state='done'),
                 Song(tid=2, path='keep=nothing_-7MEND5qWR1AliwI6mIs.mp4', title='Song2', state='done'),
                 Song(tid=3, path='keep=all_89eusrJpdJACDWxhAxzG.mp4', title='Song3', state='done'),
                 Song(tid=4, path='big.mp4', title='Song4', state='done'),
                 Song(tid=5, path='H1.mp4', title='Song5', state='done'),
                 Song(tid=6, path='H11.mp4', title='Song6', state='done'),
                 Song(tid=7, path='H2.mp4', title='Song7', state='done'),
                 Song(tid=8, path='H22.mp4', title='Song8', state='done'),
                 Song(tid=9, path='Q.mp4', title='Song9', state='done')]
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/player", methods=["GET"])
def player():
    global currently_playing_tid

    # If there's no currently playing song or it no longer exists in the playlist, pick a new one
    if not _get_song_by_tid(currently_playing_tid) or currently_playing_tid is None:
        # Try to pick the first "done" song
        done_song = _get_first_done_song()
        if done_song is not None:
            currently_playing_tid = done_song.tid
        else:
            currently_playing_tid = None
        # Find the actual Song object for the currently playing TID
    playing_song = _get_song_by_tid(currently_playing_tid)
    print("########")
    print(playing_song)
    print(playing_song.path)
    print("########")
    return render_template("player.html", song=playing_song)

@app.route("/player/next", methods=["POST"])
def player_next():
    """
    "Next Song" button handler:
    - Remove the currently playing song from the playlist.
    - Find the next available "done" song, set it as currently playing.
    - Redirect back to /player.
    """
    global currently_playing_tid
    playing_song = _get_song_by_tid(currently_playing_tid)

    # 1) Remove the old song if it exists
    if playing_song:
        _remove_song_by_tid(playing_song.tid)

    # 2) Pick the next "done" song, if any
    next_song = _get_first_done_song()
    currently_playing_tid = next_song.tid if next_song else None

    # Notify all connected sockets that the playlist changed
    socketio.emit('update_songs', [_song_to_dict_plus_playing(s) for s in song_playlist])

    return redirect(url_for('player'))

@app.route("/songs/<path:filename>")
def serve_song_file(filename):
    """Serve a file from the songs folder."""
    songs_dir = os.path.join(os.path.dirname(__file__), "songs")
    return send_from_directory(songs_dir, filename)

@socketio.on('connect')
def on_connect():
    print("Client connected.")
    emit('update_songs', [_song_to_dict_plus_playing(s) for s in song_playlist])


@socketio.on('client_add_song')
def on_client_add_song(data):
    global next_tid
    title = data.get("title", "Untitled Song")
    new_tid = next_tid
    next_tid += 1

    # For simplicity, the path can be a placeholder
    new_song = Song(new_tid, f"songs/{new_tid}.mp4", title, 'queue')
    song_playlist.append(new_song)

    print(f"Added new song tid={new_tid}: {title}")
    socketio.emit('update_songs', [_song_to_dict_plus_playing(s) for s in song_playlist])

@socketio.on('client_remove_song')
def on_client_remove_song(data):
    """
    data = { "tid": some_tid }
    Remove the song with the given TID from the playlist,
    and broadcast the updated playlist.
    """
    tid = data.get("tid")
    if tid is None:
        return

    # Find the song by TID and remove it if found
    for i, song in enumerate(song_playlist):
        if song.tid == tid:
            removed = song_playlist.pop(i)
            print(f"Removed song tid={removed.tid}: {removed.title}")
            break

    # Broadcast updates to all clients
    socketio.emit('update_songs', [_song_to_dict_plus_playing(s) for s in song_playlist])


@socketio.on('client_reorder')
def on_client_reorder(data):
    """
    data = { "oldIndex": int, "newIndex": int }
    Reorder the playlist by moving the song from oldIndex to newIndex.
    Then broadcast the updated playlist.
    """
    old_index = data.get("oldIndex")
    new_index = data.get("newIndex")

    if (old_index is not None
            and new_index is not None
            and 0 <= old_index < len(song_playlist)
            and 0 <= new_index < len(song_playlist)):
        # Extract the item
        song = song_playlist.pop(old_index)
        # Insert at new position
        song_playlist.insert(new_index, song)
        print(f"Reordered: Moved {song.title} from {old_index} -> {new_index}")

        socketio.emit('update_songs', [_song_to_dict_plus_playing(s) for s in song_playlist])


def random_playlist_modification(playlist):
    """Randomly modify the global playlist: add, remove, reorder, or change state."""
    global next_tid

    operations = ["add", "remove", "reorder", "change_state"]
    operation = random.choice(operations)

    if operation == "add":
        # Create a new random title
        random_title = "NewSong_" + ''.join(random.choices(string.ascii_uppercase, k=5))
        new_song = Song(
            tid=next_tid,
            path=f"songs/{next_tid}.mp4",
            title=random_title,
            state="queue"
        )
        playlist.append(new_song)
        print(f"Added new song: {new_song.title} (tid={new_song.tid})")
        next_tid += 1

    elif operation == "remove":
        if playlist:
            # Remove a random song from the list
            idx = random.randrange(len(playlist))
            removed_song = playlist.pop(idx)
            print(f"Removed song tid={removed_song.tid}: {removed_song.title}")

    elif operation == "reorder":
        if len(playlist) > 1:
            random.shuffle(playlist)
            print("Reordered (shuffled) the playlist.")

    elif operation == "change_state":
        if playlist:
            song = random.choice(playlist)
            possible_states = ["queue", "processing", "done", "error"]
            old_state = song.state
            song.state = random.choice(possible_states)
            print(f"Changed state of tid={song.tid} from {old_state} to {song.state}")

def background_updater():
    while True:
        time.sleep(7)
        # random_playlist_modification(song_playlist)
        print(song_playlist)
        socketio.emit('update_songs', [_song_to_dict_plus_playing(s) for s in song_playlist])
if __name__ == "__main__":
    thread = threading.Thread(target=background_updater, daemon=True)
    thread.start()
    socketio.run(app, debug=True, host="0.0.0.0",port=5000,allow_unsafe_werkzeug=True)
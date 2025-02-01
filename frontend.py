import string

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from backend import Song
import threading
import time, random
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
next_tid = 4
song_playlist = [Song(tid=1, path='songs/keep=nothing_-7MEND5qWR1AliwI6mIs.mp4', title='Song1',state='processing'),
                 Song(tid=2, path='songs/keep=video_89eusrJpdJACDWxhAxzG.mp4', title='Song2', state='queue'),
                 Song(tid=3, path='songs/keep=all_89eusrJpdJACDWxhAxzG.mp4', title='Song3', state='done')]
@app.route("/")
def index():
    return render_template("index.html")

@socketio.on('connect')
def on_connect():
    print("Client connected.")
    emit('update_songs', [song.to_dict() for song in song_playlist])

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
        time.sleep(5)
        random_playlist_modification(song_playlist)
        socketio.emit('update_songs', [s.to_dict() for s in song_playlist])
if __name__ == "__main__":
    thread = threading.Thread(target=background_updater, daemon=True)
    thread.start()
    socketio.run(app, debug=True, host="0.0.0.0",port=5000,allow_unsafe_werkzeug=True)
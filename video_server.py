from flask import Flask, send_file, Response
import os

app = Flask(__name__)

@app.route('/video')
def stream_video():
    video_path = os.path.join(os.path.dirname(__file__), 'video.mp4')
    return send_file(video_path, mimetype='video/mp4')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 
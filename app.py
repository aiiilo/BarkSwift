from flask import Flask, request, jsonify
import librosa
import numpy as np
import soundfile as sf

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_wav():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    file_path = f"./uploads/{file.filename}"
    file.save(file_path)

    # 音声解析（ピッチ・リズムなど）
    y, sr = librosa.load(file_path, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    # ピッチの平均値を計算
    avg_pitch = np.mean(pitches[pitches > 0])

    return jsonify({
        "file": file.filename,
        "tempo": tempo,
        "average_pitch": avg_pitch
    })

if __name__ == '__main__':
    app.run(debug=True)


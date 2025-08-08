# --- app.py ---
from flask import Flask, request, jsonify
from main import (
    download_youtube_audio,
    transcribe_audio,
    summarize_txt,
    translate,
    detect_language,
    save_summary_as_audio
)
from flask_cors import CORS
import base64
import os

app = Flask(__name__)
CORS(app)

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        # Only YouTube URL supported
        if 'file' in request.files:
            return jsonify({'error': 'File upload not supported. Please use a YouTube URL.'}), 400

        youtube_url = request.form.get('url')
        if not youtube_url:
            return jsonify({'error': 'No YouTube URL provided'}), 400

        target_language = request.form.get('language', 'hi')

        # Load API key (hardcoded fallback used)
        bhashini_api_key = os.getenv(
            "BHASHINI_API_KEY",
            "iDb3Qb49PnN-1F647c-IRU3AMt15BgVaaQqA7naIMmDcCsZWo8SWjioSDWLvBPTy"
        )

        print("🔊 Step 1: Downloading audio...")
        audio_path = download_youtube_audio(youtube_url)

        print("📝 Step 2: Transcribing audio...")
        transcript = transcribe_audio()
        print(f"📃 Transcript sample:\n{transcript[:300]}...\n")

        print("🌍 Step 3: Detecting transcript language...")
        lang = detect_language(transcript)
        print(f"Detected language: {lang}")

        print("📚 Step 4: Summarizing transcript...")
        english_summary = summarize_txt(transcript, api_key=bhashini_api_key)
        print(f"✅ English Summary:\n{english_summary}\n")

        print(f"🌐 Step 5: Translating summary to '{target_language}'...")
        translated_summary = translate(target_language, english_summary, bhashini_api_key)

        print("🎧 Step 6: Generating audio summary...")
        audio_file_path = save_summary_as_audio(translated_summary, target_language)

        # Encode audio as base64
        with open(audio_file_path, 'rb') as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')

        print("✅ All steps completed successfully.")

        return jsonify({
            'summary': translated_summary,
            'transcript': transcript,
            'audio': audio_base64,
            'status': 'success'
        }), 200

    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Production use should replace this with gunicorn or waitress
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
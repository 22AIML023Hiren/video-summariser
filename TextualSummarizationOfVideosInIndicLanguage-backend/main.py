from transformers import pipeline
from pathlib import Path
from gtts import gTTS
import requests
import whisper
import yt_dlp
import os
import time
from langdetect import detect
from translatepy import Translator

# -----------------------------
# Load models once
# -----------------------------
print("🔁 Loading models...")
whisper_model = whisper.load_model("medium")
summarizer_model = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
fallback_translator = Translator()
print("✅ Models loaded successfully.")

# -----------------------------
def download_youtube_audio(url):
    try:
        download_dir = Path("downloads")
        download_dir.mkdir(exist_ok=True)

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(download_dir / '%(id)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': True,
            'noplaylist': True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info.get('id')
            audio_path = download_dir / f"{video_id}.wav"

        timeout = 10
        while timeout > 0 and not audio_path.exists():
            time.sleep(1)
            timeout -= 1

        if not audio_path.exists():
            raise Exception("Audio file not found after download")

        final_audio_dir = Path("audio_files")
        final_audio_dir.mkdir(exist_ok=True)
        final_audio_path = final_audio_dir / "audio.wav"

        if final_audio_path.exists():
            final_audio_path.unlink()

        os.replace(audio_path, final_audio_path)
        return str(final_audio_path)

    except Exception as e:
        raise Exception(f"YouTube download failed: {str(e)}")

# -----------------------------
def transcribe_audio(verbose=False):
    try:
        audio_path = Path("audio_files") / "audio.wav"
        if not audio_path.exists():
            raise Exception("Audio file not found")

        result = whisper_model.transcribe(str(audio_path), verbose=verbose)

        transcript_path = Path("file") / "transcript.txt"
        transcript_path.parent.mkdir(exist_ok=True)
        with open(transcript_path, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(result['text'])

        return result['text']

    except Exception as e:
        raise Exception(f"Transcription failed: {str(e)}")

# -----------------------------
def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return "unknown"

# -----------------------------
def summarize_txt(transcript, api_key=None):
    try:
        if len(transcript.strip()) < 50:
            raise Exception("Transcript too short for summarization")

        transcript = transcript.strip()[:4000]

        detected_lang = detect_language(transcript)
        print(f"🌐 Detected transcript language: {detected_lang}")

        if detected_lang != "en":
            if not api_key:
                raise Exception("API key required for translation to English")
            print("🔁 Translating transcript to English for summarization...")
            transcript = translate("en", transcript, api_key, source_language=detected_lang)

        length = len(transcript)
        max_len = min(250, length // 3)

        result = summarizer_model(
            transcript,
            max_length=max_len,
            min_length=50,
            do_sample=False
        )

        return result[0]['summary_text']

    except Exception as e:
        raise Exception(f"Summarization failed: {str(e)}")

# -----------------------------
def translate(target_language, text, api_key, source_language='auto'):
    try:
        if not api_key:
            raise Exception("Bhashini API key missing")

        url = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"
        payload = {
            "pipelineTasks": [{
                "taskType": "translation",
                "config": {
                    "language": {
                        "sourceLanguage": source_language,
                        "targetLanguage": target_language
                    },
                    "serviceId": "ai4bharat/indictrans-v2-all-gpu--t4"
                }
            }],
            "inputData": {
                "input": [{
                    "source": text[:5000]
                }]
            }
        }

        headers = {
            "Authorization": api_key,
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()

        return response.json()['pipelineResponse'][0]['output'][0]['target']

    except Exception as bhashini_error:
        print(f"⚠️ Bhashini failed: {str(bhashini_error)}")
        print("🔁 Switching to TranslatePy (fallback)...")
        try:
            translated = fallback_translator.translate(text, target_language)
            return translated.result
        except Exception as fallback_error:
            raise Exception(f"Fallback translation also failed: {str(fallback_error)}")

# -----------------------------
def save_summary_as_audio(translated_summary, language_code):
    try:
        if not translated_summary or len(translated_summary.strip()) < 10:
            raise Exception("Summary too short for audio")

        output_dir = Path("file")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / 'summary.wav'

        tts = gTTS(
            text=translated_summary,
            lang=language_code,
            tld='co.in',
            slow=False
        )
        tts.save(output_path)

        if not output_path.exists():
            raise Exception("Audio file not saved")

        return str(output_path)

    except Exception as e:
        raise Exception(f"Audio generation failed: {str(e)}")
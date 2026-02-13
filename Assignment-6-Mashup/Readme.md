# Mashup Assignment (Roll No. 102303982)

This folder contains both required programs:

- `102303982.py` — command-line mashup generator.
- `app.py` + `templates/index.html` — Flask web service that emails a zipped mashup.

## Setup

1. Install dependencies via the root `requirements.txt` .  
2. `imageio-ffmpeg` automatically provides a local ffmpeg binary; no system ffmpeg/ffprobe install needed.

## Program 1 — CLI

```
cd "mashup assignment"
python 102303982.py "<SingerName>" <NumberOfVideos> <AudioDurationSeconds> <OutputFileName>
```

Example (from inside the folder):

```
python 102303982.py "Sharry Maan" 20 25 output.mp3
```

Constraints:
- Number of videos must be > 10
- Duration per clip must be > 20 seconds

The script validates inputs, downloads audio from YouTube, trims each to the requested length, merges them, and saves a single MP3.

## Program 2 — Streamlit Web App

1. Configure email credentials via `.env` in this folder:
   - `SMTP_SERVER` (default `smtp.gmail.com`)
   - `SMTP_PORT` (default `587`)
   - `SMTP_USERNAME`
   - `SMTP_PASSWORD`
   - `FROM_EMAIL` (defaults to `SMTP_USERNAME`)
   - `SECRET_KEY` (any random string)
2. Start the app:

```
cd "mashup assignment"
streamlit run app.py --server.port 5000
```

3. Open the printed local URL (e.g., `http://localhost:5000`), fill the form (singer, videos >10, duration >20, email). The app emails a ZIP containing the mashup MP3.

## Notes

- Temporary files are stored under your system temp directory and cleaned automatically.
- If YouTube blocks or limits downloads for some results, the program stops with a clear message; try reducing the number of videos or using a different singer.
- Windows: downloads are retried with unlock waits; if a transient lock appears, just re-run.

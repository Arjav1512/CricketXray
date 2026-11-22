# Demonstration video

`make_video.py` rebuilds the project demonstration video from scratch, fully automated:
narration with edge-tts, title and slide cards rendered from HTML, the live Streamlit
dashboard driven by Playwright, and assembly with ffmpeg.

| File | What it is |
|---|---|
| `make_video.py` | One script that rebuilds everything below |
| `script.md` | Narration script with timestamps (generated) |
| `subtitles.srt` | English subtitles (generated; also embedded in the MP4) |
| `requirements.txt` | Extra Python packages the build needs |
| `Cricket_XRay_Demo.mp4` | The video (generated, not committed: about 40 MB) |
| `frames/` | One review frame per scene (generated, not committed) |
| `build/` | Cached audio, cards and intermediate clips (not committed) |

## Rebuild

From the project root, with the project's virtual environment and the datasets in `data/`:

```bash
pip install -r requirements.txt -r demo/requirements.txt
python -m playwright install chromium   # ffmpeg must also be on PATH
python demo/make_video.py
```

Before building, the script re-checks every number in the narration against a live
pipeline run and stops if any no longer matches. It starts and stops its own Streamlit
server and finishes with a play-check (duration, A/V sync, decode errors).

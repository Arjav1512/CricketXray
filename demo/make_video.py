#!/usr/bin/env python3
"""Rebuild the Cricket X-Ray demonstration video from scratch, fully automated.

Run from anywhere, with the project's virtual environment:

    .venv/bin/python demo/make_video.py              # full build
    .venv/bin/python demo/make_video.py --reuse-recording   # re-assemble, keep last screen capture

Extra requirements (beyond requirements.txt):
    pip install edge-tts playwright && python -m playwright install chromium
    ffmpeg and ffprobe on PATH

Pipeline
    1. Check every number used in the narration against a live pipeline run.
    2. Narration: edge-tts, one file per sentence (cached in demo/build/audio).
    3. Run the real test suite and capture its output for the Tests card.
    4. Cards: HTML in the dashboard's design system -> 1080p PNGs (Playwright screenshots).
    5. Start `streamlit run app.py --server.headless true`, drive it with Playwright (Chromium)
       at 1536x864 and capture it with the Chromium screencast (upscaled to 1080p). Every sentence's audio is placed at the
       wall-clock moment its on-screen action starts, so sound and picture stay in sync.
    6. ffmpeg: clips -> concatenated H.264 video; narration mixed and loudness-normalised
       (two-pass EBU R128); soft subtitles (mov_text) muxed in.
    7. Write script.md, subtitles.srt and one review frame per scene; play-check the result.
"""

import argparse
import asyncio
import base64
import hashlib
import html
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import wave
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo"
BUILD = DEMO / "build"
OUT_VIDEO = DEMO / "Cricket_XRay_Demo.mp4"
OUT_SCRIPT = DEMO / "script.md"
OUT_SRT = DEMO / "subtitles.srt"
FRAMES_DIR = DEMO / "frames"
PPTX = ROOT / "docs" / "presentation" / "Cricket_XRay_Presentation.pptx"

# en-IN-PrakashNeural (as requested) is not offered by the edge-tts service; PrabhatNeural is
# the en-IN male neural voice that is. The script tries the requested voice first.
VOICES = ["en-IN-PrakashNeural", "en-IN-PrabhatNeural"]
TTS_RATE = "+5%"
W, H, FPS = 1920, 1080, 30
SR = 48000
LINE_GAP = 0.3       # pause after each sentence
CARD_LEAD = 0.6      # silence before the first sentence of a card
CARD_TAIL = 0.7      # hold after the last sentence of a card
DEMO_TAIL = 0.6      # extra hold after each live-demo segment
FADE = 0.35
# The dashboard is rendered at 1536x864 CSS px and upscaled 1.25x to 1080p, which makes its text
# readable on video. (CSS zoom would be sharper but breaks Streamlit's dataframe widths.)
CAP_W, CAP_H = 1536, 864
GITHUB = "github.com/Arjav1512/CricketXray"

TEAM = [("Arjav Jain", "S25CSEU1497"), ("Nishkarsh Sisodia", "S25CSEU1499"),
        ("Drishti Chauhan", "S25CSEU1503")]
COURSE = "Statistical Machine Learning · Milestone 3"
FACULTY = "Mukund Pratap Singh"
UNIVERSITY = "Bennett University · School of Computer Science Engineering & Technology (SCSET)"


# ---------------------------------------------------------------------------
# Narration: one Line per sentence. Card lines carry a card state; demo lines an action.
# ---------------------------------------------------------------------------

@dataclass
class Line:
    text: str
    state: int = 0
    action: Optional[Callable] = None  # async (Demo, duration_seconds) -> None


@dataclass
class Scene:
    id: str
    title: str
    lines: list
    card: Optional[str] = None  # card template name; None = live demo
    # filled in during the build
    starts: list = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0


def build_scenes(tests):
    """Narration script. Numbers come from the project report; the test numbers from the live run."""
    if tests["ok"] and tests["skipped"] == 0:
        test_result = f"With the real data present, all {tests['ran']} pass."
    elif tests["ok"]:
        test_result = (f"In this run, {tests['ran'] - tests['skipped']} pass and "
                       f"{tests['skipped']} are skipped.")
    else:
        test_result = f"In this run, {tests['failed']} of {tests['ran']} did not pass."

    return [
        Scene("01_title", "Title", card="title", lines=[
            Line("Welcome to Cricket X-Ray: discovering cricket player performance profiles with "
                 "unsupervised machine learning."),
            Line("A Milestone 3 project by Arjav Jain, Nishkarsh Sisodia and Drishti Chauhan."),
        ]),
        Scene("02_problem", "Problem", card="problem", lines=[
            Line("Cricket players are usually described by informal roles: batter, bowler, "
                 "all-rounder and wicket-keeper.", 0),
            Line("But roles overlap: many batters also bowl, and many keepers are leading batters.", 1),
            Line("There is no reliable role label to learn from, so this is an unsupervised problem.", 2),
            Line("Our goal: group T20 players by how they contribute, using career statistics alone, "
                 "and explain each group in cricket terms.", 3),
        ]),
        Scene("03_data", "Data", card="data", lines=[
            Line("We use three T20 international career tables: batting, bowling and fielding.", 0),
            Line("Each holds the same 2,006 players, merged on the full name with its country tag.", 1),
            Line("A dash means missing, not zero: a player who never bowled has no economy rate.", 2),
            Line("And overs are not decimals: 10.5 overs is 10 overs and 5 balls, or 65 balls.", 3),
        ]),
        Scene("04_pipeline", "Pipeline", card="pipeline", lines=[
            Line("Here is the pipeline.", 0),
            Line("We load and validate the files into one table of 2,006 players and 44 columns.", 1),
            Line("We keep the 1,105 players with at least five matches, and engineer twelve "
                 "per-innings and per-match rates.", 2),
            Line("For the model, we clip each feature to its 1st to 99th percentile, fill undefined "
                 "rates with the median, add a has-bowled flag, and standardize.", 3),
            Line("K-Means groups the players.", 4),
            Line("PCA maps them in two dimensions.", 5),
            Line("And simple rules give each cluster a readable label.", 6),
        ]),
        Scene("05a_overview", "Live demo · Overview", lines=[
            Line("Now, the live Streamlit dashboard."),
            Line("The sidebar sets the minimum matches, 5, and the number of clusters K, 4."),
            Line("The Overview shows 2,006 source players, 1,105 modelled players, 9 model features, "
                 "and a silhouette score of 0.269."),
            Line("Below are a PCA preview and the size of each cluster.",
                 action=lambda d, t: d.scroll_to("PCA preview", offset=150)),
        ]),
        Scene("05b_data", "Live demo · Data", lines=[
            Line("The Data tab checks the inputs.", action=lambda d, t: d.tab("Data")),
            Line("Three source files, 2,006 rows each, no duplicates, and every player appears in all "
                 "three.", action=lambda d, t: d.scroll_to("Source files", offset=60)),
            Line("Here are the missing values per column; a dash always stays missing.",
                 action=lambda d, t: d.scroll_to("Missing values by source column")),
        ]),
        Scene("05c_features", "Live demo · Features", lines=[
            Line("In Features, the filter removes 901 players with fewer than five matches, leaving "
                 "1,105.", action=lambda d, t: d.tab("Features")),
            Line("We engineer twelve rates in three groups: batting, bowling and fielding.",
                 action=lambda d, t: d.scroll_to("Feature categories")),
            Line("The definitions table gives each formula.",
                 action=lambda d, t: d.scroll_to("Feature definitions")),
            Line("Eight of these rates, plus the has-bowled flag, form the nine model inputs."),
        ]),
        Scene("05d_evaluation", "Live demo · Model evaluation", lines=[
            Line("Model evaluation fits K-Means for every K from 2 to 8.",
                 action=lambda d, t: d.tab("Model evaluation")),
            Line("The elbow bends most sharply at K = 3, while the silhouette peaks at K = 4, at 0.269.",
                 action=lambda d, t: d.scroll_to("Elbow method · inertia")),
            Line("We chose K = 4: the best silhouette, and four clusters that are easy to interpret.",
                 action=lambda d, t: d.scroll_to("What the metrics say")),
            Line("A score this modest means weak structure, as expected, because roles overlap."),
        ]),
        Scene("05e_clusters", "Live demo · Clusters", lines=[
            Line("The Clusters tab shows the result for K = 4.", action=lambda d, t: d.tab("Clusters")),
            Line("The PCA map keeps 51.6% of the variance; each cluster has its own colour and marker.",
                 action=lambda d, t: d.scroll_to(
                     "PCA projection · players on the first two principal components")),
            Line("The four profiles: Bowling-Oriented, 463 players; Batting-Oriented Also Bowls, 320; "
                 "Batting-Oriented Rarely Bowls, 216; and Wicket-Keeping, 106.",
                 action=lambda d, t: d.scroll_to("Cluster profiles")),
            Line("The profile table gives the real median statistics.",
                 action=lambda d, t: d.scroll_to("Profile table · real statistics")),
            Line("The heatmap shows relative strengths, where the keepers stand out at +3.0 standard "
                 "deviations.", action=lambda d, t: d.scroll_to(
                     "Relative strengths · standard deviations from the average player")),
            Line("And the results can be downloaded as CSV.", action=lambda d, t: d.show_download()),
        ]),
        Scene("05f_player", "Live demo · Player explorer", lines=[
            Line("The Player explorer looks at one player at a time.",
                 action=lambda d, t: d.tab("Player explorer")),
            Line("Let's search for V Kohli of India.",
                 action=lambda d, t: d.search_player("V Kohli", "V Kohli (INDIA)")),
            Line("He falls in Cluster 2, Batting-Oriented, Also Bowls."),
            Line("This panel shows that Cluster 2 is his nearest centre, and compares his "
                 "standardized values with it.",
                 action=lambda d, t: d.scroll_to("Why this cluster", offset=40)),
            Line("Below are his source statistics, exactly as in the CSV files.",
                 action=lambda d, t: d.scroll_to("Source statistics")),
            Line("And here are his five most similar players.",
                 action=lambda d, t: d.scroll_to("Most similar players")),
        ]),
        Scene("05g_change_k", "Live demo · Changing K", lines=[
            Line("Every tab follows the sidebar.", action=lambda d, t: d.tab("Overview")),
            Line("Let's change K from 4 to 3.", action=lambda d, t: d.set_k(3)),
            Line("The model retrains at once: the metrics, the PCA map and the cluster sizes all "
                 "update, and the silhouette is now 0.265.",
                 action=lambda d, t: d.scroll_to("PCA preview", offset=150)),
            Line("Clusters now shows three profiles.",
                 action=lambda d, t: d.tab_then("Clusters", "Cluster profiles")),
            Line("Back to K = 4, and the four profiles return.", action=lambda d, t: d.set_k(4)),
        ]),
        Scene("05h_methodology", "Live demo · Methodology", lines=[
            Line("Finally, the Methodology tab explains every step in plain words, from the problem to "
                 "the cluster labels.",
                 action=lambda d, t: d.tab_then("Methodology", "Model input used in this run",
                                                ms=t * 750)),
            Line("It ends with the model inputs, the limitations and future work.",
                 action=lambda d, t: d.scroll_to(None, y=10**6, ms=t * 800)),
        ]),
        Scene("06_tests", "Tests", card="tests", lines=[
            Line(f"The project has {tests['ran']} automated tests, written with unittest and "
                 "Streamlit's AppTest.", 0),
            Line("They cover data loading, preprocessing, modeling and the dashboard.", 1),
            Line(test_result, 1),
            Line("Without the data, as on GitHub Actions, 52 pass and 28 skip by design.", 1),
        ]),
        Scene("07_validation", "Validation", card="validation", lines=[
            Line("Real players have no true labels, so we also built a synthetic league of 60 "
                 "players, 15 of each type.", 0),
            Line("With K = 4, K-Means recovered all four types with 100% purity, and every label was "
                 "correct.", 1),
            Line("Its silhouette of 0.583 shows that the modest real score comes from overlapping "
                 "roles, not from the pipeline.", 2),
        ]),
        Scene("08_conclusion", "Conclusion", card="conclusion", lines=[
            Line("To conclude: with no labels, Cricket X-Ray found four interpretable profiles among "
                 "1,105 T20 players.", 0),
            Line("The 0.269 silhouette shows that roles form a continuum, and the synthetic test shows "
                 "that the pipeline works.", 0),
            Line("Limits: career totals ignore batting position, era and form, and K-Means assumes "
                 "compact clusters.", 1),
            Line("Next, we would add ODI and Test data, try Gaussian mixtures, and use ball-by-ball "
                 "data.", 2),
            Line("The code is on GitHub. Thank you!", 3),
        ]),
    ]


SPOKEN = [
    (r"\bK = (\d)", r"K equals \1"),
    (r"(\d)%", r"\1 percent"),
    (r"\+(\d)", r"plus \1"),
    (r"\bunittest\b", "unit test"),
    (r"\bAppTest\b", "App Test"),
    (r"\b1st to 99th\b", "first to ninety-ninth"),
]


def spoken(text):
    for pattern, repl in SPOKEN:
        text = re.sub(pattern, repl, text)
    return text


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, **kw):
    result = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(map(str, cmd))}\n"
                           f"{result.stderr[-3000:]}")
    return result


def ffprobe_duration(path, stream=None):
    args = ["ffprobe", "-v", "error"]
    if stream:
        args += ["-select_streams", stream, "-show_entries", "stream=duration"]
    else:
        args += ["-show_entries", "format=duration"]
    out = run(args + ["-of", "default=nw=1:nk=1", str(path)]).stdout.split()
    return float(out[0]) if out and out[0] != "N/A" else float("nan")


def fmt_ts(t, srt=False):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}" if srt else f"{m:02d}:{s:02d}.{ms // 100}"


def frames_for(seconds):
    return max(1, int(round(seconds * FPS)))


# ---------------------------------------------------------------------------
# 1. Fact check: every number in the narration must still match the live pipeline
# ---------------------------------------------------------------------------

def verify_facts():
    sys.path.insert(0, str(ROOT))
    from cricket_xray.data_loader import load_data
    from cricket_xray.modeling import build_model_matrix, evaluate_k_range, run_pipeline
    from cricket_xray.preprocessing import activity_indicators, prepare_features

    data = load_data(ROOT / "data")
    merged = data["merged"]
    res = run_pipeline(merged, min_matches=5, k=4)
    feats = prepare_features(merged, min_matches=5)
    ev = evaluate_k_range(build_model_matrix(feats, activity_indicators(merged, feats.index))["X"])
    sizes = res["labels"].value_counts().sort_index().to_dict()
    labels = res["interpretation"]["label"].to_dict()
    ex = res["pca"]["explained"]
    second_diff = {k: ev["inertia"][k - 1] - 2 * ev["inertia"][k] + ev["inertia"][k + 1]
                   for k in range(3, 8)}
    checks = {
        "2,006 players": len(merged) == 2006,
        "44 merged columns": merged.shape[1] == 44,
        "1,105 modeled players": len(res["labels"]) == 1105,
        "9 model features": res["matrix"]["X"].shape[1] == 9,
        "12 engineered features": feats.shape[1] == 12,
        "cluster sizes 463/320/216/106": sizes == {1: 463, 2: 320, 3: 216, 4: 106},
        "cluster labels": labels == {1: "Bowling-Oriented", 2: "Batting-Oriented, Also Bowls",
                                     3: "Batting-Oriented, Rarely Bowls", 4: "Wicket-Keeping Profile"},
        "silhouette 0.269 at K=4": round(res["silhouette"], 3) == 0.269,
        "silhouette 0.265 at K=3": round(ev["silhouette"][3], 3) == 0.265,
        "silhouette 0.266 at K=6": round(ev["silhouette"][6], 3) == 0.266,
        "best silhouette at K=4": int(ev["silhouette"].idxmax()) == 4,
        "sharpest elbow at K=3": max(second_diff, key=second_diff.get) == 3,
        "PCA 51.6%": round(ex["total"] * 100, 1) == 51.6,
        "V Kohli (INDIA) in cluster 2": res["labels"].get("V Kohli (INDIA)") == 2,
    }
    bad = [name for name, ok in checks.items() if not ok]
    if bad:
        raise SystemExit("Narration facts no longer match the data: " + ", ".join(bad))
    log(f"Fact check passed ({len(checks)} checks).")


# ---------------------------------------------------------------------------
# 2. Narration audio
# ---------------------------------------------------------------------------

async def _synth(text, path, voice):
    import edge_tts
    last = "empty audio"
    for attempt in range(4):
        try:
            await edge_tts.Communicate(text, voice, rate=TTS_RATE).save(str(path))
            if path.stat().st_size > 0:
                return
        except Exception as error:  # network hiccups: retry, then give up loudly
            last = error
        await asyncio.sleep(1 + attempt)
    path.unlink(missing_ok=True)
    raise RuntimeError(f"edge-tts failed for {voice!r}: {last}")


async def pick_voice(cache):
    for voice in VOICES:
        probe = cache / f"probe_{voice}.mp3"
        if probe.exists() and probe.stat().st_size:
            return voice
        try:
            await _synth("Test.", probe, voice)
            return voice
        except Exception:
            probe.unlink(missing_ok=True)
            log(f"Voice {voice} is not available; trying the next one.")
    raise SystemExit("No edge-tts voice available.")


def decode_trimmed(mp3):
    """Decode to 48 kHz mono float32 and trim edge-tts's trailing silence."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(mp3), "-af",
           "areverse,silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.08,areverse",
           "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def synthesize_all(scenes):
    cache = BUILD / "audio"
    cache.mkdir(parents=True, exist_ok=True)

    async def go():
        voice = await pick_voice(cache)
        log(f"Narration voice: {voice}")
        sem = asyncio.Semaphore(4)
        jobs = []
        for scene in scenes:
            for line in scene.lines:
                text = spoken(line.text)
                key = hashlib.sha1(f"{voice}|{TTS_RATE}|{text}".encode()).hexdigest()[:16]
                line.mp3 = cache / f"{key}.mp3"
                if not (line.mp3.exists() and line.mp3.stat().st_size):
                    async def job(t=text, p=line.mp3):
                        async with sem:
                            await _synth(t, p, voice)
                    jobs.append(job())
        if jobs:
            log(f"Synthesizing {len(jobs)} sentences...")
        await asyncio.gather(*jobs)
        return voice

    voice = asyncio.run(go())
    for scene in scenes:
        for line in scene.lines:
            line.audio = decode_trimmed(line.mp3)
            line.dur = len(line.audio) / SR
    return voice


# ---------------------------------------------------------------------------
# 3. Tests
# ---------------------------------------------------------------------------

def run_tests():
    log("Running the test suite...")
    proc = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                          cwd=ROOT, capture_output=True, text=True)
    out = proc.stderr + proc.stdout
    ran = re.search(r"Ran (\d+) tests? in ([\d.]+)s", out)
    skipped = re.search(r"skipped=(\d+)", out)
    failures = sum(int(n) for n in re.findall(r"(?:failures|errors)=(\d+)", out))
    result_lines = [l for l in out.splitlines()
                    if re.search(r"\.\.\. (ok|skipped.*|FAIL|ERROR)$", l)]
    tests = {
        "ran": int(ran.group(1)) if ran else 0,
        "seconds": ran.group(2) if ran else "?",
        "skipped": int(skipped.group(1)) if skipped else 0,
        "failed": failures,
        "ok": proc.returncode == 0,
        "lines": result_lines,
        "summary": [l for l in out.splitlines()[-4:] if l.strip()],
    }
    log(f"Tests: ran {tests['ran']}, skipped {tests['skipped']}, failed {tests['failed']}, "
        f"{'OK' if tests['ok'] else 'NOT OK'}")
    return tests


# ---------------------------------------------------------------------------
# 4. Cards (HTML -> PNG)
# ---------------------------------------------------------------------------

BASE_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:1920px;height:1080px;overflow:hidden;background:#0A0A0A;color:#F5F5F3;
  font-family:-apple-system,BlinkMacSystemFont,'Helvetica Neue','Segoe UI',Arial,sans-serif;
  -webkit-font-smoothing:antialiased}
.page{position:relative;width:1920px;height:1080px;padding:84px 120px 0;
  background:radial-gradient(1100px 620px at 100% 0%,rgba(242,105,58,.075),transparent 62%),#0A0A0A}
.mono{font-family:'SF Mono',Menlo,Consolas,monospace}
.eyebrow{font-family:'SF Mono',Menlo,Consolas,monospace;font-size:21px;letter-spacing:.16em;
  text-transform:uppercase;color:#707070}
.eyebrow b{color:#F2693A;font-weight:500}
h1{font-size:66px;font-weight:500;letter-spacing:-.025em;line-height:1.08;margin-top:20px}
.lead{font-size:29px;line-height:1.5;color:#A3A3A3}
.card{background:#141414;border:1px solid #292929;border-radius:22px;padding:32px 36px}
.label{font-family:'SF Mono',Menlo,Consolas,monospace;font-size:17px;letter-spacing:.14em;
  text-transform:uppercase;color:#707070;margin-bottom:14px}
.accent{color:#F2693A}.green{color:#59D08A}.violet{color:#A78BFA}.muted{color:#A3A3A3}
.subtle{color:#707070}
.footer{position:absolute;left:120px;right:120px;bottom:44px;display:flex;justify-content:space-between;
  font-family:'SF Mono',Menlo,Consolas,monospace;font-size:16px;letter-spacing:.14em;
  text-transform:uppercase;color:#5A5A5A;border-top:1px solid #1E1E1E;padding-top:18px}
[data-s]{transition:none}
"""


def img_uri(data, mime="image/png"):
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def footer(n):
    return (f'<div class="footer"><span>Cricket X-Ray · {COURSE}</span>'
            f'<span>Bennett University · SCSET</span></div>')


def card_title(state, ctx):
    team = "".join(
        f'<div style="display:flex;justify-content:space-between;padding:14px 0;'
        f'border-top:1px solid #222"><span style="font-size:30px">{name}</span>'
        f'<span class="mono muted" style="font-size:24px;letter-spacing:.06em">{enrol}</span></div>'
        for name, enrol in TEAM)
    stats = "".join(
        f'<div style="flex:1"><div style="font-size:52px;font-weight:500;letter-spacing:-.02em" '
        f'class="{cls}">{v}</div><div class="muted" style="font-size:20px;margin-top:4px">{t}</div></div>'
        for v, t, cls in [("2,006", "players", ""), ("4", "profiles found", ""),
                          ("0.269", "silhouette at K = 4", "accent")])
    return f"""
<div class="page" style="display:grid;grid-template-columns:1.02fr .98fr;gap:80px;align-items:center;
  padding-top:0;padding-bottom:0">
  <div>
    <div class="eyebrow">T20 internationals · Unsupervised learning</div>
    <div style="font-size:136px;font-weight:500;letter-spacing:-.035em;line-height:1;margin-top:26px">
      Cricket <span class="accent">X</span>-Ray</div>
    <p class="lead" style="font-size:33px;margin-top:24px;max-width:840px;color:#C9C9C6">
      Discovering cricket player performance profiles using unsupervised machine learning</p>
    <div style="margin-top:46px;max-width:780px">{team}<div style="border-top:1px solid #222"></div></div>
    <div style="margin-top:34px;font-size:24px;line-height:1.7" class="muted">
      <div><span class="label" style="font-size:16px">Course</span>&nbsp; {COURSE}</div>
      <div><span class="label" style="font-size:16px">Faculty</span>&nbsp; {FACULTY}</div>
      <div style="font-size:21px" class="subtle">{UNIVERSITY}</div>
    </div>
  </div>
  <div class="card" style="padding:26px">
    <img src="{ctx['pca_img']}" style="width:100%;border-radius:14px;display:block">
    <div class="muted" style="font-size:19px;margin:16px 6px 0">1,105 T20 players on two principal
      components, coloured by the cluster K-Means found for them.</div>
    <div style="display:flex;gap:24px;margin:26px 6px 6px;border-top:1px solid #292929;padding-top:24px">
      {stats}</div>
  </div>
</div>"""


def card_problem(state, ctx):
    bullets = [
        (0, "Players are described by informal roles: <b>batter, bowler, all-rounder, wicket-keeper</b>."),
        (1, "Roles overlap: many batters also bowl, and many wicket-keepers are leading batters."),
        (2, "No reliable role column exists, so this is an <b class='accent'>unsupervised</b> "
            "learning task."),
    ]
    items = "".join(
        f'<div data-s="{s}" style="display:flex;gap:22px;align-items:flex-start;margin-bottom:30px">'
        f'<span style="width:12px;height:12px;border-radius:6px;background:#F2693A;margin-top:16px;'
        f'flex:none"></span><span style="font-size:32px;line-height:1.45">{t}</span></div>'
        for s, t in bullets)
    stats = "".join(
        f'<div class="card" data-s="{s}" style="padding:28px 32px"><div style="font-size:72px;'
        f'font-weight:500;letter-spacing:-.03em;line-height:1" class="{cls}">{v}</div>'
        f'<div class="muted" style="font-size:23px;margin-top:10px">{t}</div></div>'
        for s, v, t, cls in [(0, "2,006", "T20 international players in each source file", ""),
                             (0, "3", "career tables: batting, bowling and fielding", ""),
                             (2, "0", "role labels given to the algorithm", "accent")])
    return f"""
<div class="page">
  <div class="eyebrow"><b>01</b> · Problem</div>
  <h1>Can performance profiles be found without labels?</h1>
  <div style="display:grid;grid-template-columns:1.45fr 1fr;gap:70px;margin-top:58px">
    <div>{items}
      <div data-s="3" class="card" style="border-left:4px solid #F2693A;margin-top:18px">
        <div class="label">Goal</div>
        <div style="font-size:29px;line-height:1.5">Group T20 international players by <i>how</i> they
          contribute, using career statistics alone, and explain every group in cricket terms.</div>
      </div>
    </div>
    <div style="display:flex;flex-direction:column;gap:22px">{stats}</div>
  </div>
  {footer(2)}
</div>"""


def card_data(state, ctx):
    files = [("Batting", "data/Batting/t20.csv", "Runs, balls faced, not-outs, highest score, fours, "
              "sixes, strike rate"),
             ("Bowling", "data/Bowling/Bowling_t20.csv", "Overs, runs conceded, wickets, best bowling, "
              "economy, strike rate"),
             ("Fielding", "data/Fielding/Fielding_t20.csv", "Dismissals, catches, stumpings, keeper "
              "catches, most dismissals")]
    fcards = "".join(
        f'<div class="card"><div style="font-size:36px;font-weight:500">{n}</div>'
        f'<div class="mono accent" style="font-size:19px;margin-top:10px">{p}</div>'
        f'<div class="muted" style="font-size:23px;line-height:1.5;margin-top:18px;min-height:70px">{d}</div>'
        f'<div class="mono" style="font-size:20px;margin-top:20px;color:#59D08A">2,006 rows</div></div>'
        for n, p, d in files)
    notes = [(1, "One row per player", "One-to-one inner join on the full name with its country tag "
              "gives a <b>2,006 × 44</b> table."),
             (2, "'-' means missing, not zero", "Did-not-bat and did-not-bowl values become NaN. Any "
              "other bad value stops with a readable error."),
             (3, "Overs are not decimals", "<b>10.5 overs</b> = 10 overs + 5 balls = <b>65 balls</b>, "
              "using <span class='mono'>b = 6o + d</span>, before economy is computed.")]
    ncards = "".join(
        f'<div class="card" data-s="{s}" style="border-top:3px solid #F2693A">'
        f'<div style="font-size:28px;font-weight:500">{t}</div>'
        f'<div class="muted" style="font-size:23px;line-height:1.5;margin-top:14px">{d}</div></div>'
        for s, t, d in notes)
    return f"""
<div class="page">
  <div class="eyebrow"><b>02</b> · Data</div>
  <h1>Three career tables, validated and merged</h1>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:28px;margin-top:50px">{fcards}</div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:28px;margin-top:28px">{ncards}</div>
  {footer(3)}
</div>"""


def card_pipeline(state, ctx):
    steps = [("Load and validate", "data_loader.py", "Check the schema, parse cricket notation, merge "
              "on the player name → 2,006 × 44"),
             ("Engineer features", "preprocessing.py", "Keep players with ≥ 5 matches (1,105); "
              "12 per-innings and per-match rates"),
             ("Clip · impute · scale", "modeling.py", "1st–99th percentile, median fill, has_bowled "
              "flag, StandardScaler"),
             ("K-Means", "modeling.py", "k-means++, n_init = 10, random_state = 42; K = 2 to 8"),
             ("PCA", "modeling.py", "Two components for the 2-D map (51.6% of the variance)"),
             ("Labels", "modeling.py · app.py", "Explicit rules turn cluster profiles into readable "
              "names")]
    boxes = []
    for i, (t, mod, d) in enumerate(steps, start=1):
        active = i == state
        border = "#F2693A" if active else "#292929"
        glow = "box-shadow:0 0 0 1px #F2693A,0 0 46px rgba(242,105,58,.22);" if active else ""
        boxes.append(
            f'<div class="card" data-s="{i}" style="padding:26px 24px;border-color:{border};{glow}'
            f'display:flex;flex-direction:column">'
            f'<div class="mono" style="font-size:19px;color:{"#F2693A" if i <= state else "#707070"}">'
            f'{i:02d}</div><div style="font-size:29px;font-weight:500;margin-top:12px;line-height:1.2">{t}</div>'
            f'<div class="mono subtle" style="font-size:15px;margin-top:10px">{mod}</div>'
            f'<div class="muted" style="font-size:20px;line-height:1.45;margin-top:16px">{d}</div></div>')
        if i < len(steps):
            boxes.append(f'<div data-s="{i + 1}" style="align-self:center;font-size:30px;'
                         f'color:#5A5A5A">→</div>')
    return f"""
<div class="page">
  <div class="eyebrow"><b>03</b> · Methodology</div>
  <h1>Pipeline at a glance</h1>
  <div style="display:grid;grid-template-columns:1fr auto 1fr auto 1fr auto 1fr auto 1fr auto 1fr;
    gap:14px;margin-top:56px;height:330px">{''.join(boxes)}</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:28px;margin-top:34px">
    <div class="card" data-s="3"><div class="label">Model input</div>
      <div style="font-size:26px;line-height:1.55">8 rate features + the <span class="mono accent">has_bowled</span>
        flag = <b>9 features</b>, each standardized to mean 0 and SD 1. Undefined rates stay missing
        until the median fill, never 0.</div></div>
    <div class="card" data-s="4"><div class="label">K-Means objective</div>
      <div style="font-size:26px;line-height:1.55">Minimise inertia
        <span class="mono accent">J = Σ<sub>k</sub> Σ<sub>i∈C<sub>k</sub></sub> ‖x<sub>i</sub> − μ<sub>k</sub>‖²</span>.
        K is chosen with the elbow method and the silhouette score; clusters are renumbered 1…K by size.</div></div>
  </div>
  {footer(4)}
</div>"""


def card_tests(state, ctx):
    t = ctx["tests"]
    shown = t["lines"][-14:]
    hidden = len(t["lines"]) - len(shown)

    def row(line):
        m = re.match(r"(.*) \.\.\. (.*)$", line)
        name, status = (m.group(1), m.group(2)) if m else (line, "")
        colour = "#59D08A" if status == "ok" else ("#F8B544" if status.startswith("skipped") else "#EF5B5B")
        return (f'<div style="display:flex;gap:14px"><span style="flex:1;overflow:hidden;'
                f'white-space:nowrap;text-overflow:ellipsis;color:#A3A3A3">{html.escape(name)} ...</span>'
                f'<span style="color:{colour}">{html.escape(status)}</span></div>')

    body = (f'<div><span class="violet">~/CricketXray</span> <span class="green">%</span> '
            f'python -m unittest discover -s tests -v'
            f'{"<span style=background:#F5F5F3>&nbsp;</span>" if state == 0 else ""}</div>')
    if state >= 1:
        body += (f'<div class="subtle" style="margin:10px 0 6px">⋮ {hidden} earlier tests</div>'
                 + "".join(row(l) for l in shown)
                 + '<div class="subtle" style="margin-top:12px">' + "-" * 70 + "</div>"
                 + f'<div style="margin-top:6px">Ran {t["ran"]} tests in {t["seconds"]}s</div>'
                 + (f'<div class="green" style="margin-top:10px;font-weight:600">OK</div>' if t["ok"] else
                    f'<div style="color:#EF5B5B;margin-top:10px">{html.escape(t["summary"][-1])}</div>'))
    passed = t["ran"] - t["skipped"] - t["failed"]
    chips = "".join(
        f'<div class="card" style="padding:24px 28px"><div style="font-size:60px;font-weight:500;'
        f'line-height:1;color:{c}">{v}</div><div class="muted" style="font-size:21px;margin-top:8px">{lbl}</div></div>'
        for v, lbl, c in [(t["ran"], "tests run", "#F5F5F3"), (passed, "passed", "#59D08A"),
                          (t["skipped"], "skipped", "#F8B544"), (t["failed"], "failed", "#EF5B5B")])
    return f"""
<div class="page">
  <div class="eyebrow"><b>05</b> · Tests</div>
  <h1>{t['ran']} automated tests: unittest and Streamlit AppTest</h1>
  <div style="display:grid;grid-template-columns:1fr 380px;gap:36px;margin-top:44px">
    <div style="background:#0F0F0F;border:1px solid #292929;border-radius:18px;overflow:hidden">
      <div style="display:flex;gap:9px;align-items:center;padding:16px 20px;border-bottom:1px solid #222;
        background:#141414"><span style="width:13px;height:13px;border-radius:7px;background:#3A3A3A"></span>
        <span style="width:13px;height:13px;border-radius:7px;background:#3A3A3A"></span>
        <span style="width:13px;height:13px;border-radius:7px;background:#3A3A3A"></span>
        <span class="mono subtle" style="font-size:15px;margin-left:14px">Terminal · real run captured
          while building this video</span></div>
      <div class="mono" style="font-size:17.5px;line-height:1.62;padding:20px 26px;height:640px;
        color:#F5F5F3">{body}</div>
    </div>
    <div data-s="1" style="display:flex;flex-direction:column;gap:18px">{chips}
      <div class="muted" style="font-size:19px;line-height:1.5;margin-top:4px">Without the dataset (as
        in GitHub Actions): 52 passed, 28 skipped, 0 failed.</div></div>
  </div>
  {footer(10)}
</div>"""


def card_validation(state, ctx):
    rows = [("Wicket-keeper", 1, "Wicket-Keeping Profile"), ("Bowler", 2, "Bowling-Oriented"),
            ("All-rounder", 3, "All-Round Statistical Profile"),
            ("Batter", 4, "Batting-Oriented, Rarely Bowls")]
    trs = "".join(
        f'<tr><td>{a}</td><td class="mono">{c}</td><td class="green mono">15 of 15</td><td>{l}</td></tr>'
        for a, c, l in rows)
    return f"""
<div class="page">
  <style>table{{width:100%;border-collapse:collapse;font-size:25px}}
    th{{text-align:left;font-family:'SF Mono',Menlo,monospace;font-size:16px;letter-spacing:.12em;
      text-transform:uppercase;color:#707070;font-weight:400;padding:0 0 16px}}
    td{{padding:18px 0;border-top:1px solid #262626}}</style>
  <div class="eyebrow"><b>06</b> · Validation</div>
  <h1>Validated on a synthetic league</h1>
  <p class="lead" style="margin-top:22px">60 synthetic players: 15 each of batters, bowlers,
    wicket-keepers and all-rounders, with up to ±10% random variation.</p>
  <div style="display:grid;grid-template-columns:420px 1fr;gap:34px;margin-top:40px" data-s="1">
    <div class="card" style="display:flex;flex-direction:column;justify-content:center">
      <div class="label">Cluster purity at K = 4</div>
      <div class="accent" style="font-size:150px;font-weight:500;letter-spacing:-.04em;line-height:1">100%</div>
      <div class="muted" style="font-size:23px;margin-top:14px">All four types recovered and correctly
        labelled.</div></div>
    <div class="card"><table><tr><th>Known type</th><th>Cluster</th><th>Recovered</th>
      <th>Label produced by the rules</th></tr>{trs}</table></div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:34px;margin-top:28px" data-s="2">
    <div class="card" style="padding:24px 32px;display:flex;justify-content:space-between;align-items:center">
      <span style="font-size:25px">Synthetic league</span><span class="mono" style="font-size:24px">
      silhouette <span class="green">0.583</span> · PCA <span class="green">76.0%</span></span></div>
    <div class="card" style="padding:24px 32px;display:flex;justify-content:space-between;align-items:center">
      <span style="font-size:25px">Real T20 data</span><span class="mono" style="font-size:24px">
      silhouette <span class="accent">0.269</span> · PCA <span class="accent">51.6%</span></span></div>
  </div>
  {footer(11)}
</div>"""


def card_conclusion(state, ctx):
    if state >= 3:
        team = " · ".join(f"{n} <span class='mono subtle' style='font-size:20px'>{e}</span>"
                          for n, e in TEAM)
        return f"""
<div class="page" style="display:flex;flex-direction:column;justify-content:center;align-items:center;
  text-align:center;padding:0">
  <div class="eyebrow">Cricket X-Ray</div>
  <div style="font-size:150px;font-weight:500;letter-spacing:-.04em;margin-top:28px">Thank you</div>
  <div class="mono accent" style="font-size:40px;margin-top:30px">{GITHUB}</div>
  <div style="font-size:26px;margin-top:70px">{team}</div>
  <div class="muted" style="font-size:23px;margin-top:20px">{COURSE} · Faculty: {FACULTY}</div>
  <div class="subtle" style="font-size:20px;margin-top:10px">{UNIVERSITY}</div>
</div>"""
    cols = [(0, "Key findings", "#59D08A", ["Four profiles found with no labels",
                                           "Silhouette 0.269: roles form a continuum",
                                           "Synthetic check: 100% recovery"]),
            (1, "Limitations", "#F8B544", ["Career totals ignore batting position, era and form",
                                          "Median filling only partly offset by has_bowled",
                                          "K-Means assumes compact clusters"]),
            (2, "Future scope", "#A78BFA", ["Add ODI and Test data",
                                           "Gaussian mixtures or hierarchical clustering",
                                           "Ball-by-ball and recent-season data"])]
    cards = "".join(
        f'<div class="card" data-s="{s}" style="border-top:3px solid {c};padding:34px 36px">'
        f'<div class="label" style="color:{c}">{t}</div>'
        + "".join(f'<div style="font-size:28px;line-height:1.4;padding:18px 0;border-top:1px solid #242424">'
                  f'{i}</div>' for i in items) + "</div>"
        for s, t, c, items in cols)
    return f"""
<div class="page">
  <div class="eyebrow"><b>07</b> · Conclusion</div>
  <h1>Conclusion and future scope</h1>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:30px;margin-top:60px">{cards}</div>
  <div style="margin-top:44px;display:flex;justify-content:space-between;align-items:center" class="muted">
    <span style="font-size:24px">1,105 of 2,006 players modelled · K = 4 · 9 standardized features</span>
    <span class="mono accent" style="font-size:24px">{GITHUB}</span></div>
  {footer(12)}
</div>"""


CARDS = {"title": card_title, "problem": card_problem, "data": card_data, "pipeline": card_pipeline,
         "tests": card_tests, "validation": card_validation, "conclusion": card_conclusion}


def card_document(name, state, ctx):
    dim = "".join(f'[data-s="{n}"]{{opacity:.17}}' for n in range(state + 1, 12))
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{BASE_CSS}{dim}</style></head>"
            f"<body>{CARDS[name](state, ctx)}</body></html>")


async def render_cards(scenes, ctx, browser):
    out = BUILD / "cards"
    out.mkdir(parents=True, exist_ok=True)
    page = await browser.new_page(viewport={"width": W, "height": H})
    for scene in scenes:
        if not scene.card:
            continue
        for state in sorted({line.state for line in scene.lines}):
            path = out / f"{scene.card}_{state}.png"
            await page.set_content(card_document(scene.card, state, ctx), wait_until="load")
            await page.wait_for_timeout(150)
            await page.screenshot(path=str(path))
    await page.close()
    log("Cards rendered.")


# ---------------------------------------------------------------------------
# 5. Live demo: Streamlit + Playwright + Chromium screencast
# ---------------------------------------------------------------------------

RECORDING_CSS = """
[data-testid="stHeader"],[data-testid="stDecoration"],[data-testid="stStatusWidget"]{display:none!important}
::-webkit-scrollbar{display:none}
"""

SCROLL_JS = """
async ([el, y, ms, offset]) => {
  const m = document.querySelector('[data-testid="stMain"]');
  // only stMain may scroll; scrollIntoView would also shift its ancestors (and the sidebar)
  const pin = () => { for (let n = m.parentElement; n; n = n.parentElement) { n.scrollTop = 0; n.scrollLeft = 0; } };
  pin();
  const start = m.scrollTop;
  let end = y;
  if (el) { el.scrollIntoView({block: 'start'}); end = m.scrollTop - offset; m.scrollTop = start; pin(); }
  end = Math.max(0, Math.min(end, m.scrollHeight - m.clientHeight));
  if (ms == null) ms = Math.min(2400, Math.max(900, Math.abs(end - start) * 1.3));
  await new Promise(done => {
    const t0 = performance.now();
    const step = now => {
      const k = Math.min(1, (now - t0) / ms);
      const q = k < .5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
      m.scrollTop = start + (end - start) * q;
      k < 1 ? requestAnimationFrame(step) : done();
    };
    requestAnimationFrame(step);
  });
}
"""


class Demo:
    """Small, slow, deliberate interactions with the dashboard."""

    def __init__(self, page):
        self.p = page
        self.k = 4

    async def park(self):
        await self.p.mouse.move(CAP_W - 8, CAP_H // 2)  # empty right margin; no hover effects

    def panel(self):
        return self.p.get_by_role("tabpanel").filter(visible=True)

    async def scroll_to(self, text, y=None, ms=None, offset=28):
        handle = None
        if text:
            loc = self.panel().get_by_text(text, exact=True).first
            await loc.wait_for(state="visible", timeout=15000)
            handle = await loc.element_handle()
        await self.p.evaluate(SCROLL_JS, [handle, y, ms, offset])

    async def scroll_top(self):
        if await self.p.evaluate("document.querySelector('[data-testid=\"stMain\"]').scrollTop") > 4:
            await self.p.evaluate(SCROLL_JS, [None, 0, 800, 0])

    async def tab(self, name):
        await self.scroll_top()
        await self.p.get_by_role("tab", name=name, exact=True).click()
        await self.park()
        await self.p.wait_for_timeout(500)

    async def tab_then(self, name, target, ms=None):
        await self.tab(name)
        await self.p.wait_for_timeout(400)
        await self.scroll_to(target, ms=ms)

    async def wait_for_k(self, k):
        await self.p.wait_for_function(
            "k => { const m = document.querySelector('.cx-meta');"
            " return m && m.innerText.includes('K = ' + k); }", arg=k, timeout=60000)
        # the rerun has finished once no element is shown as stale
        await self.p.wait_for_timeout(400)
        await self.p.wait_for_function(
            "() => !document.querySelector('[data-stale=\"true\"]')", timeout=60000)

    async def set_k(self, k):
        slider = self.p.get_by_role("slider", name="Number of clusters (K)")
        await slider.focus()
        key = "ArrowLeft" if k < self.k else "ArrowRight"
        for _ in range(abs(k - self.k)):
            await self.p.keyboard.press(key)
            await self.p.wait_for_timeout(120)
        self.k = k
        await self.wait_for_k(k)
        await slider.evaluate("e => e.blur()")
        await self.park()

    async def search_player(self, query, full_name):
        box = self.p.get_by_role("combobox", name="Player (type to search)")
        await box.click()
        await self.p.wait_for_timeout(300)
        await self.p.keyboard.type(query, delay=130)
        await self.p.wait_for_timeout(450)
        await self.p.keyboard.press("Enter")
        await self.panel().get_by_text(full_name, exact=True).first.wait_for(timeout=30000)
        await box.evaluate("e => e.blur()")
        await self.park()

    async def show_download(self):
        await self.scroll_to("Download results", offset=220)
        await self.p.wait_for_timeout(500)
        await self.p.get_by_role("button", name="Player results (CSV)").hover()
        await self.p.wait_for_timeout(1400)
        await self.park()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_streamlit():
    port = free_port()
    logf = open(BUILD / "streamlit.log", "w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
         "--server.port", str(port), "--browser.gatherUsageStats", "false",
         "--server.fileWatcherType", "none"],
        cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    url = f"http://localhost:{port}"
    for _ in range(120):
        if proc.poll() is not None:
            raise RuntimeError("Streamlit exited early; see demo/build/streamlit.log")
        try:
            if urllib.request.urlopen(f"{url}/_stcore/health", timeout=1).status == 200:
                log(f"Streamlit is up at {url}")
                return proc, url
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("Streamlit did not respond within 60 s")


def stop_streamlit(proc):
    if proc and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            os.killpg(proc.pid, signal.SIGKILL)
        log("Streamlit server stopped.")


async def open_dashboard(browser, url):
    ctx = await browser.new_context(viewport={"width": CAP_W, "height": CAP_H}, device_scale_factor=1)
    page = await ctx.new_page()
    await page.goto(url)
    await page.get_by_text("Model trained").wait_for(timeout=180000)
    await page.add_style_tag(content=RECORDING_CSS)
    await page.wait_for_function(
        "() => [...document.images].every(i => i.complete) && document.images.length > 1",
        timeout=60000)
    await page.wait_for_timeout(1500)
    return ctx, page


async def record_demo(browser, url, demo_scenes):
    # Warm the Streamlit cache (K = 3 and K = 4) so the recorded reruns are instant.
    ctx, page = await open_dashboard(browser, url)
    warm = Demo(page)
    await warm.set_k(3)
    await warm.set_k(4)
    await ctx.close()
    log("Cache warmed; recording the live demo...")

    fdir = BUILD / "screencast"
    shutil.rmtree(fdir, ignore_errors=True)
    fdir.mkdir(parents=True)
    ctx, page = await open_dashboard(browser, url)
    demo = Demo(page)
    await demo.park()
    cdp = await ctx.new_cdp_session(page)
    frames = []

    def on_frame(ev):
        ts = ev.get("metadata", {}).get("timestamp") or time.time()
        name = f"{len(frames):06d}.jpg"
        (fdir / name).write_bytes(base64.b64decode(ev["data"]))
        frames.append((ts, name))
        asyncio.ensure_future(cdp.send("Page.screencastFrameAck", {"sessionId": ev["sessionId"]}))

    cdp.on("Page.screencastFrame", on_frame)
    await cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 95, "maxWidth": CAP_W,
                                            "maxHeight": CAP_H, "everyNthFrame": 1})
    await asyncio.sleep(1.2)
    marks = []
    for scene in demo_scenes:
        log(f"  recording {scene.id}")
        scene_marks = []
        for line in scene.lines:
            t = time.time()
            scene_marks.append(t)
            if line.action:
                await line.action(demo, line.dur)
            remaining = t + line.dur + LINE_GAP - time.time()
            if remaining > 0:
                await asyncio.sleep(remaining)
        await asyncio.sleep(DEMO_TAIL)
        marks.append({"id": scene.id, "starts": scene_marks, "end": time.time()})
    await asyncio.sleep(0.4)
    end = time.time()
    await cdp.send("Page.stopScreencast")
    await ctx.close()
    info = {"frames": frames, "marks": marks, "end": end}
    (BUILD / "recording.json").write_text(json.dumps(info))
    log(f"Captured {len(frames)} frames over {end - marks[0]['starts'][0]:.1f} s.")
    return info


def encode_demo_clip(info, path):
    """Turn the timestamped screencast frames into a constant 30 fps clip."""
    frames = info["frames"]
    t0 = info["marks"][0]["starts"][0] - CARD_LEAD
    t1 = info["end"]
    n_frames = frames_for(t1 - t0)
    t1 = t0 + n_frames / FPS
    before = [i for i, (ts, _) in enumerate(frames) if ts <= t0]
    first = before[-1] if before else 0
    sel = [(max(ts, t0), name) for ts, name in frames[first:] if ts < t1]
    lst = ["ffconcat version 1.0"]
    for i, (ts, name) in enumerate(sel):
        nxt = sel[i + 1][0] if i + 1 < len(sel) else t1
        lst += [f"file 'screencast/{name}'", f"duration {max(nxt - ts, 0.001):.6f}"]
    lst.append(f"file 'screencast/{sel[-1][1]}'")
    (BUILD / "screencast.ffconcat").write_text("\n".join(lst) + "\n")
    dur = n_frames / FPS
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i",
         str(BUILD / "screencast.ffconcat"),
         "-vf", f"scale={W}:{H}:flags=lanczos,unsharp=5:5:0.4:5:5:0.0,fps={FPS},format=yuv420p,"
                f"fade=t=in:st=0:d={FADE},fade=t=out:st={dur - FADE:.3f}:d={FADE}",
         "-frames:v", str(n_frames), *X264, str(path)])
    return t0, dur


X264 = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p",
        "-r", str(FPS), "-profile:v", "high", "-video_track_timescale", "15360"]


# ---------------------------------------------------------------------------
# 6. Assembly
# ---------------------------------------------------------------------------

def encode_card_clip(scene, path):
    """Card clip: state images switch at sentence starts; returns the clip duration."""
    starts, t = [], CARD_LEAD
    for line in scene.lines:
        starts.append(t)
        t += line.dur + LINE_GAP
    total = t - LINE_GAP + CARD_TAIL
    n_frames = frames_for(total)
    total = n_frames / FPS
    segments = []  # (state, start)
    for line, s in zip(scene.lines, starts):
        if not segments or segments[-1][0] != line.state:
            segments.append((line.state, 0.0 if not segments else s))
    lst = ["ffconcat version 1.0"]
    for i, (state, s) in enumerate(segments):
        e = segments[i + 1][1] if i + 1 < len(segments) else total
        lst += [f"file 'cards/{scene.card}_{state}.png'", f"duration {e - s:.6f}"]
    lst.append(f"file 'cards/{scene.card}_{segments[-1][0]}.png'")
    lf = BUILD / f"{scene.id}.ffconcat"
    lf.write_text("\n".join(lst) + "\n")
    fade_in = 0.8 if scene.id.startswith("01") else FADE
    fade_out = 1.2 if scene.id.startswith("08") else FADE
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lf),
         "-vf", f"fps={FPS},format=yuv420p,fade=t=in:st=0:d={fade_in},"
                f"fade=t=out:st={total - fade_out:.3f}:d={fade_out}",
         "-frames:v", str(n_frames), *X264, str(path)])
    return starts, total


def write_wav(path, samples):
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def loudnorm(src, dst):
    target = "I=-16:TP=-1.5:LRA=11"
    p = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(src), "-af",
                        f"loudnorm={target}:print_format=json", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = json.loads(p.stderr[p.stderr.rindex("{"):p.stderr.rindex("}") + 1])
    af = (f"loudnorm={target}:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
          f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:"
          f"offset={m['target_offset']}:linear=true,aresample={SR}")
    run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-af", af, "-ar", str(SR), str(dst)])
    return m


def split_cue(text, limit=86):
    if len(text) <= limit:
        return [text]
    mid = len(text) // 2
    cands = [m.end() for m in re.finditer(r"[,:;.]\s", text)] or [m.end() for m in re.finditer(r"\s", text)]
    cut = min(cands, key=lambda i: abs(i - mid))
    return split_cue(text[:cut].strip(), limit) + split_cue(text[cut:].strip(), limit)


def wrap2(text, width=46):
    if len(text) <= width:
        return text
    words, best = text.split(), None
    for i in range(1, len(text.split())):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        score = max(len(a), len(b))
        if best is None or score < best[0]:
            best = (score, a, b)
    return f"{best[1]}\n{best[2]}"


def write_outputs(scenes, total, voice, tests):
    cues = []
    for scene in scenes:
        for line, start in zip(scene.lines, scene.starts):
            parts = split_cue(line.text)
            weights = [len(p) for p in parts]
            t = start
            for part, wgt in zip(parts, weights):
                d = line.dur * wgt / sum(weights)
                cues.append((t, t + d, part))
                t += d
    srt = []
    for i, (a, b, text) in enumerate(cues, start=1):
        b = min(b + 0.25, cues[i][0] if i < len(cues) else b + 0.25)
        srt.append(f"{i}\n{fmt_ts(a, True)} --> {fmt_ts(b, True)}\n{wrap2(text)}\n")
    OUT_SRT.write_text("\n".join(srt))

    md = ["# Cricket X-Ray: demonstration video script", "",
          f"Video: `demo/Cricket_XRay_Demo.mp4` · {fmt_ts(total)} · 1920×1080, 30 fps, H.264 + AAC, "
          "soft English subtitles", f"Narration voice: `{voice}` (edge-tts). Rebuild with "
          "`.venv/bin/python demo/make_video.py`.", "",
          "Timestamps are generated from the final timeline, so they match the video exactly.", "",
          "| # | Scene | Start | End | Length |", "|---|---|---|---|---|"]
    for i, scene in enumerate(scenes, start=1):
        md.append(f"| {i} | {scene.title} | {fmt_ts(scene.start)} | {fmt_ts(scene.end)} | "
                  f"{scene.end - scene.start:.1f} s |")
    for scene in scenes:
        md += ["", f"## {fmt_ts(scene.start)} · {scene.title}", ""]
        md.append("*On screen:* " + (f"card `{scene.card}`" if scene.card else
                                     "live Streamlit dashboard, driven by Playwright") + "")
        md.append("")
        for line, start in zip(scene.lines, scene.starts):
            md.append(f"- `[{fmt_ts(start)}]` {line.text}")
    md += ["", "## Sources", "",
           "All figures come from `Cricket_XRay_Project_Report.pdf` and the repository; "
           "`make_video.py` re-checks them against a live pipeline run before building. "
           f"The test figures ({tests['ran']} run, {tests['skipped']} skipped, {tests['failed']} failed) "
           "come from the test run captured during the build.", ""]
    OUT_SCRIPT.write_text("\n".join(md))


def assemble(scenes, rec_t0, demo_dur, voice, tests):
    clips, t = [], 0.0
    demo_scenes = [s for s in scenes if not s.card]
    rec = json.loads((BUILD / "recording.json").read_text())
    marks = {m["id"]: m for m in rec["marks"]}
    demo_done = False
    for scene in scenes:
        if scene.card:
            path = BUILD / f"clip_{scene.id}.mp4"
            starts, dur = encode_card_clip(scene, path)
            scene.start, scene.end = t, t + dur
            scene.starts = [t + s for s in starts]
            clips.append(path)
            t += dur
        else:
            if not demo_done:
                demo_start = t
                clips.append(BUILD / "clip_demo.mp4")
                t += demo_dur
                demo_done = True
            m = marks[scene.id]
            scene.starts = [demo_start + (s - rec_t0) for s in m["starts"]]
            scene.start = scene.starts[0] - (CARD_LEAD if scene is demo_scenes[0] else 0)
            scene.end = demo_start + (m["end"] - rec_t0)
    total = t
    demo_scenes[-1].end = demo_start + demo_dur
    for a, b in zip(scenes, scenes[1:]):  # contiguous scene boundaries
        if not a.card and not b.card:
            a.end = b.starts[0]

    (BUILD / "clips.ffconcat").write_text(
        "ffconcat version 1.0\n" + "".join(f"file '{c.name}'\n" for c in clips))
    run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i",
         str(BUILD / "clips.ffconcat"), "-c", "copy", str(BUILD / "video.mp4")])

    track = np.zeros(int(round(total * SR)) + SR, dtype=np.float32)
    for scene in scenes:
        for line, start in zip(scene.lines, scene.starts):
            i = int(round(start * SR))
            track[i:i + len(line.audio)] += line.audio[:max(0, len(track) - i)]
    track = track[:int(round(total * SR))]
    write_wav(BUILD / "narration.wav", track)
    meas = loudnorm(BUILD / "narration.wav", BUILD / "narration_norm.wav")
    log(f"Loudness: measured {meas['input_i']} LUFS -> target -16 LUFS")

    write_outputs(scenes, total, voice, tests)
    run(["ffmpeg", "-v", "error", "-y", "-i", str(BUILD / "video.mp4"),
         "-i", str(BUILD / "narration_norm.wav"), "-i", str(OUT_SRT),
         "-map", "0:v", "-map", "1:a", "-map", "2:s", "-c:v", "copy",
         "-c:a", "aac", "-b:a", "192k", "-ar", str(SR), "-ac", "2",
         "-c:s", "mov_text", "-metadata:s:s:0", "language=eng", "-disposition:s:0", "default",
         "-metadata", "title=Cricket X-Ray: project demonstration",
         "-movflags", "+faststart", "-t", f"{total:.3f}", str(OUT_VIDEO)])
    return total


def review_frames(scenes):
    shutil.rmtree(FRAMES_DIR, ignore_errors=True)
    FRAMES_DIR.mkdir(parents=True)
    for scene in scenes:
        last = scene.lines[-1]
        t = scene.starts[-1] + min(last.dur * 0.6, max(0.5, last.dur - 0.3))
        run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(OUT_VIDEO),
             "-frames:v", "1", str(FRAMES_DIR / f"{scene.id}.png")])
    log(f"Review frames written to {FRAMES_DIR.relative_to(ROOT)}/")


def play_check(scenes, total):
    v = ffprobe_duration(OUT_VIDEO, "v:0")
    a = ffprobe_duration(OUT_VIDEO, "a:0")
    fmt = ffprobe_duration(OUT_VIDEO)
    info = json.loads(run(["ffprobe", "-v", "error", "-show_streams", "-of", "json",
                           str(OUT_VIDEO)]).stdout)["streams"]
    decode = subprocess.run(["ffmpeg", "-v", "error", "-i", str(OUT_VIDEO), "-f", "null", "-"],
                            capture_output=True, text=True)
    print("\n=== Play-check ===")
    for s in info:
        desc = {"video": lambda s: f"{s['codec_name']} {s['width']}x{s['height']} {s['r_frame_rate']} fps",
                "audio": lambda s: f"{s['codec_name']} {s['sample_rate']} Hz {s['channels']} ch",
                "subtitle": lambda s: f"{s['codec_name']}"}[s["codec_type"]](s)
        print(f"  {s['codec_type']:<9} {desc}")
    print(f"  duration  container {fmt:.2f} s · video {v:.2f} s · audio {a:.2f} s · "
          f"A/V difference {abs(v - a) * 1000:.0f} ms")
    print(f"  decode    {'clean (no errors)' if not decode.stderr.strip() else decode.stderr[:500]}")
    print(f"  length    {fmt_ts(total)} ({'within' if 300 <= total <= 420 else 'OUTSIDE'} 5–7 min)")
    print("  scenes:")
    for s in scenes:
        print(f"    {fmt_ts(s.start)}–{fmt_ts(s.end)}  {s.title}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reuse-recording", action="store_true",
                    help="reuse the last screen capture (only valid if the demo narration is unchanged)")
    ap.add_argument("--keep-frames", action="store_true", help="keep raw screencast frames")
    args = ap.parse_args()
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise SystemExit(f"{tool} is required")
    BUILD.mkdir(parents=True, exist_ok=True)

    verify_facts()
    tests = run_tests()
    scenes = build_scenes(tests)
    voice = synthesize_all(scenes)
    demo_scenes = [s for s in scenes if not s.card]

    with zipfile.ZipFile(PPTX) as z:
        pca_img = img_uri(z.read("ppt/media/image-1-2.png"))
    ctx = {"tests": tests, "pca_img": pca_img}

    async def browser_work():
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            await render_cards(scenes, ctx, browser)
            if args.reuse_recording and (BUILD / "recording.json").exists():
                log("Reusing the previous screen recording.")
                info = json.loads((BUILD / "recording.json").read_text())
            else:
                proc = None
                try:
                    proc, url = start_streamlit()
                    info = await record_demo(browser, url, demo_scenes)
                finally:
                    stop_streamlit(proc)
            await browser.close()
            return info

    info = asyncio.run(browser_work())
    if args.reuse_recording:
        for scene, m in zip(demo_scenes, info["marks"]):
            gaps = [b - a for a, b in zip(m["starts"], m["starts"][1:])]
            if any(g + 0.05 < line.dur for g, line in zip(gaps, scene.lines)):
                raise SystemExit("Narration changed since the recording; run without --reuse-recording.")
    if not (args.reuse_recording and (BUILD / "clip_demo.mp4").exists()):
        log("Encoding the screen recording...")
        rec_t0, demo_dur = encode_demo_clip(info, BUILD / "clip_demo.mp4")
        (BUILD / "clip_demo.json").write_text(json.dumps({"t0": rec_t0, "dur": demo_dur}))
    else:
        d = json.loads((BUILD / "clip_demo.json").read_text())
        rec_t0, demo_dur = d["t0"], d["dur"]
    if not args.keep_frames:
        shutil.rmtree(BUILD / "screencast", ignore_errors=True)

    log("Assembling the final video...")
    total = assemble(scenes, rec_t0, demo_dur, voice, tests)
    review_frames(scenes)
    play_check(scenes, total)
    print(f"\nDone: {OUT_VIDEO.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""Speech-to-text web app (Whisper + Gradio).

Works in three places with the same file:
  - your laptop / Colab:   python app.py   (set SHARE=1 for a public link)
  - Hugging Face ZeroGPU:  just upload it (the @spaces.GPU decorator is used)
"""
import os
import tempfile

import gradio as gr
import librosa
import numpy as np
import torch
from transformers import pipeline

try:
    import spaces  # only present on Hugging Face Spaces

    def _duration(y, segs, model_id, lang):
        # ask ZeroGPU for only as much GPU time as this clip needs,
        # so the daily quota lasts longer (SR is defined below, used at call time)
        secs = len(y) / SR
        return int(min(120, max(20, 15 + 0.5 * secs)))

    gpu = spaces.GPU(duration=_duration)
except ImportError:

    def gpu(fn):
        return fn


ON_SPACE = bool(os.environ.get("SPACE_ID"))
SR = 16000
MAX_SEG = 28  # seconds; Whisper handles up to 30 s per window
MAX_AUDIO_S = 180  # cap on Spaces so one request stays inside the GPU time limit

# label -> (model id, language code passed to Whisper)
# The Kinyarwanda model's card says to use language="sw".
CHOICES = {
    "Kinyarwanda": ("mbazaNLP/Whisper-Small-Kinyarwanda", "sw"),
    "English": ("openai/whisper-small", "en"),
    "French": ("openai/whisper-small", "fr"),
    "Swahili": ("openai/whisper-small", "sw"),
}

_pipes = {}
DEVICE = 0 if torch.cuda.is_available() else -1


def get_pipe(model_id):
    if model_id not in _pipes:
        _pipes[model_id] = pipeline(
            "automatic-speech-recognition", model=model_id, device=DEVICE
        )
    return _pipes[model_id]


if ON_SPACE:  # load models once at startup so requests don't wait on downloads
    for _mid in sorted({m for m, _ in CHOICES.values()}):
        try:
            get_pipe(_mid)
        except Exception as _e:  # e.g. gated model without access/token yet
            print(f"WARNING: could not preload {_mid}: {_e}")


def split_audio(y, top_db=30):
    """Cut on silences, merge into segments <= MAX_SEG seconds.
    Returns a list of (start_sample, end_sample)."""
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        return []
    max_len = MAX_SEG * SR
    segs = []
    cur_s, cur_e = int(intervals[0][0]), int(intervals[0][1])
    for s, e in intervals[1:]:
        s, e = int(s), int(e)
        if e - cur_s <= max_len:
            cur_e = e
        else:
            segs.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    segs.append((cur_s, cur_e))
    out = []
    for s, e in segs:  # hard-split anything still too long
        while e - s > max_len:
            out.append((s, s + max_len))
            s += max_len
        out.append((s, e))
    return out


def srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def run_asr(pipe, audio, lang):
    try:
        res = pipe(
            {"raw": audio, "sampling_rate": SR},
            generate_kwargs={"language": lang, "task": "transcribe"},
        )
    except Exception:
        # Some fine-tuned checkpoints lack the language table: force the
        # decoder prompt directly instead.
        pipe.model.config.forced_decoder_ids = (
            pipe.tokenizer.get_decoder_prompt_ids(language=lang, task="transcribe")
        )
        res = pipe({"raw": audio, "sampling_rate": SR})
    return res["text"].strip()


@gpu
def transcribe_segments(y, segs, model_id, lang):
    """The only part that needs the GPU. Returns [(start, end, text), ...]."""
    pipe = get_pipe(model_id)
    out = []
    for s, e in segs:
        text = run_asr(pipe, y[s:e].astype(np.float32), lang)
        if text:
            out.append((s, e, text))
    return out


def transcribe(audio_path, choice):
    if audio_path is None:
        return "Please record or upload an audio file first.", None, None

    model_id, lang = CHOICES[choice]
    y, _ = librosa.load(audio_path, sr=SR, mono=True)
    if len(y) < SR * 0.3:
        return "Audio is too short.", None, None
    if ON_SPACE and len(y) > MAX_AUDIO_S * SR:
        return (
            f"Audio is longer than {MAX_AUDIO_S // 60} minutes. "
            "Please upload a shorter clip (this demo's GPU time is limited).",
            None,
            None,
        )

    segs = split_audio(y)
    if not segs:
        return "No speech detected.", None, None

    try:
        results = transcribe_segments(y, segs, model_id, lang)
    except Exception as e:
        return (
            f"Could not run the {choice} model: {e}\n"
            "If this is the Kinyarwanda model, access to it (and an HF_TOKEN) may be missing.",
            None,
            None,
        )
    if not results:
        return "No speech detected.", None, None

    full = " ".join(t for _, _, t in results)
    srt = [
        f"{i}\n{srt_time(s / SR)} --> {srt_time(e / SR)}\n{t}\n"
        for i, (s, e, t) in enumerate(results, 1)
    ]
    d = tempfile.mkdtemp()
    txt_path = os.path.join(d, "transcript.txt")
    srt_path = os.path.join(d, "transcript.srt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full)
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt))
    return full, txt_path, srt_path


with gr.Blocks(title="Speech Transcriber") as demo:
    gr.Markdown(
        "# Speech Transcriber\n"
        "Record or upload audio, pick the language, get the text. "
        "Kinyarwanda uses a Whisper-small model fine-tuned by a third party on "
        "Common Voice data, so accuracy drops on noisy audio and on speech unlike "
        "its training data. Subtitle timestamps are per speech segment, not per word."
    )
    with gr.Row():
        audio = gr.Audio(
            sources=["upload", "microphone"], type="filepath", label="Audio"
        )
        with gr.Column():
            lang = gr.Dropdown(list(CHOICES), value="Kinyarwanda", label="Language")
            btn = gr.Button("Transcribe", variant="primary")
    out = gr.Textbox(label="Transcript", lines=10, show_copy_button=True)
    with gr.Row():
        f_txt = gr.File(label="Download .txt")
        f_srt = gr.File(label="Download .srt")
    btn.click(transcribe, [audio, lang], [out, f_txt, f_srt])

if __name__ == "__main__":
    demo.queue().launch(share=os.environ.get("SHARE") == "1")

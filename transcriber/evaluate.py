"""Measure word error rate (WER) on clips you transcribed by hand.

Make a folder `test_clips/` with audio files and a `test_clips/labels.csv`:

    file,text
    clip1.wav,your correct transcript here
    clip2.wav,another correct transcript

Run:  python evaluate.py --language Kinyarwanda
"""
import argparse
import csv
import os
import re

import jiwer
import librosa
import numpy as np

from app import CHOICES, SR, get_pipe, run_asr, split_audio


def norm(t):
    t = t.lower()
    t = re.sub(r"[^\w\s']", " ", t)
    return re.sub(r"\s+", " ", t).strip()


ap = argparse.ArgumentParser()
ap.add_argument("--language", default="Kinyarwanda", choices=list(CHOICES))
ap.add_argument("--folder", default="test_clips")
args = ap.parse_args()

model_id, lang = CHOICES[args.language]
pipe = get_pipe(model_id)

refs, hyps = [], []
with open(os.path.join(args.folder, "labels.csv"), encoding="utf-8") as f:
    for row in csv.DictReader(f):
        y, _ = librosa.load(os.path.join(args.folder, row["file"]), sr=SR, mono=True)
        parts = [
            run_asr(pipe, y[s:e].astype(np.float32), lang) for s, e in split_audio(y)
        ]
        hyp = " ".join(parts)
        refs.append(norm(row["text"]))
        hyps.append(norm(hyp))
        print(f"\n{row['file']}\n  REF: {refs[-1]}\n  HYP: {hyps[-1]}")

print(f"\nClips: {len(refs)}")
print(f"WER: {jiwer.wer(refs, hyps) * 100:.1f}%")
print(f"CER: {jiwer.cer(refs, hyps) * 100:.1f}%")

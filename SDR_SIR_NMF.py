import os
import numpy as np
import librosa
import pandas as pd

# =========================
# Paths
# =========================
pred_44_path = r"C:\Users\izhak\OneDrive\Desktop\NMF_SDR\source_1_44kHz.wav"
pred_96_path = r"C:\Users\izhak\OneDrive\Desktop\NMF_SDR\source_2_96kHz.wav"

gt_44_path = r"C:\Users\izhak\OneDrive\Desktop\NMF_SDR\snar - 44_cut.wav"
gt_96_path = r"C:\Users\izhak\OneDrive\Desktop\NMF_SDR\snr G m - 96_cut.wav"

other_44_path = r"C:\Users\izhak\OneDrive\Desktop\NMF_SDR\mix without drums- 44KHZ_cut.wav"
other_96_path = r"C:\Users\izhak\OneDrive\Desktop\NMF_SDR\mix without drums- 96KHZ_cut.wav"

output_dir = r"C:\Users\izhak\OneDrive\Desktop\NMF"
os.makedirs(output_dir, exist_ok=True)


def load_audio(path, sr):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float64)


def trim_to_same_length(*signals):
    n = min(len(s) for s in signals)
    return [s[:n] for s in signals]


def sdr(ref, est, eps=1e-9):
    error = ref - est
    return 10 * np.log10((np.sum(ref ** 2) + eps) / (np.sum(error ** 2) + eps))


def si_sdr(ref, est, eps=1e-9):
    ref = ref - np.mean(ref)
    est = est - np.mean(est)

    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    target = alpha * ref
    noise = est - target

    return 10 * np.log10((np.sum(target ** 2) + eps) / (np.sum(noise ** 2) + eps))


def sir(ref, other, est, eps=1e-9):
    ref = ref - np.mean(ref)
    other = other - np.mean(other)
    est = est - np.mean(est)

    target_part = (np.dot(est, ref) / (np.dot(ref, ref) + eps)) * ref
    interference_part = (np.dot(est, other) / (np.dot(other, other) + eps)) * other

    return 10 * np.log10(
        (np.sum(target_part ** 2) + eps) /
        (np.sum(interference_part ** 2) + eps)
    )


def evaluate(condition, pred_path, gt_path, other_path, sr):
    pred = load_audio(pred_path, sr)
    gt = load_audio(gt_path, sr)
    other = load_audio(other_path, sr)

    pred, gt, other = trim_to_same_length(pred, gt, other)

    return {
        "Condition": condition,
        "Sample Rate": sr,
        "SDR": sdr(gt, pred),
        "SI-SDR": si_sdr(gt, pred),
        "SIR": sir(gt, other, pred),
        "Duration_sec": len(gt) / sr
    }


results_44 = evaluate(
    "Separated 44.1 kHz vs GT 44.1 kHz",
    pred_44_path,
    gt_44_path,
    other_44_path,
    44100
)

results_96 = evaluate(
    "Separated 96 kHz vs GT 96 kHz",
    pred_96_path,
    gt_96_path,
    other_96_path,
    96000
)

df = pd.DataFrame([results_44, results_96])

csv_path = os.path.join(output_dir, "metrics_native_sr_fast.csv")
txt_path = os.path.join(output_dir, "metrics_native_sr_fast.txt")

df.to_csv(csv_path, index=False)

with open(txt_path, "w", encoding="utf-8") as f:
    f.write(df.to_string(index=False))

print(df)
print("\nSaved to:")
print(csv_path)
print(txt_path)

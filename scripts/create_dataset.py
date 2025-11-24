# scripts/create_dataset.py

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import re

DATE_FMT = "%d.%m.%Y %H:%M:%S,%f"

# ---------------------------------------------------------
# Robust parser for Flow / Thorac / SPO2 data
# ---------------------------------------------------------

def parse_signal_txt(path):
    """
    Robust parser for Flow / Thorac / SPO2 TXT files.
    - Finds 'Data:' anywhere in file
    - Reads only actual sample rows
    - Splits timestamp and numeric value
    - Handles dd.MM.yyyy HH:mm:ss,fff timestamps
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    data_idx = None

    # Find "Data:" anywhere (no 50-line limit)
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith("data"):
            data_idx = i + 1
            break

    # Fallback: detect first timestamp line
    if data_idx is None:
        for i, ln in enumerate(lines):
            if re.match(r"\d{2}\.\d{2}\.\d{4}", ln.strip()):
                data_idx = i
                break

    if data_idx is None:
        print(f"⚠️ Could not detect Data block in {path}")
        return pd.DataFrame(columns=["timestamp", "value"])

    timestamps = []
    values = []

    for ln in lines[data_idx:]:
        ln = ln.strip()
        if not ln:
            continue
        if ";" not in ln:
            continue

        try:
            ts_str, val_str = ln.split(";", 1)
            ts_str = ts_str.strip()
            val_str = val_str.strip().replace(",", ".")

            # Parse timestamp in European format
            ts = pd.to_datetime(ts_str, format="%d.%m.%Y %H:%M:%S.%f", errors="coerce")
            if pd.isna(ts):
                ts = pd.to_datetime(ts_str, dayfirst=True, errors="coerce")
            if pd.isna(ts):
                continue

            # Parse numeric value
            try:
                val = float(val_str.split()[0])
            except:
                val = float(re.findall(r"-?\d+", val_str)[0])

            timestamps.append(ts)
            values.append(val)

        except:
            continue

    df = pd.DataFrame({"timestamp": timestamps, "value": values})
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    return df

# ---------------------------------------------------------
# Event File Parser
# ---------------------------------------------------------

def parse_events(path):
    text = Path(path).read_text(encoding='utf-8', errors='ignore').splitlines()
    rows = []
    for ln in text:
        ln = ln.strip()
        if not ln:
            continue
        if ln.lower().startswith("signal") or ln.lower().startswith("unit"):
            continue
        if ";" not in ln:
            continue

        parts = [p.strip() for p in ln.split(";")]
        if len(parts) < 1:
            continue

        if "-" not in parts[0]:
            continue

        left, right = parts[0].split("-", 1)

        # Parse start time
        try:
            start = pd.to_datetime(left, dayfirst=True)
        except:
            continue

        # Parse end time
        if re.match(r"\d{2}\.\d{2}\.\d{4}", right.strip()):
            end = pd.to_datetime(right.strip(), dayfirst=True)
        else:
            end = pd.to_datetime(start.strftime("%d.%m.%Y ") + right.strip(), dayfirst=True)

        label = None
        if len(parts) >= 3:
            label = parts[2]
        elif len(parts) >= 2:
            label = parts[1]

        rows.append((start, end, label))

    return pd.DataFrame(rows, columns=["start_time", "end_time", "event"])

# ---------------------------------------------------------
# Filtering
# ---------------------------------------------------------

def bandpass(sig, fs, low=0.07, high=0.8, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, sig)

# ---------------------------------------------------------
# Window generator
# ---------------------------------------------------------

def window_iter(start, end, window_s=30, overlap=0.5):
    step = window_s * (1 - overlap)
    cur = start
    while cur + pd.Timedelta(seconds=window_s) <= end:
        yield cur, cur + pd.Timedelta(seconds=window_s)
        cur += pd.Timedelta(seconds=step)

# ---------------------------------------------------------
# Window labeling
# ---------------------------------------------------------

def label_window(wstart, wend, events):
    dur = (wend - wstart).total_seconds()

    for _, ev in events.iterrows():
        s = ev["start_time"]; e = ev["end_time"]
        if pd.isna(s) or pd.isna(e):
            continue

        overlap = max(0, (min(e, wend) - max(s, wstart)).total_seconds())
        if overlap / dur > 0.50:

            lab = str(ev["event"]).strip().lower()
            if "obstruct" in lab or "apnea" in lab:
                return "Obstructive Apnea"
            if "hypopnea" in lab:
                return "Hypopnea"
            return ev["event"]

    return "Normal"

# ---------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------

def compute_feats(segment):
    v = segment["value"].values
    return {
        "mean": float(np.nanmean(v)),
        "std": float(np.nanstd(v)),
        "min": float(np.nanmin(v)) if v.size else np.nan,
        "max": float(np.nanmax(v)) if v.size else np.nan,
        "rms": float(np.sqrt(np.nanmean(v**2))) if v.size else np.nan,
    }

# ---------------------------------------------------------
# Process one participant folder
# ---------------------------------------------------------

def process_participant(folder, out_rows, window_s=30, overlap=0.5):
    folder = Path(folder)
    print(f"📂 Processing {folder.name}")

    flow = thor = spo2 = events = None

    for f in folder.iterdir():
        n = f.name.lower()
        if n.startswith("flow") and f.suffix == ".txt":
            flow = f
        elif ("thor" in n or "thorac" in n) and f.suffix == ".txt":
            thor = f
        elif "spo2" in n and f.suffix == ".txt":
            spo2 = f
        elif "event" in n and f.suffix == ".txt":
            events = f

    if flow is None:
        print("❌ No Flow file found in", folder)
        return

    flow_df = parse_signal_txt(flow)
    thor_df = parse_signal_txt(thor) if thor else None
    spo2_df = parse_signal_txt(spo2) if spo2 else None
    events_df = parse_events(events) if events else pd.DataFrame(columns=["start_time", "end_time", "event"])

    # -----------------------------
    # Sample rate estimation
    # -----------------------------
    if len(flow_df) > 1:
        dt = (flow_df["timestamp"].iloc[1] - flow_df["timestamp"].iloc[0]).total_seconds()
    else:
        print(f"⚠️ Not enough samples to compute dt in {folder.name}")
        return

    fs = 1.0 / dt if dt > 0 else 32.0

    # -----------------------------
    # Filtering
    # -----------------------------
    flow_df["value_filt"] = bandpass(flow_df["value"].values, fs)
    if thor_df is not None:
        thor_df["value_filt"] = bandpass(thor_df["value"].values, fs)
    if spo2_df is not None:
        spo2_df["value_filt"] = spo2_df["value"].rolling(window=4, min_periods=1).mean()

    # -----------------------------
    # Windowing
    # -----------------------------
    rec_start = flow_df["timestamp"].iloc[0]
    rec_end = flow_df["timestamp"].iloc[-1]

    for wstart, wend in window_iter(rec_start, rec_end, window_s, overlap):

        row = {
            "participant": folder.name,
            "window_start": wstart,
            "window_end": wend
        }

        # FLOW features
        seg = flow_df[(flow_df["timestamp"] >= wstart) & (flow_df["timestamp"] < wend)]
        feats = compute_feats(seg[["timestamp", "value_filt"]].rename(columns={"value_filt": "value"}))
        for k, v in feats.items():
            row[f"flow_{k}"] = v

        # THORACIC features
        if thor_df is not None:
            seg = thor_df[(thor_df["timestamp"] >= wstart) & (thor_df["timestamp"] < wend)]
            feats = compute_feats(seg[["timestamp", "value_filt"]].rename(columns={"value_filt": "value"}))
            for k, v in feats.items():
                row[f"thor_{k}"] = v

        # SPO2 features
        if spo2_df is not None:
            seg = spo2_df[(spo2_df["timestamp"] >= wstart) & (spo2_df["timestamp"] < wend)]
            feats = compute_feats(seg[["timestamp", "value_filt"]].rename(columns={"value_filt": "value"}))
            for k, v in feats.items():
                row[f"spo2_{k}"] = v

        row["label"] = label_window(wstart, wend, events_df)
        out_rows.append(row)

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-in_dir", required=True)
    parser.add_argument("-out_dir", required=True)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--overlap", type=float, default=0.5)
    args = parser.parse_args()

    input_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for p in sorted(input_dir.iterdir()):
        if p.is_dir():
            process_participant(p, rows, window_s=args.window, overlap=args.overlap)

    df = pd.DataFrame(rows)
    out_fp = out_dir / "breathing_dataset.csv"
    df.to_csv(out_fp, index=False)
    print("\n✅ Saved dataset to:", out_fp)

if __name__ == "__main__":
    main()

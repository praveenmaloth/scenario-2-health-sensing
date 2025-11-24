# scripts/vis.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
import re
from datetime import datetime

DATE_FMT = "%d.%m.%Y %H:%M:%S,%f"  # dd.MM.YYYY HH:MM:SS,fff

def parse_signal_txt(path):
    """Parse your signal txt file with header and 'Data:' line.
       Returns DataFrame with timestamp (datetime) and value (float)."""
    lines = Path(path).read_text(encoding='utf-8', errors='ignore').splitlines()
    # find Start Time and Sample Rate if present
    start_time = None
    sample_rate = None
    data_idx = None
    for i, ln in enumerate(lines[:40]):
        ln_low = ln.lower()
        if ln_low.startswith("start time:"):
            # start time line like 'Start Time: 5/30/2024 8:59:00 PM'
            val = ln.split(":", 1)[1].strip()
            try:
                # some header start times use MM/DD/YYYY with AM/PM
                # Parse with pandas to be robust:
                start_time = pd.to_datetime(val)
            except:
                start_time = None
        if "sample rate" in ln_low:
            try:
                sample_rate = float(re.findall(r"\d+", ln)[0])
            except:
                sample_rate = None
        if ln.strip().lower() == "data:":
            data_idx = i + 1
            break
    if data_idx is None:
        # try to find the line that begins data
        for i, ln in enumerate(lines):
            if re.match(r"\d{2}\.\d{2}\.\d{4}", ln.strip()):
                data_idx = i
                break
    rows = []
    for ln in lines[data_idx:]:
        ln = ln.strip()
        if ln == "" or ln.lower().startswith("signal") or ln.lower().startswith("unit"):
            continue
        # expected format: "30.05.2024 20:59:00,031; 120"
        # some lines might contain trailing commas or spaces
        if ";" in ln:
            left, right = ln.split(";", 1)
            ts_txt = left.strip()
            val_txt = right.strip().split()[0]
        else:
            # fallback to whitespace
            parts = ln.split()
            if len(parts) >= 2:
                ts_txt = " ".join(parts[:-1])
                val_txt = parts[-1]
            else:
                continue
        # normalize timestamp: some lines may have comma ms already
        ts_txt = ts_txt.replace(".", ".")  # safe no-op
        # fix possible missing date in end-of-range lines in events parsing - that is handled separately
        try:
            ts = datetime.strptime(ts_txt, "%d.%m.%Y %H:%M:%S,%f")
        except Exception:
            # try to be robust with pandas
            try:
                ts = pd.to_datetime(ts_txt, dayfirst=True)
            except:
                continue
        try:
            val = float(val_txt.replace(",","."))
        except:
            # strip non-numeric
            m = re.search(r"-?\d+", val_txt)
            val = float(m.group()) if m else np.nan
        rows.append((ts, val))
    df = pd.DataFrame(rows, columns=["timestamp","value"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df, start_time, sample_rate

def parse_events_txt(path):
    """Parse Flow Events file lines like:
       30.05.2024 23:48:45,119-23:49:01,408; 16;Hypopnea; N1
       returns DataFrame with start_time,end_time,event_label"""
    text = Path(path).read_text(encoding='utf-8', errors='ignore').splitlines()
    rows = []
    for ln in text:
        ln = ln.strip()
        if ln == "" or ln.lower().startswith("signal") or ln.lower().startswith("unit"):
            continue
        # split by ';'
        parts = [p.strip() for p in ln.split(";")]
        if len(parts) < 1:
            continue
        # first part often contains start-end separated by '-'
        first = parts[0]
        if "-" in first:
            left, right = first.split("-",1)
            left = left.strip()
            right = right.strip()
            # If right doesn't contain a date, copy date from left's date
            try:
                start = datetime.strptime(left, DATE_FMT)
            except:
                try:
                    start = pd.to_datetime(left, dayfirst=True)
                except:
                    continue
            # if right has date
            if re.match(r"\d{2}\.\d{2}\.\d{4}", right):
                try:
                    end = datetime.strptime(right, DATE_FMT)
                except:
                    try:
                        end = pd.to_datetime(right, dayfirst=True)
                    except:
                        end = start
            else:
                # right contains only time e.g. "23:49:01,408" -> combine date of start
                # build string as dd.MM.YYYY <space> right
                date_str = start.strftime("%d.%m.%Y")
                combined = f"{date_str} {right}"
                try:
                    end = datetime.strptime(combined, DATE_FMT)
                except:
                    end = start
            # event label often in parts[2] or parts[1]
            label = None
            for p in parts[1:]:
                # find alphabetic token like 'Hypopnea' or 'Obstructive Apnea'
                if any(x.isalpha() for x in p):
                    label = p.split()[0] if len(p.split())==1 else p
                    break
            if label is None and len(parts)>1:
                label = parts[1]
            rows.append((start, end, label))
        else:
            # fallback: if line contains two datetimes separated by space
            tokens = ln.split()
            if len(tokens) >= 2:
                try:
                    start = pd.to_datetime(tokens[0] + " " + tokens[1], dayfirst=True)
                    end = start
                    rows.append((start,end,None))
                except:
                    continue
    ev_df = pd.DataFrame(rows, columns=["start_time","end_time","event"])
    return ev_df

def plot_participant(folder, out_pdf):
    folder = Path(folder)
    # find files by pattern
    flow_file = None; thor_file=None; spo2_file=None; events_file=None
    for f in folder.iterdir():
        name = f.name.lower()
        if name.startswith("flow") or "flow" in name and f.suffix.lower() in [".txt",".csv"]:
            flow_file = f
        if name.startswith("thor") or "thorac" in name or "thorac" in name.lower():
            thor_file = f
        if name.startswith("sp") or "spo2" in name:
            spo2_file = f
        if "event" in name and ("flow" in name or "event" in name):
            events_file = f
    # parse
    flow_df, _, _ = parse_signal_txt(flow_file) if flow_file else (None,None,None)
    thor_df, _, _ = parse_signal_txt(thor_file) if thor_file else (None,None,None)
    spo2_df, _, _ = parse_signal_txt(spo2_file) if spo2_file else (None,None,None)
    events_df = parse_events_txt(events_file) if events_file else pd.DataFrame(columns=["start_time","end_time","event"])

    # plotting
    with PdfPages(out_pdf) as pdf:
        fig, axs = plt.subplots(3,1, figsize=(12,9), sharex=True)
        if flow_df is not None and not flow_df.empty:
            axs[0].plot(flow_df['timestamp'], flow_df['value'], linewidth=0.4)
            axs[0].set_ylabel("Flow")
        else:
            axs[0].text(0.5,0.5,"No Flow", ha='center')

        if thor_df is not None and not thor_df.empty:
            axs[1].plot(thor_df['timestamp'], thor_df['value'], linewidth=0.4)
            axs[1].set_ylabel("Thorax")
        else:
            axs[1].text(0.5,0.5,"No Thorax", ha='center')

        if spo2_df is not None and not spo2_df.empty:
            axs[2].plot(spo2_df['timestamp'], spo2_df['value'], linewidth=0.6)
            axs[2].set_ylabel("SpO2")
            axs[2].set_ylim(50,100)
        else:
            axs[2].text(0.5,0.5,"No SpO2", ha='center')

        # overlay events
        for _, r in events_df.iterrows():
            st = r['start_time']; ed = r['end_time']; lab = r['event']
            color = 'red' if lab and 'obstruct' in str(lab).lower() or (lab and 'apnea' in str(lab).lower()) else 'orange'
            for ax in axs:
                ax.axvspan(st, ed, color=color, alpha=0.25)
        handles = [mpatches.Patch(color='red',alpha=0.25,label='Obstructive Apnea/Apnea'),
                   mpatches.Patch(color='orange',alpha=0.25,label='Hypopnea')]
        axs[-1].legend(handles=handles, loc='upper right')
        fig.suptitle(folder.name)
        plt.xlabel("Time")
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', required=True, help='folder containing AP01.. or single participant folder')
    args = parser.parse_args()
    root = Path(args.name)
    outdir = root.parent.parent / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    # if root contains participant folders
    if any((root / p).is_dir() for p in root.iterdir()):
        for p in sorted(root.iterdir()):
            if p.is_dir():
                out_pdf = outdir / f"{p.name}_visualization.pdf"
                print("Plotting", p.name)
                plot_participant(p, out_pdf)
                print("Saved", out_pdf)
    else:
        # single participant
        out_pdf = outdir / f"{root.name}_visualization.pdf"
        plot_participant(root, out_pdf)
        print("Saved", out_pdf)

if __name__ == "__main__":
    main()

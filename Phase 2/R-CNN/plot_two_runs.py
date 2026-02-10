import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
# explicit two completed runs
RUN_FILES = [
    'offline-run-20260205_135124-s9rf2iqy.csv',
    'offline-run-20260206_002151-b68wabli.csv'
]
LABELS = ['Run1', 'Run2']
SMOOTH_W = 5
os.makedirs(RESULTS_DIR, exist_ok=True)

def smooth(s, w=SMOOTH_W):
    return s.rolling(window=w, min_periods=1, center=True).mean()

frames = []
for f in RUN_FILES:
    p = os.path.join(RESULTS_DIR, f)
    if not os.path.exists(p):
        raise SystemExit(f'Missing {p}')
    df = pd.read_csv(p)
    frames.append(df.sort_values('epoch').reset_index(drop=True))

# detect columns
cols = frames[0].columns.tolist()
train_col = next((c for c in cols if 'train' in c.lower() and 'loss' in c.lower()), None)
val_col = next((c for c in cols if 'valid' in c.lower() and 'loss' in c.lower()), None)
ap_col = next((c for c in cols if 'ap' in c.lower() or 'mitotic' in c.lower()), None)

# Training loss plot
if train_col:
    plt.figure(figsize=(8,5))
    for df, lab in zip(frames, LABELS):
        plt.plot(df['epoch'], smooth(df[train_col]), label=lab)
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Training Loss — Run1 vs Run2')
    plt.legend()
    out = os.path.join(RESULTS_DIR, 'two_runs_training_loss.png')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print('Saved', out)

# Validation loss plot
if val_col:
    plt.figure(figsize=(8,5))
    for df, lab in zip(frames, LABELS):
        plt.plot(df['epoch'], smooth(df[val_col]), label=lab)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss — Run1 vs Run2')
    plt.legend()
    out = os.path.join(RESULTS_DIR, 'two_runs_validation_loss.png')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print('Saved', out)

# AP plot
if ap_col:
    plt.figure(figsize=(8,5))
    for df, lab in zip(frames, LABELS):
        plt.plot(df['epoch'], smooth(df[ap_col]), label=lab)
    plt.xlabel('Epoch')
    plt.ylabel(ap_col)
    plt.title(f'{ap_col} — Run1 vs Run2')
    plt.legend()
    out = os.path.join(RESULTS_DIR, 'two_runs_AP.png')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print('Saved', out)

print('Done.')

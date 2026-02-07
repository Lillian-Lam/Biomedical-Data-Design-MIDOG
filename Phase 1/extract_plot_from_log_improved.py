import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(__file__) + '/..'
LOG_PATH = os.path.join(ROOT, 'training_full.log')
RESULTS_DIR = os.path.join(ROOT, 'results')

os.makedirs(RESULTS_DIR, exist_ok=True)

with open(LOG_PATH, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

# Find indices of run markers and the run name on the same line
run_idxs = []  # tuples (start_idx, run_name)
for i, L in enumerate(lines):
    m = re.search(r'offline-run-[0-9_\-a-zA-Z]+', L)
    if m:
        run_idxs.append((i, m.group(0)))

# Add EOF sentinel
run_segments = []
for idx, (start_i, run_name) in enumerate(run_idxs):
    end_i = run_idxs[idx+1][0] if idx+1 < len(run_idxs) else len(lines)
    run_segments.append((run_name, start_i, end_i))

print(f'Found {len(run_segments)} runs in log; parsing each for epoch tables...')

number_re = re.compile(r'[-+]?(?:\d*\.\d+|\d+)')

extracted = []
for run_name, s_i, e_i in run_segments:
    seg = lines[s_i:e_i]
    # find header line starting with 'epoch'
    header_idx = None
    header_line = None
    for j, L in enumerate(seg):
        if re.match(r'^\s*epoch\b', L, re.I):
            header_idx = j
            header_line = L.strip()
            break
    if header_idx is None:
        continue
    # collect rows after header
    rows = []
    buf = None
    for L in seg[header_idx+1:]:
        s = L.strip()
        if not s:
            continue
        # skip progress lines like 'Epoch 95/100' or bar lines
        if re.match(r'^Epoch\s+\d+\s*/\s*\d+', s, re.I):
            continue
        if '|' in s and '%' in s:
            # likely a progress bar, skip
            continue
        # If line starts with integer, start a new buffer
        if re.match(r'^\s*\d+\b', s):
            buf = s
        else:
            # continuation of previous buffer
            if buf is None:
                continue
            buf = buf + ' ' + s
        # try to extract numbers from buffer
        nums = number_re.findall(buf)
        # require at least epoch + 1 metric
        if len(nums) >= 2:
            # accept this as a completed row
            try:
                epoch = int(float(nums[0]))
            except Exception:
                # not an epoch index
                buf = None
                continue
            metrics = [float(x) for x in nums[1:]]
            rows.append([epoch] + metrics)
            buf = None
        else:
            # wait for more continuation lines
            continue
    if not rows:
        continue
    # normalize row lengths
    maxlen = max(len(r) for r in rows)
    norm_rows = [r + [np.nan] * (maxlen - len(r)) for r in rows]
    # parse header tokens to get names
    toks = re.findall(r"[A-Za-z0-9_\-]+", header_line)
    # remove leading 'epoch' token(s)
    toks = [t for t in toks if not re.match(r'^epoch$', t, re.I)]
    colnames = ['epoch']
    for k in range(maxlen-1):
        if k < len(toks):
            colnames.append(toks[k])
        else:
            colnames.append(f'metric_{k+1}')
    df = pd.DataFrame(norm_rows, columns=colnames)
    df = df.sort_values('epoch').drop_duplicates('epoch', keep='first').reset_index(drop=True)
    outcsv = os.path.join(RESULTS_DIR, f'{run_name}.csv')
    df.to_csv(outcsv, index=False)
    print(f'Wrote {outcsv} with {len(df)} epochs parsed')
    extracted.append((run_name, df))

if not extracted:
    print('No epoch tables extracted.')
else:
    print(f'Extracted epoch tables for {len(extracted)} runs; now plotting core metrics...')
    # plotting similar to earlier scripts
    def save_plot(xs, ys, labels, xlabel, ylabel, outname):
        plt.figure(figsize=(8,5))
        for x, y, lab in zip(xs, ys, labels):
            plt.plot(x, y, label=lab)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, outname))
        plt.close()
        print('Saved', outname)

    # detect common columns
    sample_cols = extracted[0][1].columns.tolist()
    train_col = next((c for c in sample_cols if 'train' in c.lower() and 'loss' in c.lower()), None)
    val_col = next((c for c in sample_cols if 'valid' in c.lower() and 'loss' in c.lower()), None)
    ap_col = next((c for c in sample_cols if 'ap' in c.lower() or 'mitotic' in c.lower()), None)

    # build per-run series
    labels = []
    xs = []
    train_ys = []
    val_ys = []
    ap_ys = []
    for name, df in extracted:
        labels.append(name)
        x = df['epoch'].values
        xs.append(x)
        train_ys.append(df[train_col].values if train_col in df.columns else np.full_like(x, np.nan, dtype=float))
        val_ys.append(df[val_col].values if val_col in df.columns else np.full_like(x, np.nan, dtype=float))
        ap_ys.append(df[ap_col].values if ap_col in df.columns else np.full_like(x, np.nan, dtype=float))

    if train_col:
        save_plot(xs, train_ys, labels, 'Epoch', 'Train Loss', 'improved_training_loss_all_runs.png')
    if val_col:
        save_plot(xs, val_ys, labels, 'Epoch', 'Validation Loss', 'improved_validation_loss_all_runs.png')
    if ap_col:
        save_plot(xs, ap_ys, labels, 'Epoch', ap_col, 'improved_AP_all_runs.png')

    print('Improved extraction complete.')

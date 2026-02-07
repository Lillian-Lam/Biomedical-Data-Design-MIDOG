#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
import subprocess
import time

ROOT = os.path.dirname(__file__) + '/..'
RESULTS = os.path.join(ROOT, 'results')
EXTRACT_SCRIPT = os.path.join(os.path.dirname(__file__), 'extract_plot_from_log_improved.py')

COMPLETION_EPOCH = 99
MONITOR_INTERVAL = 30  # seconds

os.makedirs(RESULTS, exist_ok=True)


def compute_stats_for_csv(path):
    df = pd.read_csv(path)
    run = os.path.splitext(os.path.basename(path))[0]
    max_epoch = int(df['epoch'].max())
    final_row = df[df['epoch'] == df['epoch'].max()].iloc[-1]
    def get_col(df, names):
        for n in names:
            if n in df.columns:
                return n
        return None
    val_col = get_col(df, ['valid_loss', 'valid', 'val_loss', 'validloss'])
    train_col = get_col(df, ['train_loss', 'train'])
    ap_col = get_col(df, [c for c in df.columns if 'ap' in c.lower() or 'mitotic' in c.lower()])
    voc_col = get_col(df, [c for c in df.columns if 'pascal' in c.lower() or 'voc' in c.lower()])
    stats = {
        'run': run,
        'epochs_parsed': max_epoch + 1 if df['epoch'].min() == 0 else max_epoch,
        'final_epoch': max_epoch,
    }
    if train_col:
        stats['final_train_loss'] = float(final_row[train_col])
        stats['mean_train_loss'] = float(df[train_col].mean())
        stats['std_train_loss'] = float(df[train_col].std())
        stats['min_train_loss'] = float(df[train_col].min())
    if val_col:
        stats['final_valid_loss'] = float(final_row[val_col])
        stats['best_valid_loss'] = float(df[val_col].min())
        stats['best_valid_epoch'] = int(df.loc[df[val_col].idxmin()]['epoch'])
        stats['mean_valid_loss'] = float(df[val_col].mean())
        stats['std_valid_loss'] = float(df[val_col].std())
    if ap_col:
        stats['final_AP'] = float(final_row[ap_col])
        stats['best_AP'] = float(df[ap_col].max())
        stats['best_AP_epoch'] = int(df.loc[df[ap_col].idxmax()]['epoch'])
        stats['mean_AP'] = float(df[ap_col].mean())
        stats['std_AP'] = float(df[ap_col].std())
    if voc_col:
        stats['final_VOC'] = float(final_row[voc_col])
        stats['best_VOC'] = float(df[voc_col].max())
        stats['best_VOC_epoch'] = int(df.loc[df[voc_col].idxmax()]['epoch'])
    return stats


def find_completed_run_csvs(min_rows=99):
    """Consider a run completed if we parsed at least `min_rows` epoch rows."""
    csvs = sorted(glob.glob(os.path.join(RESULTS, '*.csv')))
    completed = []
    for c in csvs:
        try:
            df = pd.read_csv(c)
        except Exception:
            continue
        if 'epoch' not in df.columns:
            continue
        if df.shape[0] >= min_rows:
            completed.append(c)
    return completed


def compute_and_write_summary():
    completed = find_completed_run_csvs()
    rows = []
    for c in completed:
        rows.append(compute_stats_for_csv(c))
    if not rows:
        print('No completed runs found (parsed rows >= {}).'.format(99))
        return None
    sdf = pd.DataFrame(rows)
    out = os.path.join(RESULTS, 'summary_stats_detailed.csv')
    sdf.to_csv(out, index=False)
    print('Wrote', out)
    print(sdf.to_string(index=False))
    return out


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--once', action='store_true', help='Compute stats once and exit')
    p.add_argument('--monitor', action='store_true', help='Monitor log and exit when current run completes')
    args = p.parse_args()

    if args.once:
        compute_and_write_summary()
        raise SystemExit(0)

    if args.monitor:
        print('Starting monitor: will re-run extractor and check for completion every {}s'.format(MONITOR_INTERVAL))
        try:
            while True:
                # re-run extraction to refresh CSVs
                subprocess.run(['python3', EXTRACT_SCRIPT], check=False)
                # recompute summary
                out = compute_and_write_summary()
                # check the specific current run
                target = 'offline-run-20260206_103855-ki60kkio'
                target_csv = os.path.join(RESULTS, f'{target}.csv')
                if os.path.exists(target_csv):
                    try:
                        df = pd.read_csv(target_csv)
                        max_epoch = int(df['epoch'].max())
                        print('Current run max epoch =', max_epoch)
                        if max_epoch >= COMPLETION_EPOCH:
                            print('Current run reached completion epoch. Exiting monitor.')
                            break
                    except Exception as e:
                        print('Error reading target csv:', e)
                time.sleep(MONITOR_INTERVAL)
        except KeyboardInterrupt:
            print('Monitor interrupted by user.')
        raise SystemExit(0)

    # default: compute once
    compute_and_write_summary()

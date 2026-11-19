"""Generate publication-ready, separate Phase 5 figures without hardcoded colours."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def figures(training_csv,condition_csv,paired_csv,out):
    out=Path(out)
    if out.exists():raise FileExistsError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True)
    history=load(training_csv)
    condition=load(condition_csv)
    paired=load(paired_csv)
    if not history or not condition or not paired:
        raise ValueError('missing training or statistical results')
    epochs=[int(r['episode']) for r in history]
    weighted=[float(r['train_weighted_admission_ratio']) for r in history]
    plt.figure(figsize=(8,4.5))
    plt.plot(epochs,weighted,label='Training weighted admission')
    val=[r for r in history if r['validation_weighted_admission']!='']
    if val:
        plt.plot([int(r['episode']) for r in val],
                 [float(r['validation_weighted_admission']) for r in val],
                 marker='o',label='Validation weighted admission')
    plt.ylim(0,1.02);plt.xlabel('Training episode');plt.ylabel('Priority-weighted admission')
    plt.title('Phase 5 — Training and validation performance')
    plt.grid(axis='y',alpha=0.25);plt.legend();plt.tight_layout()
    plt.savefig(out/'phase5_learning_curve.png',dpi=180);plt.close()

    labels=[r['policy'] for r in condition]
    means=[float(r['admission_mean'])*100 for r in condition]
    yerrlo=[max(0,(float(r['admission_mean'])-float(r['admission_lower_95']))*100)
            if r['admission_lower_95']!='' else 0 for r in condition]
    yerrhi=[max(0,(float(r['admission_upper_95'])-float(r['admission_mean']))*100)
            if r['admission_upper_95']!='' else 0 for r in condition]
    plt.figure(figsize=(8,4.5))
    plt.bar(labels,means,yerr=[yerrlo,yerrhi],capsize=4)
    plt.ylabel('Mean protected admission (%)')
    plt.title('Phase 5 — Held-out admission (95% seed-level t intervals)')
    plt.ylim(0,max(means+[10])+15);plt.grid(axis='y',alpha=0.25);plt.tight_layout()
    plt.savefig(out/'phase5_heldout_admission.png',dpi=180);plt.close()

    costs=[float(r['selection_ms_mean']) for r in condition]
    plt.figure(figsize=(8,4.5))
    plt.bar(labels,costs)
    plt.ylabel('Mean policy selection (ms/request)')
    plt.title('Phase 5 — Measured decision cost (same computer)')
    plt.grid(axis='y',alpha=0.25);plt.tight_layout()
    plt.savefig(out/'phase5_selection_time.png',dpi=180);plt.close()
    return [str(x) for x in sorted(out.glob('*.png'))]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--training',required=True,type=Path,help='training_history.csv')
    ap.add_argument('--statistics-dir',required=True,type=Path)
    ap.add_argument('--out',required=True,type=Path)
    args=ap.parse_args()
    for f in figures(args.training,args.statistics_dir/'condition_summary.csv',
                     args.statistics_dir/'paired_summary.csv',args.out):print(f)


if __name__=='__main__':main()

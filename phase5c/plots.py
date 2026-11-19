"""Reproducible Phase5C plots; no plot implies MSRP compliance."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phase5.experiment import new_output


def plot_results(stats:Path,out:Path):
    new_output(out)
    with (stats/'condition_summary.csv').open(newline='',encoding='utf8') as f:
        rows=list(csv.DictReader(f))
    if not rows:raise ValueError('empty statistics')
    print('Available condition_summary columns:',list(rows[0]))
    # Compatible with phase5.statistics, which writes one row per policy.
    for value_col,title,outname in (
        ('admission_mean','Mean held-out protected admission','phase5c_admission.png'),
        ('weighted_mean','Mean priority-weighted admission','phase5c_weighted_admission.png'),
        ('selection_ms_mean','Mean policy selection time (ms/request)','phase5c_selection_time.png')):
        if value_col not in rows[0]:continue
        labels=[r['policy'] for r in rows]; values=[float(r[value_col]) for r in rows]
        fig,ax=plt.subplots(figsize=(10,5.5))
        ax.bar(labels,values)
        ax.set_title(title);ax.tick_params(axis='x',rotation=35)
        fig.tight_layout();fig.savefig(out/outname,dpi=180);plt.close(fig)
    return out

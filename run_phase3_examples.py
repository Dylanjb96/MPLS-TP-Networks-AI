#Repeatable Phase 3 demonstration across four frozen Phase 2 configurations.
from pathlib import Path
import json
from railway_tp.topology import TopologyConfig
from phase3.cli import run
root=Path(__file__).resolve().parent
for fname in ('debug_12_dual.json','contrast_12_single.json','primary_60_dual.json',
              'scale_100_dual.json','optional_150_dual.json'):
    cfg=TopologyConfig(**json.loads((root/'config'/fname).read_text(encoding='utf-8')))
    run(cfg,root/'outputs'/'phase3',requests=20,seed=2026,headroom=10)
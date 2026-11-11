#Same network and ordered demand, only headroom varies (exploratory, single seed).
from pathlib import Path
from railway_tp.topology import TopologyConfig
from phase3.cli import run
import json
root=Path(__file__).resolve().parent
cfg=TopologyConfig(**json.loads((root/'config'/'primary_60_dual.json').read_text(encoding='utf-8')))
for h in (0,10,20):
    run(cfg,root/'outputs'/'phase3_sensitivity',requests=50,seed=2026,headroom=h)

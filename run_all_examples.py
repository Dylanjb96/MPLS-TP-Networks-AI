#Recreate required evidence for the 12/60/100-node cases and single-gateway contrast.
import json
from pathlib import Path
from railway_tp.topology import TopologyConfig
from railway_tp.cli import run

ROOT=Path(__file__).resolve().parent
cases=['debug_12_dual.json','contrast_12_single.json','primary_60_dual.json','scale_100_dual.json', 'optional_150_dual.json']
for f in cases:
    cfg=TopologyConfig(**json.loads((ROOT/'config'/f).read_text()))
    run(cfg,ROOT/'outputs',request_count=50)
print('Phase 2 evidence generated. See outputs/*_manifest.json.')
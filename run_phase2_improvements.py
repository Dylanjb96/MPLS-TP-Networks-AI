#Run all six supplementary Phase 2 studies with reproducible evidence.
from pathlib import Path
import argparse
import json
from enhancements.six_improvements import run_all

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='All six Phase 2 additions (topology only, no protected LSPs)')
    parser.add_argument('--out',type=Path,default=Path('outputs/phase2_enhancements'))
    parser.add_argument('--requests-per-split',type=int,default=500)
    parser.add_argument('--samples-per-category',type=int,default=35)
    parser.add_argument('--no-create-missing',action='store_true')
    a=parser.parse_args()
    root=Path(__file__).resolve().parent
    result=run_all(root,a.out,create_missing=not a.no_create_missing,
                   requests_per_split=a.requests_per_split,samples_per_category=a.samples_per_category)
    print(json.dumps(result,indent=2))

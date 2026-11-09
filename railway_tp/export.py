#Portable, documented JSON/CSV exports; all data synthetic.
from __future__ import annotations
from pathlib import Path
import json,csv,hashlib
from dataclasses import asdict
import networkx as nx

def _write_json(path:Path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return path

def export_topology(G:nx.Graph,rings:dict[str,list[int]],path:str|Path)->Path:
    path=Path(path)
    payload={
     'metadata':dict(G.graph),
     'rings':rings,
     'nodes':[{'id':n,**data} for n,data in sorted(G.nodes(data=True))],
     'links':[{'source':u,'target':v,**data} for u,v,data in sorted((min(u,v),max(u,v),d) for u,v,d in G.edges(data=True))],
    }
    return _write_json(path,payload)

def write_table(path:Path,records:list[dict],headers:list[str])->Path:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=headers,extrasaction='ignore')
        w.writeheader();w.writerows(records)
    return path

def checksum(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

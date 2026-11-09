#Synthetic request inventory; no actual railway data or reservation state.
from __future__ import annotations
from dataclasses import asdict,dataclass
import random
import networkx as nx
from .routes import classify_request

@dataclass(frozen=True)
class ServiceRequest:
    request_id: str
    source: int
    destination: int
    bandwidth_mbps: int
    service_class: str
    priority_weight: int
    endpoint_category: str

#Explicit, synthetic assumptions — not derived from engineer screenshots.
CLASSES={
 'critical':{'bandwidth_mbps':[10,20,50],'priority_weight':5},
 'operational':{'bandwidth_mbps':[50,100,200],'priority_weight':3},
 'non_critical':{'bandwidth_mbps':[100,200,400],'priority_weight':1},
}

def generate_requests(G:nx.Graph,count:int,seed:int=42)->list[ServiceRequest]:
    if count<0 or G.number_of_nodes()<2: raise ValueError('invalid count or graph')
    rng=random.Random(seed); nodes=sorted(G.nodes); out=[]
    for i in range(count):
        src,dst=rng.sample(nodes,2)
        name=rng.choices(list(CLASSES),weights=[2,5,3])[0]
        rule=CLASSES[name]
        out.append(ServiceRequest(f'R{i+1:04d}',src,dst,
                  rng.choice(rule['bandwidth_mbps']),name,rule['priority_weight'],
                  classify_request(G,src,dst)))
    return out

def requests_to_dict(items:list[ServiceRequest])->list[dict]:
    return [asdict(item) for item in items]

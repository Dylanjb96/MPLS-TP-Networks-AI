#Phase 2 structural acceptance tests (not protocol conformance tests).
import json
import subprocess
import sys
from pathlib import Path
import pytest
import networkx as nx
from railway_tp.topology import (TopologyConfig,make_topology,ring_is_valid_cycle,
    link_failure_connectivity,node_failure_connectivity,validate_topology)
from railway_tp.routes import candidate_paths,ring_arcs,classify_request
from railway_tp.requests import generate_requests
from railway_tp.export import export_topology

@pytest.mark.parametrize('size,subrings',[(12,1),(20,1),(60,4),(100,6),(150,6)])
def test_sizes_and_cycles(size,subrings):
    g,rings=make_topology(TopologyConfig(nodes=size))
    assert g.number_of_nodes()==size
    assert len(rings)==subrings+1
    assert validate_topology(g,rings)['all_rings_valid_cycles']
    assert nx.is_connected(g)

@pytest.mark.parametrize('mode,gw_count', [('single',1),('dual',2)])
def test_gateway_modes(mode,gw_count):
    g,rings=make_topology(TopologyConfig(nodes=12,gateway_mode=mode))
    assert len(set(rings['main']) & set(rings['sub_1']))==gw_count
    assert all(row['connected'] for row in link_failure_connectivity(g))
    gw=sorted(set(rings['main'])&set(rings['sub_1']))[0]
    node_fail={row['failed_node']:row for row in node_failure_connectivity(g)}
    assert node_fail[gw]['connected']==(mode=='dual')

def test_shared_edge_counted_once():
    g,rings=make_topology(TopologyConfig(nodes=12,gateway_mode='dual'))
    shared=[(u,v,d) for u,v,d in g.edges(data=True) if len(d['rings'])>1]
    assert len(shared)==1
    assert shared[0][2]['capacity_mbps']==10000
    assert all(d['reserved_mbps']==0 for _,_,d in g.edges(data=True))

def test_reproducible_seed_and_seed_changes_gateway_rotation():
    a,_=make_topology(TopologyConfig(nodes=60,seed=42))
    b,_=make_topology(TopologyConfig(nodes=60,seed=42))
    assert sorted(a.edges())==sorted(b.edges())
    assert nx.get_node_attributes(a,'rings')==nx.get_node_attributes(b,'rings')
    assert len(a.nodes)==60
    c,_=make_topology(TopologyConfig(nodes=60,seed=43))
    assert sorted(n for n,d in a.nodes(data=True) if d['gateway']) != sorted(n for n,d in c.nodes(data=True) if d['gateway'])

def test_ring_arcs_follow_ring_edges():
    g,rings=make_topology(TopologyConfig(nodes=12))
    a,b=rings['sub_1'][1:3]
    arcs=ring_arcs(rings,'sub_1',a,b)
    assert len(arcs)==2
    assert arcs[0]!=arcs[1]
    for path in arcs:
        assert path[0]==a and path[-1]==b
        assert all(ring_is_valid_cycle(g,rings['sub_1'],'sub_1') and 'sub_1' in g.edges[u,v]['rings'] for u,v in zip(path,path[1:]))

def test_route_categories_and_k_paths():
    g,rings=make_topology(TopologyConfig(nodes=60))
    sub1=[n for n in rings['sub_1'] if 'main' not in g.nodes[n]['rings']]
    sub2=[n for n in rings['sub_2'] if 'main' not in g.nodes[n]['rings']]
    assert classify_request(g,sub1[0],sub1[1])=='within-subring'
    assert classify_request(g,sub1[0],sub2[0])=='subring-to-subring'
    assert classify_request(g,0,sub1[0])=='main-to-subring'
    for x,y in [(0,1),(sub1[0],sub2[0]),(0,sub1[0])]:
        paths=candidate_paths(g,x,y,5)
        assert 1<=len(paths)<=5
        assert all(p[0]==x and p[-1]==y for p in paths)

def test_request_reproducibility():
    g,_=make_topology(TopologyConfig())
    a=generate_requests(g,50,10);b=generate_requests(g,50,10)
    assert a==b and all(0<r.bandwidth_mbps<=400 for r in a)
    assert all(r.source!=r.destination for r in a)

def test_export_json_and_inventory_metadata(tmp_path):
    g,r=make_topology(TopologyConfig())
    output=export_topology(g,r,tmp_path/'topology.json')
    data=json.loads(output.read_text())
    assert len(data['nodes'])==12
    assert len(data['links'])==g.number_of_edges()
    assert all('physical_link_id' in e for e in data['links'])
    assert len({e['physical_link_id'] for e in data['links']})==len(data['links'])

def test_cli_builds_all_evidence_files(tmp_path):
    root=Path(__file__).resolve().parents[1]
    cmd=[sys.executable,'-m','railway_tp.cli','--config',str(root/'config/debug_12_dual.json'),
         '--requests','12','--out',str(tmp_path)]
    subprocess.run(cmd,cwd=root,check=True,capture_output=True,text=True)
    files=list(tmp_path.iterdir())
    assert len(files)==12
    m=json.loads(next(tmp_path.glob('*_manifest.json')).read_text())
    assert m['passed_structural_gate']
    assert m['request_count']==12
    assert len(m['output_files_sha256'])==11

@pytest.mark.parametrize('invalid',[
 {'nodes':11}, {'nodes':60,'subring_count':99}, {'nodes':12,'main_nodes':11},
 {'gateway_mode':'third'}, {'sub_capacity_mbps':0}, {'length_km':0},
])
def test_bad_input_rejected(invalid):
    with pytest.raises(ValueError):
        g,r=make_topology(TopologyConfig(**invalid))


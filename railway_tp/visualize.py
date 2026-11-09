#Schematic physical graph plot; coordinates DO NOT reflect geography.
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

def save_topology_figure(G:nx.Graph,out:str|Path)->Path:
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    pos=nx.spring_layout(G,seed=G.graph.get('seed',42),iterations=140)
    fig,ax=plt.subplots(figsize=(12.4,7.5))
    colors=['#E5A34A' if d['gateway'] else '#235783' if 'main' in d['rings'] else '#469A91' for _,d in G.nodes(data=True)]
    widths=[3.4 if len(d['rings'])>1 else 2.2 if 'main' in d['rings'] else 1.25 for _,_,d in G.edges(data=True)]
    nx.draw_networkx_edges(G,pos,ax=ax,width=widths,edge_color='#8293A4',alpha=.84)
    nx.draw_networkx_nodes(G,pos,ax=ax,node_color=colors,node_size=520 if len(G)<=20 else 105,
                           edgecolors='white',linewidths=1)
    if len(G)<=20:
        nx.draw_networkx_labels(G,pos,ax=ax,font_size=9,font_color='white',font_weight='bold')
    ax.set_title(f'Phase 2 synthetic physical network | {G.number_of_nodes()} nodes | {G.graph["gateway_mode"]} gateways',
                 fontsize=15,fontweight='bold',pad=16)
    ax.text(.015,.02,'MAIN nodes    SUB-RING nodes    GATEWAYS  |  Shared physical edges shown heavier\n'
             'Schematic topology only. Graph reachability is not MPLS-TP service protection.',
             transform=ax.transAxes,fontsize=9,color='#344256',va='bottom')
    ax.axis('off');fig.tight_layout();fig.savefig(out,dpi=155,bbox_inches='tight');plt.close(fig)
    return out

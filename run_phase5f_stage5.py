"""Phase 5F Stage 5 locked final heldout runner; no train/retrain commands."""
from __future__ import annotations
import argparse,json
from phase5f_stage5.workflow import preflight,devcheck,final_run,verify_results


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('preflight',help='verify frozen selection/model/source SHA; NO heldout access')
    dev=sub.add_parser('devcheck',help='replay 60-node validation-seed engineering case only')
    dev.add_argument('--out',required=True)
    final=sub.add_parser('final',help='one-time preregistered heldout 8-cell factorial')
    final.add_argument('--out',required=True)
    final.add_argument('--confirm-final',action='store_true',help='explicitly authorize opening reserved seeds')
    final.add_argument('--resume',action='store_true',help='recover interrupted exact frozen plan')
    verify=sub.add_parser('verify',help='read-only integrity verification after final is complete')
    verify.add_argument('--root',required=True)
    a=p.parse_args(argv)
    if a.command=='preflight':
        result=preflight()
    elif a.command=='devcheck':
        result=devcheck(a.out)
    elif a.command=='final':
        result=final_run(a.out,resume=a.resume,confirm_final=a.confirm_final)
    elif a.command=='verify':
        result=verify_results(a.root)
    else:raise AssertionError(a.command)
    print(json.dumps(result,indent=2,sort_keys=True))
    return result

if __name__=='__main__':main()

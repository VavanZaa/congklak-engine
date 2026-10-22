"""Metadata preparation and final whole-footprint reservation publication."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from congklak import config,footprint_screen_data as data,footprint_screen_reservations as reservations

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('pool','reservations','completed'))
    args=parser.parse_args()
    if args.command=='pool':body=data.prepare_pool()
    else:
        from congklak import footprint_screen_worker as worker,footprint_screen_review as review
        snapshot=worker.completed_snapshot();audit=review.completed()
        body=(reservations.prepare(snapshot,audit) if args.command=='reservations' else reservations.completed())
        body={k:v for k,v in body.items() if k not in ('report','roots','components','quotient_components','family_components')}
    print(config.canonical(body))

if __name__=='__main__':main()

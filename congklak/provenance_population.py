"""Prospectively declared custom-position population; importing draws nothing.

This file defines the counter constructor, not a reservation or training job.
The complete spec and code must receive independent review before its first
constructor call.  Metadata probes and graph registration belong to a separate
source-bound, recoverable producer; this module never searches or trains.
"""
import hashlib
from . import config, neurodiag_data as metadata

SCHEMA='prospective-custom-ownership-population-v1'
SEED=20261016
HASH_SALT='ownership-implementation-v1/synthetic-metadata-population/20261016/v1/'
MAX_DRAWS=16384
CELLS=metadata.CELLS
ARENA_SALT='ownership-protocol-v1/arena/20261016'
FIT_SALT='ownership-implementation-v1/fit/20261016'


def specification():
    return dict(schema=SCHEMA,version=1,seed=SEED,hash_salt=HASH_SALT,
        max_draws=MAX_DRAWS,counter_range=[0,MAX_DRAWS-1],cells=list(CELLS),
        counter_cell='CELLS[counter %8]; exactly2048 declared counters per cell',
        active_seeds='25+H(active)%25 for25-49 cells;50+H(active)%25 for50-74 cells',
        actor='last cell character; exact raw actor, no actor canonicalization',
        occupied_pits='actor2+H(own_positive)%6; opponent1+H(opponent_positive)%7; SHA-ranked distinct pit indices',
        own_total='own_positive+H(own_total)%(active-own_positive-opponent_positive+1)',
        pit_allocation='one seed per occupied pit; each remaining own/opponent seed goes to H(side/index)%occupied_count',
        stores='stored=98-active; store0=clamp(stored//2+H(store_offset)%13-6,0,stored); store1=stored-store0',
        hash_integer='H(label)=little-endian integer of SHA256(ASCII(hash_salt+str(seed)+/+str(counter)+/+label))',
        population='metadata-defined legal stable custom positions, mechanism-filtered after all legal action probes',
        fixed_scan='Process all16384 counters regardless of early quota completion; no adaptive seed/salt or sample-size extension',
        new_origin_id='synthetic-ownership-population-v1/'+str(SEED)+'/counter formatted as five decimal digits',
        origin_closure='root plus every normalized legal-action child consulted for the metadata mechanism classifier',
        mechanism='Pinned pilot._mechanism precedence pass, capture, bonus, ordinary, using every legal action; target mechanism is metadata filter only',
        require_normalized_root=True,min_legal_choices=2,require_abs_store_difference_less_than_active=True,
        graph='Merge actual new family/rawstate/quotient incidences with certified full known-source graph; retain every>=25 alias and all origin vertices;<=24 ledger retained separately',
        duplicate_policy='Keep actual cross-candidate bridges; joint matching may assign at most one role/root per resulting new component. Do not split or arbitrarily discard an origin to fill quotas',
        prior_graph_conflicts='A component touching any frozen support or protected/consumed origin cannot serve new fit/arena; old512 support pool remains unchanged',
        acquisition_policy='Every counter/root/probe and all rejection reasons remain in immutable journal; metadata acquisition recorded separately from analysis/optimizer exposure',
        rejection_reasons=['normalization changes root','not at least two legal choices','mathematically decided store margin',
            'metadata mechanism differs from declared counter cell','known support/protected high-domain component',
            'opaque unresolved high-domain component','joint quota/component shortage'],
        rejected_origins='Retained with observed incidences and exposure tags; rejection does not create a replacement chosen by target/model/outcome',
        unselected_origins='Retain incidence/exposure and optimizer-blocked development status; no silently enlarged future training support',
        no_features=True,no_model_forward=True,no_search=True,no_targets=True,no_training=True,no_production_games=True,
        roles=dict(existing_support_roots=512,new_fit_roots=128,fit_per_cell=16,fit_case_subset=32,cases_per_cell=4,
            new_arena_roots=24,arena_per_cell={'50-74/ordinary/actor0':2,'50-74/ordinary/actor1':2,
                '25-49/ordinary/actor0':2,'25-49/ordinary/actor1':2,'25-49/bonus/actor0':4,
                '25-49/bonus/actor1':4,'25-49/capture/actor0':4,'25-49/capture/actor1':4}),
        selection=dict(arena_salt=ARENA_SALT,fit_salt=FIT_SALT,metadata_only=True,
            joint_component_matching_required=True,source_graph_before_selection=True,
            targets_predictions_outcomes_uninspected=True,shortage='HOLD without shrinking quotas, extending draw range, changing salt or adding fillers'),
        limitations=['No standard-opening reachability or natural-play representativeness claim.',
            'Balanced stores and variable occupancy are deliberate population restrictions.',
            'Deterministic salted selection is not a random sample of all Congklak positions.',
            'New metadata reservations establish next-update/known-ownership separation, not historically unseen generalization.',
            'Old arena remains invalid for strength; candidate remains quarantined; no labels or training are authorized by this constructor.'],
        first_draw_requires='Independent review binds this complete specification, constructor source, certified graph and native classifier build before counter0 is evaluated',
        candidate_constructor_calls_at_declaration=0,production_admission=False)


def candidate(counter):
    """Pure lossless root construction; only call after source/spec review."""
    if type(counter) is not int or not 0<=counter<MAX_DRAWS:
        raise ValueError('Synthetic population counter outside predeclared range')
    cell=CELLS[counter%len(CELLS)];actor=int(cell[-1])
    def integer(label):
        text=HASH_SALT+str(SEED)+'/'+str(counter)+'/'+label
        return int.from_bytes(hashlib.sha256(text.encode('ascii')).digest(),'little')
    active=(50 if cell.startswith('50') else 25)+integer('active')%25
    own_positive=2+integer('own_positive')%6
    opponent_positive=1+integer('opponent_positive')%7
    own_total=own_positive+integer('own_total')%(active-own_positive-opponent_positive+1)
    pits=[[0]*7,[0]*7]
    for side,occupied,total in ((actor,own_positive,own_total),(1-actor,opponent_positive,active-own_total)):
        name='own' if side==actor else 'opponent'
        indices=sorted(range(7),key=lambda pit:(integer(name+'_index/'+str(pit)),pit))[:occupied]
        for pit in indices:pits[side][pit]=1
        for index in range(total-occupied):pits[side][indices[integer(name+'_seed/'+str(index))%occupied]]+=1
    stored=98-active;first=max(0,min(stored,stored//2+integer('store_offset')%13-6))
    state=config.position(dict(pits=pits,stores=[first,stored-first],controller=actor))
    return dict(counter=counter,cell=cell,family='synthetic-ownership-population-v1/'+str(SEED)+'/'+f'{counter:05d}',
                state=state,state_key=metadata.common.rawkey(state),active_seeds=active,
                legal_mask=sum(1<<pit for pit,count in enumerate(pits[actor]) if count),
                synthetic_custom=True,opening_reachability_claimed=False)

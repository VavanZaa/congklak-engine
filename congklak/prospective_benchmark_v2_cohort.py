"""Fixed metadata population with a complete future high-domain certificate.

These legal stable custom starts contain exactly25 active seeds. Every legal
move for either hypothetical actor passes that actor's store, so every completed
child has at most24 active seeds. No model, search, feature encoder, or teacher
is invoked by this module. Standard-opening reachability is not claimed.
"""
import hashlib
from . import config, neurodiag_common as common
from . import value_run_arena as literal

SCHEMA='prospective-benchmark-v2-boundary-cohort'
SEED=20261017
SALT='prospective-benchmark-v2/legal-custom25/20261017/v1/'
MAX_DRAWS=4096
QUOTA=8
OPTIONAL_DRAWS=32
OPTIONAL_QUOTA=4
OPTIONAL_STATE_CAP=20000


def specification():
    return dict(schema=SCHEMA,seed=SEED,salt=SALT,counters=[0,MAX_DRAWS-1],complete_fixed_scan=True,
        active_seeds=25,total_seeds=98,actor='counter modulo2',
        occupied='independently2or3pits per side; indices ranked by SHA256(side/index), then index',
        minimum_pit_seeds='8-localpitindex; every occupied pit crosses own store before its initial hand empties',
        minimum_sum_above25='retain metadata rejection without constructing a position',
        remaining_seeds='each of25-minus-minimumsum seeds assigned by SHA256(extra/index) modulo occupied slots',
        stores='store0=36+SHA256(store_offset) modulo7 minus3; store1=73-store0',
        mechanism='actual normalized actor: all legal actions must be ordinary, no capture or bonus',
        complete_structural_certificate='both rows nonempty; for BOTH actor0/actor1 EVERY legal completed child has active<=24; every complete stable intermediate is input, normalized input or final child only',
        high_domain='exact root board under both controller values, canonicalized by player/row exchange',
        protected_filter='reject if either root quotient appears in ANY immutable prior known-component map or any retained old query/global claim footprint',
        selected='first8 distinct raw root states in ascending counter order AFTER all4096 metadata draws finish',
        units='all roots share one joint prospective evaluation owner; no ownership-independent root count or cluster inference',
        cohort_shortage='HOLD with all counters retained; no larger counter range, seed change, filler or native-result selection',
        historical_conditioning=True,historically_unseen=False,opening_reachability_claimed=False,
        natural_play_generalization_claimed=False,conditional_descriptive_benchmark=True,
        labels=False,features=False,model_forwards=False,searches=False,training=False,
        branch_coverage='learned value branch may run at root25 only; all subsequent states shared<=24; no ordinary midgame strength claim',
        optional_high_closure=dict(counters=[0,OPTIONAL_DRAWS-1],active='26+counter modulo7',quota=OPTIONAL_QUOTA,
            maximum_high_states_per_root=OPTIONAL_STATE_CAP,frontier='all normalized and pre-normalization hypothetical actors, every legal move until<=24',
            selected='first4 complete protected-safe roots after all32 draws; cap/protected conflicts retained; shortage is separate and never filled with25 roots'))


def candidate(counter,lane='boundary'):
    maximum=MAX_DRAWS if lane=='boundary' else OPTIONAL_DRAWS
    if lane not in ('boundary','optional') or type(counter) is not int or not 0<=counter<maximum:raise ValueError('Counter outside fixed cohort population')
    active=25 if lane=='boundary' else 26+counter%7
    def integer(label):return int.from_bytes(hashlib.sha256((SALT+('optional/' if lane=='optional' else '')+str(SEED)+'/'+str(counter)+'/'+label).encode()).digest(),'little')
    slots=[]
    for side in (0,1):
        occupied=2+integer('occupied/'+str(side))%2
        indices=sorted(range(7),key=lambda index:(integer('pit/'+str(side)+'/'+str(index)),index))[:occupied]
        slots.extend((side,index) for index in indices)
    minimum=sum(8-index for side,index in slots)
    metadata=dict(counter=counter,lane=lane,origin='prospective-benchmark-v2/'+lane+'/'+f'{counter:04d}',
        occupied_slots=[list(v) for v in slots],minimum_active=minimum,constructor_rejected=minimum>active)
    if minimum>active:return metadata
    pits=[[0]*7,[0]*7]
    for side,index in slots:pits[side][index]=8-index
    for index in range(active-minimum):
        side,pit=slots[integer('extra/'+str(index))%len(slots)];pits[side][pit]+=1
    stored=98-active;first=stored//2+integer('store_offset')%7-3
    state=config.position(dict(pits=pits,stores=[first,stored-first],controller=counter%2))
    return dict(metadata,state=state,state_key=common.rawkey(state),active_seeds=active)


def quotient(state):
    a=bytes(state['pits'][0]);b=bytes(state['pits'][1]);actor=state['controller']
    return min(a+b+bytes([actor]),b+a+bytes([1-actor])).hex()


def certify(state,reference=None):
    """Literal exhaustive first frontier; later high states are impossible."""
    state=config.position(state);need=common.require
    need(sum(map(sum,state['pits']))==25 and all(any(row) for row in state['pits']),
         'Boundary certificate requires both nonempty rows and exactly25 active seeds')
    ref=reference or literal._literal();probes=[];ordinary=True;roots=[]
    for actor in (0,1):
        root=dict(state,controller=actor);normal=literal.normalize(root,ref)
        need(normal['state']==root and not normal['terminal'],'Hypothetical root is not stable normalized')
        roots.append(dict(state=root,state_key=common.rawkey(root),quotient=quotient(root)))
        for action in normal['legal']:
            transition=literal.transition(root,action,ref);child=transition['after'];active=sum(map(sum,child['pits']))
            need(active<=24 and sum(map(sum,child['pits']))+sum(child['stores'])==98,
                 'A hypothetical legal action escapes the certified shared-low future')
            probes.append(dict(actor=actor,action=action,transition=transition,child_active=active,
                child_state_key=common.rawkey(child),child_quotient=quotient(child)))
            if actor==state['controller'] and (transition['captured'] or transition['bonus']):ordinary=False
    return dict(schema=SCHEMA+'-structural-certificate',root_actor_states=roots,probes=probes,
        root_active=25,maximum_future_child_active=max(p['child_active'] for p in probes),ordinary=ordinary,
        future_high_states='none after any completed first action; active seeds never increase',
        universal_legal_frontier=True,both_hypothetical_actors=True,model_or_outcome_consulted=False)


def certify_high_closure(state,forbidden,reference=None,cap=OPTIONAL_STATE_CAP):
    """Optional exact finite closure, rejecting rather than truncating a proof."""
    ref=reference or literal._literal();queue=[config.position(state)];seen={};edges=[];conflicts=[];cursor=0
    while cursor<len(queue):
        board=queue[cursor];cursor+=1
        for actor in (0,1):
            root=dict(board,controller=actor);key=common.rawkey(root)
            if key in seen:continue
            if len(seen)>=cap:return dict(complete=False,reason='high-state cap',states=list(seen.values()),edges=edges,conflicts=conflicts)
            q=quotient(root);seen[key]=dict(state=root,state_key=key,quotient=q)
            cause=forbidden(q)
            if cause:
                conflicts.append(dict(quotient=q,cause=cause))
                return dict(complete=False,reason='protected prior high state',states=list(seen.values()),edges=edges,conflicts=conflicts)
            normal=literal.normalize(root,ref)
            for action in normal['legal']:
                transition=literal.transition(root,action,ref);child=transition['after'];active=sum(map(sum,child['pits']))
                edges.append(dict(from_state_key=key,action=action,child_state_key=common.rawkey(child),child_active=active))
                if active>=25:queue.append(child)
    return dict(complete=True,reason='complete universal high closure',states=list(seen.values()),edges=edges,conflicts=[])

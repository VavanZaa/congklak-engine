"""Source-witnessed provenance reconstruction; no target/model/search access.

The caller durably commits bounded units before summarizing.  True raw
incidences and opaque legacy union constraints are different record types.
Every known >=25 lineage vertex, including public openings, is retained.
"""
from collections import Counter, defaultdict
from contextlib import closing
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from . import config, neurodiag_common as common, neurodiag_data as metadata
from . import value_run_arena as arena
from . import valuepilot_reservations as legacy
from . import ownership_domain_projection as domain

ROOT = common.ROOT
SCHEMA = 'source-witnessed-provenance-v1'
DEFAULT = ROOT / 'build/ownership-implementation-v1/provenance'
INVENTORY = ROOT / 'data/neural-value-pilot-v2/reservations/inventory.json'
SELECTION = ROOT / 'data/neural-value-pilot-v2/reservations/selection.json'
CURRENT = ROOT / 'data/value-run-v1/arena'
DATASETS = ('pilot-10000', 'stronger-10000-v2', 'stronger-10000')
JOURNALS = ('deeper-9000-v2', 'ordinary-data-pilot-v1', 'natural-study-v1',
    'midgame-data-pilot-v1', 'midgame-label-pilot-v1',
    'ordinary-selfplay-pilot-v1', 'selfplay-training-pilot-v1',
    'horizon-probe-v1', 'horizon-finish-v1', 'ordinary-value-labels-v1',
    'divergence-teacher-v1', 'allocation-weighting-v1',
    'stratum-weighting-v1', 'qstability-v1', 'allocation-trace-v1')
JSON_INPUTS = (INVENTORY, SELECTION,
    ROOT/'data/ordinary-development-v1/candidates.json',
    ROOT/'build/quality-data.json', ROOT/'build/ordinary-value-rows.json',
    ROOT/'build/value-pilot-rows.json', ROOT/'build/neurodiag-panel.json',
    ROOT/'build/phase6-protected-registry.json')
SOURCES = ('congklak/provenance_reconstruction.py', 'tools/reconstruct-provenance.py',
    'congklak/ownership_domain_projection.py', 'congklak/neurodiag_data.py',
    'congklak/neurodiag_common.py', 'congklak/divergence_study_common.py',
    'congklak/config.py', 'congklak/value_run_arena.py',
    'congklak/valuepilot_reservations.py', 'tests/reference.py')
MODELS = (ROOT/'data/quality-train-v1/model.json',ROOT/'data/value-training-v1/training/model.json')
ZERO = dict(model_forwards=0, feature_simulations=0, searches=0, new_games=0,
            new_teacher_labels=0, optimizer_steps=0, optimizer_draws=0,
            target_values_read=0)
need, read, sha = common.require, common.read, common.sha


def rel(path): return Path(path).resolve().relative_to(ROOT).as_posix()
def dbpath(name): return ROOT / 'data' / name / 'work.sqlite3'
def ro(path): return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro', uri=True)


def _tables(db):
    return {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _hash_text(text, digest):
    need(hashlib.sha256(text.encode('utf-8')).hexdigest() == digest,
         'Source SQLite row hash differs')


def _body_rows(db, table, key, offset, limit, fields):
    """Project metadata in SQLite; teacher/value fields are never decoded."""
    query = ('SELECT '+key+',hash,body,' + ','.join("json_extract(body,'$."+f+"')" for f in fields)
             + ' FROM '+table+' ORDER BY '+key+' LIMIT ? OFFSET ?')
    for ident, digest, text, *values in db.execute(query, (limit, offset)):
        _hash_text(text, digest)
        yield ident, digest, dict(zip(fields, values))


def _source(task, table=None, key=None, digest=None, pointer=None, **extra):
    out = dict(path=task['source'], sha256=task['source_sha256'])
    if table is not None: out.update(table=table, key=key, row_sha256=digest)
    if pointer is not None: out['pointer'] = pointer
    out.update(extra); return out


def _edge(family, state, witness):
    state = config.position(state)
    need(type(family) is str and family, 'Actual incidence lacks an origin family')
    q = metadata.quotient(state); a = domain.quotient_active(q)
    return dict(family=family, raw_state_key=common.rawkey(state), quotient=q,
                active_seeds=a, domain='owned' if a >= 25 else 'shared', witness=witness)


def _marker(kind, value, witness, reason, optimizer=False, fit=False, arena=False,
            ancestral_training=False, acquisition=False):
    need(kind in ('family', 'quotient') and type(value) is str and value,
         'Invalid provenance role marker')
    return dict(kind=kind, value=value, witness=witness, reason=reason,
                optimizer_blocked=optimizer, fit_blocked=fit, arena_blocked=arena,
                ancestral_training=ancestral_training, acquisition_exposed=acquisition)


def plan():
    need('torch' not in sys.modules, 'Provenance reconstruction forbids Torch')
    inv = read(INVENTORY)
    files = set(JSON_INPUTS) | {ROOT/p for p in inv['source_inputs']}
    files.update(ROOT/p for p in SOURCES)
    files.update(MODELS)
    files.update((CURRENT/'identity.json', CURRENT/'head.json'))
    tasks = []; source_hashes={}
    def task(kind, source, **fields):
        source = Path(source); files.add(source)
        if source not in source_hashes:source_hashes[source]=sha(source)
        tasks.append(dict(index=len(tasks), kind=kind, source=rel(source),
                          source_sha256=source_hashes[source], **fields))
    for name in DATASETS:
        path = dbpath(name)
        with closing(ro(path)) as db:
            families = db.execute('SELECT count(*) FROM families').fetchone()[0]
            origins = db.execute('SELECT count(*) FROM origins').fetchone()[0]
        for offset in range(0, families, 128):
            task('dataset_families', path, offset=offset, limit=min(128, families-offset),
                 expected_total=families, family_key='id', dataset=name,
                 quarantined_target_payloads_accessed=False)
        for offset in range(0, origins, 512):
            task('dataset_origins', path, offset=offset, limit=min(512, origins-offset),
                 expected_total=origins, dataset=name)
    missing = []
    for name in JOURNALS:
        path = dbpath(name)
        if not path.exists():
            missing.append(dict(name=name, status='optional later journal absent')); continue
        with closing(ro(path)) as db:
            tables = _tables(db)
            for table in ('roots', 'families', 'sp_rows', 'sp_games', 'sp_decisions'):
                if table not in tables: continue
                count = db.execute('SELECT count(*) FROM '+table).fetchone()[0]
                size = 64 if table == 'families' else 512
                for offset in range(0, count, size):
                    task('journal', path, table=table, offset=offset,
                         limit=min(size, count-offset), expected_total=count, journal=name)
    resolver_inputs=common.pins([dbpath('deeper-9000-v2'),dbpath('pilot-10000')])
    for filename in legacy.ARENAS:
        path = ROOT/filename
        with closing(ro(path)) as db:
            count = db.execute('SELECT count(*) FROM games').fetchone()[0]
        for offset in range(0, count, 32):
            task('historical_arena', path, offset=offset, limit=min(32,count-offset), expected_total=count,
                 origin_source_inputs=resolver_inputs)
    path=dbpath('allocation-trace-v1')
    with closing(ro(path)) as db:
        count=db.execute('SELECT count(*) FROM steps').fetchone()[0]
    for offset in range(0,count,256):
        task('recorded_search_trace',path,offset=offset,limit=min(256,count-offset),expected_total=count)
    head = read(CURRENT/'head.json')
    for entry in head['games']+head['decisions']: files.add(CURRENT/entry['name'])
    for offset in range(0, len(head['games']), 16):
        task('current_arena', CURRENT/'head.json', offset=offset, limit=min(16,len(head['games'])-offset),
             expected_total=len(head['games']), identity_pin=common.pins([CURRENT/'identity.json']))
    task('json_metadata', INVENTORY, metadata_pins=common.pins(JSON_INPUTS))
    return dict(schema=SCHEMA+'-plan', tasks=tasks, inputs=common.pins(files),
                source_catalog=sorted({t['source'] for t in tasks}),
                optional_absent_journals=missing, known_material_sources_declared=True,
                historical_internal_leaf_ancestry='unknown and not reconstructed',
                no_target_or_model_access=True, **ZERO)


def _trajectory(edges, family, start, moves, witness, ref, final=None):
    p = arena.normalize(start, ref)['state']; transitions = 0
    for choice, action in enumerate([*moves, None]):
        edges.append(_edge(family, p, dict(witness, trajectory_choice=choice)))
        if action is None: break
        need(action in arena.normalize(p, ref)['legal'], 'Source lineage action illegal')
        after = arena.transition(p, action, ref)['after']
        after = arena.normalize(after, ref)['state']
        domain.mass_transition(p, after); p = after; transitions += 1
    if final is not None: need(p == final, 'Source lineage final state differs')
    return transitions


@lru_cache(maxsize=2)
def _origin_map(deeper_sha, pilot_sha):
    need(sha(dbpath('deeper-9000-v2')) == deeper_sha and sha(dbpath('pilot-10000')) == pilot_sha,
         'Historical origin resolution source changed')
    with closing(ro(dbpath('deeper-9000-v2'))) as db:
        out = defaultdict(set)
        for key,family in db.execute("SELECT state,json_extract(body,'$.family') FROM roots"):out[key].add(family)
    with closing(ro(dbpath('pilot-10000'))) as db:
        for key, family in db.execute("SELECT state,family FROM selected WHERE split='development'"):
            out[bytes(key).hex()].add(family)
    return {key:sorted(families) for key,families in out.items()}


def unit(task):
    need('torch' not in sys.modules, 'Provenance unit forbids Torch')
    need(sha(ROOT/task['source']) == task['source_sha256'], 'Provenance task source changed')
    if task['kind'] == 'json_metadata': return json_unit(task)
    if task['kind'] == 'current_arena': return current_unit(task)
    edges = []; markers = []; processed = 0; transitions = 0; families = []; opaque_constraints=[]
    ref = arena._literal()
    with closing(ro(ROOT/task['source'])) as db:
        kind = task['kind']; offset, limit = task['offset'], task['limit']
        if kind == 'dataset_origins':
            need(db.execute('SELECT count(*) FROM origins').fetchone()[0] == task['expected_total'],
                 'Dataset origin cardinality changed')
            sql = ('SELECT hex(o.state),o.family,o.step,o.swapped,c.position FROM origins o '
                   'JOIN candidates c ON c.state=o.state ORDER BY o.state,o.family,o.step,o.swapped LIMIT ? OFFSET ?')
            for key, family, step, swapped, text in db.execute(sql,(limit,offset)):
                state = json.loads(text); need(common.rawkey(state) == key.lower(), 'Dataset candidate key differs')
                row = dict(state_key=key.lower(),family=family,step=step,swapped=swapped,state=state)
                witness = _source(task, 'origins JOIN candidates', key=[key.lower(),family,step,swapped],
                                  digest=config.digest(row), row_digest_kind='projected_metadata')
                edges.append(_edge(family,state,witness)); processed += 1
        elif kind == 'dataset_families':
            need(db.execute('SELECT count(*) FROM families').fetchone()[0] == task['expected_total'],
                 'Dataset family cardinality changed')
            query = ("SELECT id,hash,body,split,json_extract(body,'$.start'),json_extract(body,'$.moves') "
                     'FROM families ORDER BY id LIMIT ? OFFSET ?')
            for family, digest, text, split, start, moves in db.execute(query,(limit,offset)):
                _hash_text(text,digest); witness = _source(task,'families',family,digest)
                transitions += _trajectory(edges,family,json.loads(start),json.loads(moves),witness,ref)
                families.append(family); processed += 1
                if split != 'train': markers.append(_marker('family',family,witness,'ancestral dataset development origin',optimizer=True,acquisition=True))
        elif kind == 'journal':
            table = task['table']; count = db.execute('SELECT count(*) FROM '+table).fetchone()[0]
            need(count == task['expected_total'], 'Journal metadata cardinality changed')
            if table == 'families':
                fields = ['family','start','moves','final','candidates','cohort']
                for ident,digest,row in _body_rows(db,table,'idx',offset,limit,fields):
                    family = row['family']; witness = _source(task,table,ident,digest)
                    transitions += _trajectory(edges,family,json.loads(row['start']),json.loads(row['moves']),witness,ref,json.loads(row['final']))
                    for index,candidate in enumerate(json.loads(row['candidates'])):
                        edges.append(_edge(family,candidate['state'],dict(witness,metadata_pointer='/candidates/'+str(index)+'/state')))
                    if row['cohort'] != 'training_candidate':
                        markers.append(_marker('family',family,witness,'source validation-family cohort',optimizer=True,fit=True,arena=True,acquisition=True))
                    processed += 1; families.append(family)
            elif table in ('roots','sp_rows'):
                key = 'state' if table == 'roots' else 'id'
                fields = ['state','state_key','family','cohort','split']
                for ident,digest,row in _body_rows(db,table,key,offset,limit,fields):
                    state = json.loads(row['state']); family = row['family']; witness = _source(task,table,ident,digest,pointer='/state')
                    need(row['state_key'] is None or common.rawkey(state)==row['state_key'], 'Journal raw root key differs')
                    edges.append(_edge(family,state,witness)); processed += 1
                    if task['journal'] in ('horizon-probe-v1','horizon-finish-v1','ordinary-value-labels-v1',
                                             'divergence-teacher-v1','allocation-weighting-v1','stratum-weighting-v1','qstability-v1','allocation-trace-v1'):
                        markers.append(_marker('family',family,witness,'previously requested/inspected diagnostic root',optimizer=True,fit=True,arena=True,acquisition=True))
            elif table == 'sp_games':
                for ident,digest,row in _body_rows(db,table,'id',offset,limit,['family','start','position']):
                    witness = _source(task,table,ident,digest); family = row['family']
                    edges.extend(_edge(family,json.loads(row[field]),dict(witness,metadata_pointer='/'+field)) for field in ('start','position'))
                    processed += 1
            elif table == 'sp_decisions':
                family_by_game = dict(db.execute("SELECT id,json_extract(body,'$.family') FROM sp_games"))
                query = ("SELECT gameid,choice,hash,body,json_extract(body,'$.state'),json_extract(body,'$.after'),json_extract(body,'$.action') "
                         'FROM sp_decisions ORDER BY gameid,choice LIMIT ? OFFSET ?')
                for game,choice,digest,text,before,after,action in db.execute(query,(limit,offset)):
                    _hash_text(text,digest); family = family_by_game[game]; before=json.loads(before); after=json.loads(after)
                    witness = _source(task,table,[game,choice],digest)
                    transition = arena.transition(before,action,ref)
                    need(transition['after']==after, 'Stored self-play decision literal replay differs')
                    domain.mass_transition(before,after); transitions += 1
                    edges.extend(_edge(family,s,dict(witness,metadata_pointer='/'+field)) for field,s in (('state',before),('after',after)))
                    processed += 1
        elif kind == 'historical_arena':
            common.current(task['origin_source_inputs'])
            family_map = _origin_map(*(task['origin_source_inputs'][rel(dbpath(name))] for name in ('deeper-9000-v2','pilot-10000')))
            need(db.execute('SELECT count(*) FROM games').fetchone()[0] == task['expected_total'],
                 'Historical game cardinality changed')
            fields = ['start','moves','family','start_key','final']
            # Saved row hashes uniquely order games across differing legacy keys.
            query = "SELECT hash,body,"+','.join("json_extract(body,'$."+f+"')" for f in fields)+' FROM games ORDER BY hash LIMIT ? OFFSET ?'
            for digest,text,start,moves,family,start_key,final in db.execute(query,(limit,offset)):
                _hash_text(text,digest)
                witness = _source(task,'games',digest,digest)
                if family is None:
                    possibilities=family_map.get(start_key,[])
                    need(possibilities,'Historical origin cannot be resolved')
                    witness['owner_resolution']=dict(start_key=start_key,possible_families=possibilities,
                        source_inputs=task['origin_source_inputs'])
                    if len(possibilities)==1:family=possibilities[0]
                    else:
                        family='ambiguous-saved-arena/'+task['source']+'/'+digest
                        opaque_constraints.append(dict(kind='ambiguous_historical_owner',origin=family,
                            possible_families=possibilities,witness=witness,actual_incidence=False))
                transitions += _trajectory(edges,family,json.loads(start),json.loads(moves),witness,ref,json.loads(final))
                markers.append(_marker('family',family,witness,'prior saved arena trajectory',optimizer=True,fit=True,arena=True,acquisition=True))
                processed += 1
        elif kind == 'recorded_search_trace':
            need(db.execute('SELECT count(*) FROM steps').fetchone()[0]==task['expected_total'],
                 'Recorded search trace cardinality changed')
            family_by_key=dict(db.execute("SELECT state,json_extract(body,'$.family') FROM roots"))
            units={cell:json.loads(unit) for cell,unit in db.execute("SELECT cell,json_extract(body,'$.unit') FROM root_setups")}
            query=("SELECT cell,seq,hash,body,json_extract(body,'$.leaf.state'),json_extract(body,'$.new_edge.node.state'),"
                   "json_extract(body,'$.new_edge.transition.position') FROM steps ORDER BY cell,seq LIMIT ? OFFSET ?")
            for cell,seq,digest,text,leaf,node,child in db.execute(query,(limit,offset)):
                _hash_text(text,digest);family=family_by_key[units[cell]['state_key']]
                witness=_source(task,'steps',[cell,seq],digest)
                for pointer,value in (('/leaf/state',leaf),('/new_edge/node/state',node),('/new_edge/transition/position',child)):
                    if value is not None:edges.append(_edge(family,json.loads(value),dict(witness,metadata_pointer=pointer)))
                markers.append(_marker('family',family,witness,'previous recorded internal search trace',optimizer=True,fit=True,arena=True,acquisition=True))
                processed+=1
        else: raise ValueError('Unknown provenance unit kind')
    need(processed == task['limit'], 'Provenance unit missing projected source rows')
    return dict(schema=SCHEMA+'-unit',index=task['index'],task_sha256=config.digest(task),kind=task['kind'],
                source_rows=processed,literal_replayed_actions=transitions,edges=edges,markers=markers,
                origin_vertices=sorted(set(families)),opaque_constraints=opaque_constraints,**ZERO)


def current_unit(task):
    common.current(task['identity_pin']); identity=read(CURRENT/'identity.json'); head=read(CURRENT/'head.json')
    need(len(head['games'])==task['expected_total'],'Current arena count changed')
    by_game=defaultdict(list); edges=[]; markers=[]; transitions=0
    for entry in head['decisions']:
        need(sha(CURRENT/entry['name'])==entry['sha256'],'Current decision source changed')
        row=read(CURRENT/entry['name'])
        if task['offset']<=row['game_index']<task['offset']+task['limit']:by_game[row['game_index']].append((entry,row))
    ref=arena._literal()
    for index in range(task['offset'],task['offset']+task['limit']):
        schedule=identity['schedule'][index]; family=schedule['family']; p=arena.normalize(schedule['start'],ref)['state']
        witness=dict(path=rel(CURRENT/'identity.json'),sha256=sha(CURRENT/'identity.json'),pointer='/schedule/'+str(index)+'/start')
        edges.append(_edge(family,p,witness)); markers.append(_marker('family',family,witness,'consumed current192-game arena origin',optimizer=True,fit=True,arena=True,acquisition=True))
        rows=[]
        for choice,(entry,row) in enumerate(by_game[index]):
            need(row['before']==p and row['choice_index']==choice,'Current trajectory cursor differs')
            transition=arena.transition(p,row['action'],ref)
            need(transition==row['transition'],'Current literal transition differs')
            domain.mass_transition(p,transition['after']); p=transition['after']; transitions+=1; rows.append(row)
            edges.append(_edge(family,p,dict(path=rel(CURRENT/entry['name']),sha256=entry['sha256'],pointer='/transition/after')))
        entry=head['games'][index];need(sha(CURRENT/entry['name'])==entry['sha256'],'Current game source changed')
        arena._validate_game(read(CURRENT/entry['name']),schedule,rows,p,ref)
    return dict(schema=SCHEMA+'-unit',index=task['index'],task_sha256=config.digest(task),kind=task['kind'],
                source_rows=task['limit'],literal_replayed_actions=transitions,edges=edges,markers=markers,origin_vertices=[],**ZERO)


def json_unit(task):
    common.current(task['metadata_pins']); inv=read(INVENTORY); selection=read(SELECTION)
    edges=[];markers=[];origin_vertices=[]
    def witness(path,pointer):return dict(path=rel(path),sha256=task['metadata_pins'][rel(path)],pointer=pointer)
    for pool in ('primary_pool','arena_pool','protected_cases'):
        for index,row in enumerate(inv[pool]):
            w=witness(INVENTORY,'/'+pool+'/'+str(index))
            if row.get('family'):
                edges.append(_edge(row['family'],row['state'],w))
                if pool=='protected_cases':markers.append(_marker('family',row['family'],w,'old case/arena plan origin',optimizer=True,fit=True,arena=True,acquisition=True))
            else:markers.append(_marker('quotient',metadata.quotient(row['state']),w,'old case/arena quotient tombstone',optimizer=True,fit=True,arena=True,acquisition=True))
    # Legacy reasons are immutable conservative exclusions, not invented edges.
    for cid,component in inv['components'].items():
        for family in component['families']:origin_vertices.append(family)
        opt=any(r.startswith('optimizer/') for r in component['reasons'])
        ar=any(r.startswith('arena/') for r in component['reasons'])
        if opt or ar:
            w=witness(INVENTORY,'/components/'+cid)
            for kind,values in (('family',component['families']),('quotient',component['quotients'])):
                for value in values:
                    markers.append(_marker(kind,value,w,'inherited '+','.join(component['reasons']),optimizer=opt,fit=ar,arena=ar,acquisition=ar))
    # Fixed old support can be reused only by explicit parent treatment; it is
    # always unavailable for any future fit or arena reservation.
    for role in ('train','fit_check','arena'):
        for index,row in enumerate(selection[role]):
            w=witness(SELECTION,'/'+role+'/'+str(index)); component=inv['components'][row['component']]
            for family in component['families']:
                markers.append(_marker('family',family,w,'current frozen '+role+' full origin',optimizer=role!='train',fit=True,arena=True,acquisition=role!='train'))
            for q in component['quotients']:
                markers.append(_marker('quotient',q,w,'current frozen '+role+' full aliases',optimizer=role!='train',fit=True,arena=True,acquisition=role!='train'))
    path=ROOT/'data/ordinary-development-v1/candidates.json'
    for index,row in enumerate(read(path)['accepted']):
        for suffix,state in [('state',row['state'])]+[('probes/'+str(i)+'/state',p['state']) for i,p in enumerate(row['probes'])]:
            w=witness(path,'/accepted/'+str(index)+'/'+suffix); edges.append(_edge(row['family'],state,w))
        markers.append(_marker('family',row['family'],witness(path,'/accepted/'+str(index)),
                               'all32 accepted synthetic development-only origins',optimizer=True,acquisition=True))
    path=ROOT/'build/quality-data.json'; quality=read(path)
    for family in quality['train_families']:
        markers.append(_marker('family',family,witness(path,'/train_families'),'ancestral base-network training origin',ancestral_training=True))
    # These selected caches were inspected diagnostics; extract state/origin
    # metadata only through SQLite JSON projection, never feature/target values.
    for filename in ('build/ordinary-value-rows.json','build/value-pilot-rows.json'):
        path=ROOT/filename;text=path.read_text(encoding='utf-8')
        with closing(sqlite3.connect(':memory:')) as db:
            groups=[r[0] for r in db.execute("SELECT key FROM json_each(?,'$.groups')",(text,))]
            for group in groups:
                query="SELECT key,json_extract(value,'$.state'),json_extract(value,'$.family') FROM json_each(?,?)"
                for index,state,family in db.execute(query,(text,'$.groups.'+group)):
                    if state is None or family is None:continue
                    w=witness(path,'/groups/'+group+'/'+str(index));edges.append(_edge(family,json.loads(state),w))
                    # Rehearsal is an ancestral corpus; it is not all newly
                    # inspected/optimized support for the current sidecar.
                    if group!='rehearsal':markers.append(_marker('family',family,w,'previous value diagnostic cache',optimizer=True,fit=True,arena=True,acquisition=True))
    path=ROOT/'build/neurodiag-panel.json'
    for group in ('primary','paired','MC'):
        for index,row in enumerate(read(path)[group]):
            w=witness(path,'/'+group+'/'+str(index));edges.append(_edge(row['family'],row['state'],w))
            markers.append(_marker('family',row['family'],w,'consumed neural target/value diagnostic',optimizer=True,fit=True,arena=True,acquisition=True))
    return dict(schema=SCHEMA+'-unit',index=task['index'],task_sha256=config.digest(task),kind=task['kind'],
                source_rows=len(edges),literal_replayed_actions=0,edges=edges,markers=markers,
                origin_vertices=sorted(set(origin_vertices)),**ZERO)


class Graph:
    def __init__(self):self.parent={}
    def find(self,node):
        self.parent.setdefault(node,node);root=node
        while self.parent[root]!=root:root=self.parent[root]
        while self.parent[node]!=node:node,self.parent[node]=self.parent[node],root
        return root
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parent[b]=a


def summarize(results):
    planned=plan();tasks=planned['tasks'];need(len(results)==len(tasks),'Provenance unit count incomplete')
    indexed={u['index']:u for u in results};need(set(indexed)==set(range(len(tasks))),'Duplicate/missing provenance unit')
    full=Graph();high=Graph();markers=[];source_counts=Counter();edge_count=shared_count=0
    raw_edges=set();qold={};fold={};inv=read(INVENTORY); selection=read(SELECTION)
    for cid,component in inv['components'].items():
        for f in component['families']:fold[f]=cid
        for q in component['quotients']:qold[q]=cid
    transitions=0;source_rows=0;source_opaque=[]
    for task in tasks:
        u=indexed[task['index']]
        need(u['schema']==SCHEMA+'-unit' and u['kind']==task['kind'] and u['task_sha256']==config.digest(task),
             'Provenance unit source/task differs')
        need(all(u[k]==0 for k in ZERO),'Provenance forbidden-work counter changed')
        source_counts[task['source']]+=u['source_rows'];source_rows+=u['source_rows'];transitions+=u['literal_replayed_actions']
        markers.extend(u['markers'])
        source_opaque.extend(u.get('opaque_constraints',[]))
        for f in u['origin_vertices']:full.find(('family',f));high.find(('family',f))
        for edge in u['edges']:
            f,q=edge['family'],edge['quotient']; a=domain.quotient_active(q)
            need(a==edge['active_seeds'] and edge['domain']==('owned' if a>=25 else 'shared'),
                 'Raw edge domain differs')
            # Lossless raw key is retained in each committed unit witness.
            key=bytes.fromhex(edge['raw_state_key']);need(len(key)==17,'Raw provenance key width differs')
            state=dict(pits=[list(key[:7]),list(key[8:15])],stores=[key[7],key[15]],controller=key[16])
            need(metadata.quotient(state)==q,'Raw provenance quotient differs')
            need(edge['witness']['path'] in planned['inputs'] and edge['witness']['sha256']==planned['inputs'][edge['witness']['path']],
                 'Actual incidence witness source unbound')
            full.join(('family',f),('quotient',q));high.find(('family',f))
            if a>=25:high.join(('family',f),('quotient',q))
            else:shared_count+=1
            raw_edges.add((f,edge['raw_state_key']));edge_count+=1
    # Add pure tombstone vertices without converting them into incidences.
    for marker in markers:
        node=(marker['kind'],marker['value']);full.find(node)
        if node[0]=='family' or domain.quotient_active(node[1])>=25:high.find(node)
    opaque=[];old_partition_reconstructed=0
    for cid,component in inv['components'].items():
        nodes=[('family',f) for f in component['families']]+[('quotient',q) for q in component['quotients']]
        roots={full.find(n) for n in nodes}
        if len(roots)<=1:old_partition_reconstructed+=1;continue
        retained=[n for n in nodes if n[0]=='family' or domain.quotient_active(n[1])>=25]
        constraint=dict(legacy_component=cid,original_nodes=len(nodes),recovered_full_graph_fragments=len(roots),
                        retained_high_nodes=len(retained),source=dict(path=rel(INVENTORY),sha256=sha(INVENTORY),pointer='/components/'+cid),
                        actual_incidence=False,kind='conservative_opaque_union')
        opaque.append(constraint)
        if retained:
            anchor=('opaque_legacy_union',cid)
            for n in retained:high.join(anchor,n)
    for constraint in source_opaque:
        ident=config.digest(constraint);anchor=('opaque_legacy_union',ident)
        high.join(anchor,('family',constraint['origin']))
        for family in constraint['possible_families']:high.join(anchor,('family',family))
        opaque.append(dict(constraint,id=ident,retained_high_connection=True))
    grouped=defaultdict(lambda:dict(families=[],quotients=[],opaque_legacy_unions=[],markers=[]))
    for node in high.parent:
        body=grouped[high.find(node)]
        body[{'family':'families','quotient':'quotients','opaque_legacy_union':'opaque_legacy_unions'}[node[0]]].append(node[1])
    for marker in markers:
        node=(marker['kind'],marker['value'])
        if node in high.parent:grouped[high.find(node)]['markers'].append(marker)
    components={};family_lookup={};quotient_lookup={};marker_ledger=[]
    for body in grouped.values():
        entries=body.pop('markers'); flags={k:any(m[k] for m in entries) for k in ('optimizer_blocked','fit_blocked','arena_blocked','ancestral_training','acquisition_exposed')}
        base={k:sorted(set(v)) for k,v in body.items()};base['schema']=SCHEMA+'-component'
        cid=config.digest(base);base.update(flags,marker_indices=[])
        for marker in entries:
            base['marker_indices'].append(len(marker_ledger));marker_ledger.append(marker)
        components[cid]=base
        for f in base['families']:family_lookup[f]=cid
        for q in base['quotients']:quotient_lookup[q]=cid
    pools={};availability={}
    support_keys={r['state_key'] for r in selection['train']}
    for role,pool in (('support_reuse',selection['train']),('fit_candidate',inv['primary_pool']),('arena_candidate',inv['arena_pool'])):
        out=[]
        for row in pool:
            a=sum(map(sum,row['state']['pits']))
            if not 25<=a<=74 or row['legal_mask'].bit_count()<2:continue
            cid=family_lookup[row['family']];need(cid==quotient_lookup[row['quotient']],'Candidate family/quotient closure differs')
            component=components[cid]; reasons=[]
            if role=='support_reuse':
                if component['optimizer_blocked']:reasons.append('whole component has optimizer exclusion')
            elif component['fit_blocked' if role=='fit_candidate' else 'arena_blocked']:
                reasons.append('whole component has prior role/analysis exposure')
            if component['opaque_legacy_unions']:reasons.append('opaque legacy union remains conservatively unsplittable')
            if role=='fit_candidate' and row['state_key'] in support_keys:reasons.append('current frozen512 support root')
            slim={k:v for k,v in row.items() if k not in ('component',)}
            out.append(dict(slim,new_domain_component=cid,old_component=row['component'],eligible=not reasons,
                            exclusion_reasons=reasons,ancestral_training=component['ancestral_training']))
        pools[role]=sorted(out,key=lambda r:r['state_key'])
        by_cell={}
        for cell in metadata.CELLS:
            candidates=[r for r in out if r['cell']==cell and r['eligible']]
            by_cell[cell]=dict(roots=len(candidates),components=len({r['new_domain_component'] for r in candidates}))
        availability[role]=dict(total_roots=len(out),eligible_roots=sum(r['eligible'] for r in out),
                                eligible_components=len({r['new_domain_component'] for r in out if r['eligible']}),cells=by_cell)
    support_components={r['new_domain_component'] for r in pools['support_reuse']}
    support_overlap=len(pools['support_reuse'])-len(support_components)
    # Certificate covers declared known sources, not unrecorded historical
    # internal leaves or an absence of ancestral optimizer exposure.
    return dict(schema=SCHEMA+'-report',inputs=planned['inputs'],units=len(tasks),source_catalog=planned['source_catalog'],
        source_projected_row_counts=dict(sorted(source_counts.items())),source_rows=source_rows,
        actual_incidence_occurrences=edge_count,distinct_family_raw_incidences=len(raw_edges),shared_incidence_occurrences=shared_count,
        owned_incidence_occurrences=edge_count-shared_count,literal_replayed_actions=transitions,
        actual_incidence_catalog='All raw edges/witnesses reside in corresponding committed unit artifacts; never family×quotient inferred edges.',
        components=dict(sorted(components.items())),family_components=dict(sorted(family_lookup.items())),quotient_components=dict(sorted(quotient_lookup.items())),
        marker_ledger=marker_ledger,opaque_legacy_unions=opaque,
        legacy_components_reconstructed_from_actual_incidence=old_partition_reconstructed,
        legacy_component_total=len(inv['components']),candidate_pools=pools,availability=availability,
        exact_current_support_roots=512,current_support_roots_present=len(pools['support_reuse']),
        current_support_distinct_components=len(support_components),current_support_component_collisions=support_overlap,
        shared_ledger='Every <=24 incidence remains in immutable unit artifacts with exact origin, raw stores/controller and source witness.',
        known_material_source_coverage_complete=True,known_source_graph_certifiable=True,
        coverage_scope='All exact source files and metadata tables declared in this plan, including persisted lineage/search-trace witnesses; not every historical internal leaf.',
        opaque_connections_preserved=True,cartesian_flattened_edges_created=0,
        historical_internal_search_feature_tt_leaves='not comprehensively recorded; unknown ancestral exposure, not reconstructed or claimed absent',
        historically_unseen_claim=False,production_admission=False,models_and_policy_fixed=True,
        new_support_broadened=False,training_authorized=False,**ZERO)

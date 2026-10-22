"""Disk-backed exact ownership closure with a small owner union-find.

Every acquired root's complete quotient incidence is durable in SQLite. Only
the roughly20,000 component owners occupy the union-find; no Python graph node
is allocated per searched quotient. Final readers use read-only immutable mode.
"""
from collections import defaultdict
from contextlib import closing
import hashlib,itertools,json,os,sqlite3,struct
from . import config,footprint_screen_common as s

SCHEMA=s.SCHEMA+'-disk-closure'
SOURCES=('congklak/footprint_screen_graph.py',)
MAP_NAME='quotient-components.sqlite3'
_SQL=(
 'CREATE TABLE meta(k TEXT PRIMARY KEY,v TEXT NOT NULL) WITHOUT ROWID',
 'CREATE TABLE owners(id INTEGER PRIMARY KEY,anchor TEXT NOT NULL UNIQUE,parent INTEGER NOT NULL)',
 'CREATE TABLE prior(q BLOB PRIMARY KEY,owner INTEGER NOT NULL) WITHOUT ROWID',
 'CREATE TABLE keys(q BLOB PRIMARY KEY,owner INTEGER NOT NULL,first_root INTEGER NOT NULL,final_owner INTEGER) WITHOUT ROWID',
 'CREATE TABLE incidence(root INTEGER NOT NULL,q BLOB NOT NULL,PRIMARY KEY(root,q)) WITHOUT ROWID',
 'CREATE TABLE roots(idx INTEGER PRIMARY KEY,family TEXT NOT NULL,origin TEXT NOT NULL,owner INTEGER NOT NULL,qcount INTEGER NOT NULL,qsha TEXT NOT NULL,refs_sha TEXT NOT NULL)',
 'CREATE TABLE components(owner INTEGER PRIMARY KEY,cid TEXT NOT NULL UNIQUE,qcount INTEGER NOT NULL,qsha TEXT NOT NULL,summary TEXT NOT NULL)',
)
need=s.need

def same(a,b):return config.canonical(a)==config.canonical(b)
def _key(text):
    need(type(text) is str and len(text)==30 and text==text.lower(),'Malformed ownership quotient')
    q=bytes.fromhex(text)
    need(len(q)==15 and q[14] in (0,1) and 25<=sum(q[:14])<=98 and
         q==min(q,q[7:14]+q[:7]+bytes([1-q[14]])),'Shared low or noncanonical ownership quotient')
    return q
def _hash(rows):
    digest=hashlib.sha256();count=0
    for row in rows:digest.update(row);count+=1
    return count,digest.hexdigest()

class UF:
    def __init__(self,count):self.parent=list(range(count+1))
    def find(self,index):
        root=index
        while self.parent[root]!=root:root=self.parent[root]
        while self.parent[index]!=index:index,self.parent[index]=self.parent[index],root
        return root
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)

class Paused(RuntimeError):
    def __init__(self,cursor):self.cursor=cursor;super().__init__('Durable ownership cursor '+str(cursor))

def _layout(prior,records):
    anchors=['prior/'+cid for cid in sorted(prior['components'])]
    anchors.extend('family/'+f for f in sorted({root['family'] for root,_ in records}-set(prior['family_components'])))
    owner={anchor:i+1 for i,anchor in enumerate(anchors)}
    old={cid:owner['prior/'+cid] for cid in prior['components']}
    families={f:old[cid] for f,cid in prior['family_components'].items()}
    families.update({f:owner['family/'+f] for f in {root['family'] for root,_ in records} if f not in families})
    return anchors,old,families

def _connect(path,readonly=False):
    if readonly:return sqlite3.connect(path.resolve().as_uri()+'?mode=ro&immutable=1',uri=True)
    db=sqlite3.connect(path,timeout=30)
    db.execute('PRAGMA journal_mode=DELETE');db.execute('PRAGMA synchronous=FULL')
    db.execute('PRAGMA temp_store=FILE');db.execute('PRAGMA cache_size=-32768')
    return db

def _meta(db,key):return json.loads(db.execute('SELECT v FROM meta WHERE k=?',(key,)).fetchone()[0])
def _set(db,key,value):db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)',(key,config.canonical(value)))

def _schema(db,finished=False):
    actual=dict(db.execute("SELECT name,sql FROM sqlite_master WHERE type='table'"))
    need(set(actual)=={'meta','owners','prior','keys','incidence','roots','components'},'Ownership SQLite schema tables differ')
    expected={sql.split()[2].split('(')[0]:sql for sql in _SQL}
    need(actual==expected,'Ownership SQLite schema definitions differ')
    indexes=dict(db.execute("SELECT name,sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"))
    allowed={'incidence_by_q':'CREATE INDEX incidence_by_q ON incidence(q,root)'}
    if finished:allowed['keys_by_final']='CREATE INDEX keys_by_final ON keys(final_owner,q)'
    need(indexes==allowed,'Ownership SQLite index scope differs')
    need(not list(db.execute("SELECT name FROM sqlite_master WHERE type IN ('trigger','view')")),'Untrusted ownership SQLite code')

def _init(db,prior,records,identity):
    anchors,old,_=_layout(prior,records)
    with db:
        db.execute('BEGIN IMMEDIATE')
        for sql in _SQL:db.execute(sql)
        db.execute('CREATE INDEX incidence_by_q ON incidence(q,root)')
        db.executemany('INSERT INTO owners VALUES(?,?,?)',((i+1,a,i+1) for i,a in enumerate(anchors)))
        db.executemany('INSERT INTO prior VALUES(?,?)',((_key(q),old[c]) for q,c in prior['quotient_components'].items()))
        db.execute('INSERT INTO keys(q,owner,first_root) SELECT q,owner,-1 FROM prior')
        _set(db,'identity',identity);_set(db,'cursor',0);_set(db,'finished',False)

def _validate(db,prior,records,identity,finished=False):
    """Source-derived incidence and partition verification, bounded working RAM."""
    _schema(db,finished);need(same(_meta(db,'identity'),identity),'Disk ownership source identity differs')
    cursor=_meta(db,'cursor');need(type(cursor) is int and 0<=cursor<=len(records),'Invalid durable ownership cursor')
    need(_meta(db,'finished') is finished,'Disk ownership completion flag differs')
    anchors,old,families=_layout(prior,records);owners=list(db.execute('SELECT id,anchor,parent FROM owners ORDER BY id'))
    need([(i,a) for i,a,_ in owners]==list(enumerate(anchors,1)) and all(type(p) is int and 1<=p<=len(anchors) for _,_,p in owners),
         'Disk component-owner inventory differs')
    expected=_hash(_key(q)+struct.pack('<I',old[c]) for q,c in sorted(prior['quotient_components'].items()))
    actual=_hash(q+struct.pack('<I',owner) for q,owner in db.execute('SELECT q,owner FROM prior ORDER BY q'))
    need(actual==expected,'Disk prior high quotient constraints differ')
    rows=list(db.execute('SELECT idx,family,origin,owner,qcount,qsha,refs_sha FROM roots ORDER BY idx'))
    need(len(rows)==cursor,'Disk incidence root cursor differs')
    uf=UF(len(anchors));incidence_count=0
    for index,(root,result) in enumerate(records[:cursor]):
        expected_q=_hash(_key(q) for q in result['quotients'])
        row=(index,root['family'],root['origin'],families[root['family']],*expected_q,config.digest(result['source_refs']))
        need(rows[index]==row,'Disk root/source footprint binding differs')
        actual_q=_hash(q for (q,) in db.execute('SELECT q FROM incidence WHERE root=? ORDER BY q',(index,)))
        need(actual_q==expected_q,'Disk complete actual root incidence differs')
        incidence_count+=expected_q[0]
        for (other,) in db.execute('SELECT DISTINCT k.owner FROM incidence i JOIN keys k ON k.q=i.q WHERE i.root=?',(index,)):
            need(type(other) is int and 1<=other<=len(anchors),'Disk key has an unknown owner');uf.join(families[root['family']],other)
    need(db.execute('SELECT count(*) FROM incidence').fetchone()[0]==incidence_count,'Disk contains undeclared root incidence')
    bad=db.execute('SELECT count(*) FROM keys k LEFT JOIN prior p ON p.q=k.q WHERE k.first_root=-1 AND (p.q IS NULL OR p.owner!=k.owner)').fetchone()[0]
    bad+=db.execute('SELECT count(*) FROM prior p LEFT JOIN keys k ON p.q=k.q WHERE k.q IS NULL OR k.first_root!=-1 OR k.owner!=p.owner').fetchone()[0]
    bad+=db.execute('SELECT count(*) FROM keys k LEFT JOIN roots r ON r.idx=k.first_root WHERE k.first_root!=-1 AND '
        '(r.idx IS NULL OR r.owner!=k.owner OR NOT EXISTS(SELECT 1 FROM incidence i WHERE i.root=k.first_root AND i.q=k.q) '
        'OR EXISTS(SELECT 1 FROM incidence i WHERE i.root<k.first_root AND i.q=k.q))').fetchone()[0]
    bad+=db.execute('SELECT count(*) FROM incidence i LEFT JOIN keys k ON i.q=k.q WHERE k.q IS NULL').fetchone()[0]
    need(bad==0,'Disk key ownership contains missing, invented or wrong first-source keys')
    stored=UF(len(anchors));stored.parent=[0]+[p for _,_,p in owners]
    # A forged parent cycle must be refused without entering an endless find.
    for index in range(1,len(anchors)+1):
        current=index;seen=set()
        while stored.parent[current]!=current:
            need(current not in seen,'Disk owner-parent cycle');seen.add(current);current=stored.parent[current]
        need(uf.find(index)==current,'Disk union partition differs from full actual incidences')
    return uf,cursor

def _summaries(db,prior,records,uf):
    anchors,old,families=_layout(prior,records);groups=defaultdict(list);origins=defaultdict(list)
    for i,anchor in enumerate(anchors,1):groups[uf.find(i)].append(anchor)
    for root,_ in records:origins[uf.find(families[root['family']])].append(root['origin'])
    hashes={};counts={}
    for owner,values in itertools.groupby(db.execute('SELECT final_owner,q FROM keys ORDER BY final_owner,q'),lambda x:x[0]):
        count,digest=_hash(q for _,q in values);hashes[owner]=digest;counts[owner]=count
    result={};ids={}
    for owner,names in sorted(groups.items()):
        oldids=sorted(name[6:] for name in names if name.startswith('prior/'))
        blocked=[c for c in oldids if prior['components'][c]['evaluation_blocked']]
        qsha=hashes.get(owner,hashlib.sha256(b'').hexdigest());count=counts.get(owner,0)
        actual=sorted(origins[owner]);cid=config.digest(dict(schema=SCHEMA+'-component',anchors=sorted(names),
            acquired_origins=actual,quotient_count=count,quotient_sha256=qsha))
        ids[owner]=cid;result[cid]=dict(prior_components=oldids,evaluation_blocked=bool(blocked),blocking_prior_components=blocked,
            acquisition_exposed=bool(actual),acquired_origins_optimizer_blocked=bool(actual),
            acquisition_is_optimizer_use=False,support192_permission_not_reassessed=True,
            acquired_origins=actual,quotient_count=count,quotient_sha256=qsha)
    return result,ids,{f:ids[uf.find(owner)] for f,owner in families.items()}

def build(folder,prior,records,identity,should_pause=lambda:False):
    """Commit one full root at a time; no engine or target computation occurs."""
    folder=s.path_of(folder);folder.mkdir(parents=True,exist_ok=True);final=folder/MAP_NAME;work=folder/'work.sqlite3'
    if final.exists():return
    db=_connect(work)
    try:
        if not list(db.execute("SELECT name FROM sqlite_master WHERE type='table'")):_init(db,prior,records,identity)
        finished=_meta(db,'finished');need(type(finished) is bool,'Disk completion flag must be boolean')
        uf,cursor=_validate(db,prior,records,identity,finished)
        _,_,families=_layout(prior,records)
        if not finished:
            db.execute('CREATE TEMP TABLE incoming(q BLOB PRIMARY KEY) WITHOUT ROWID')
            for index in range(cursor,len(records)):
                if should_pause():raise Paused(index)
                root,result=records[index];owner=families[root['family']]
                before=uf.parent.copy()
                try:
                    with db:
                        db.execute('DELETE FROM incoming')
                        db.executemany('INSERT INTO incoming VALUES(?)',((_key(q),) for q in result['quotients']))
                        for (other,) in db.execute('SELECT DISTINCT k.owner FROM incoming i JOIN keys k ON i.q=k.q'):uf.join(owner,other)
                        db.execute('INSERT OR IGNORE INTO keys(q,owner,first_root) SELECT q,?,? FROM incoming',(owner,index))
                        db.execute('INSERT INTO incidence SELECT ?,q FROM incoming',(index,))
                        count,digest=_hash(_key(q) for q in result['quotients'])
                        db.execute('INSERT INTO roots VALUES(?,?,?,?,?,?,?)',(index,root['family'],root['origin'],owner,count,digest,config.digest(result['source_refs'])))
                        db.executemany('UPDATE owners SET parent=? WHERE id=?',((uf.parent[i],i) for i in range(1,len(before)) if uf.parent[i]!=before[i]))
                        _set(db,'cursor',index+1)
                except BaseException:uf.parent=before;raise
            if should_pause():raise Paused(len(records))
            with db:
                db.execute('CREATE TEMP TABLE final_owners(id INTEGER PRIMARY KEY,root INTEGER NOT NULL)')
                db.executemany('INSERT INTO final_owners VALUES(?,?)',((i,uf.find(i)) for i in range(1,len(uf.parent))))
                db.execute('UPDATE keys SET final_owner=(SELECT root FROM final_owners f WHERE f.id=keys.owner)')
                db.execute('CREATE INDEX keys_by_final ON keys(final_owner,q)')
                summaries,ids,_=_summaries(db,prior,records,uf)
                db.executemany('INSERT INTO components VALUES(?,?,?,?,?)',((owner,cid,summaries[cid]['quotient_count'],
                    summaries[cid]['quotient_sha256'],config.canonical(summaries[cid])) for owner,cid in ids.items()))
                _set(db,'finished',True)
    finally:db.close()
    # DELETE-journal transactions are complete before this immutable publication.
    with work.open('r+b') as stream:stream.flush();os.fsync(stream.fileno())
    os.replace(work,final)

def load(folder,prior,records,identity):
    """Independent pure closure reader. Source incidences, not self-hashes, win."""
    path=s.path_of(folder)/MAP_NAME
    with closing(_connect(path,True)) as db:
        uf,cursor=_validate(db,prior,records,identity,True)
        need(cursor==len(records),'Final disk closure omits acquired roots')
        for owner,root in db.execute('SELECT DISTINCT owner,final_owner FROM keys'):
            need(root==uf.find(owner),'Final quotient component lookup differs')
        summaries,ids,families=_summaries(db,prior,records,uf)
        expected=sorted((owner,cid,summaries[cid]['quotient_count'],summaries[cid]['quotient_sha256'],config.canonical(summaries[cid])) for owner,cid in ids.items())
        need(list(db.execute('SELECT owner,cid,qcount,qsha,summary FROM components ORDER BY owner'))==expected,'Final component summaries differ from incidence closure')
        count,digest=_hash(q+bytes.fromhex(ids[owner]) for q,owner in db.execute('SELECT q,final_owner FROM keys ORDER BY q'))
        incidence_count=db.execute('SELECT count(*) FROM incidence').fetchone()[0]
    return dict(schema=SCHEMA+'-validated',components=summaries,family_components=families,
        quotient_component_map=dict(path=s.relative(path),sha256=s.sha(path),quotient_count=count,
            sorted_packed_mapping_sha256=digest,packed_record_bytes=47,incidence_count=incidence_count,
            schema=SCHEMA+'-sqlite',lookup='SELECT c.cid FROM keys k JOIN components c ON c.owner=k.final_owner WHERE k.q=?'),
        identity_sha256=config.digest(identity),cursor=cursor,owner_union_find_only=True)

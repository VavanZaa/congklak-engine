"""Source-bound additive executor projection; never edits the sealed source."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'congklak/ordinary_epoch_execute.py'
NEW = ROOT / 'congklak/guarded_value_pilot_v2_execute.py'
RECEIPT = ROOT / 'build/guarded-value-pilot-v2/audit/execute-projection.json'


def replace_function(source, name, replacement):
    tree = ast.parse(source)
    node = next(item for item in tree.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == name)
    lines = source.splitlines(keepends=True)
    lines[node.lineno - 1:node.end_lineno] = [replacement.rstrip() + '\n']
    return ''.join(lines)


def project():
    original = OLD.read_text(encoding='utf-8')
    source = original.replace('ordinary_epoch_', 'guarded_value_pilot_v2_')
    source = source.replace('Every operation extends a separate immutable journal from the sealed labels\nprefix.',
                            'Every operation extends the admitted complete capture closure and the\npreceding durable live-claim prefix.')
    source = source.replace('ordinary-epoch-execute.py', 'guarded-value-pilot-v2-execute.py')
    source = source.replace("DEFAULT=FOLDER/'jobs/labels'", "DEFAULT=FOLDER/'jobs/conformance'")
    source = source.replace("SELECTION=ROOT/'build/ownership-implementation-v1/reservations/production-v2/selection.json'",
                            "SELECTION=ROOT/'build/footprint-screen-v1/reservations/report.json'")
    source = source.replace("OPERATIONS=('labels','conformance','guards','arena')", "OPERATIONS=('conformance','guards','arena')")
    source = source.replace("operation='labels'", "operation='conformance'")
    source = source.replace("s.DATA/'cache-v2'", "s.DATA/'cache'")
    source = source.replace("default='labels'", "default='conformance'")
    source = source.replace("event=s.normalized(event)", "event=json.loads(config.canonical(event))")
    source = source.replace("from .valuepilot_reservations import Writer", "from .valuepilot_reservations import Writer\nfrom .guarded_value_pilot_v2_job_views import admission_scope,admission_receipt")
    source = source.replace(" 'congklak/guarded_value_pilot_v2_query_contract.py',", " 'congklak/guarded_value_pilot_v2_query_contract.py','congklak/guarded_value_pilot_v2_job_views.py',")
    source = replace_function(source, 'prepare_registry', '''def prepare_registry():
 """Read the independently prepared complete shared lookup; no query work."""
 from . import guarded_value_pilot_v2_guard as guard
 s.admit_plan(False)
 return guard.registry()''')
    source = replace_function(source, '_guard_spec', '''def _guard_spec(origin,registry,batch=None):
 from . import guarded_value_pilot_v2_guard as guard
 return guard.spec(origin,registry,batch)''')
    source = replace_function(source, '_expected_guard_refs', '''def _expected_guard_refs(origin,registry,batch=None):
 from . import guarded_value_pilot_v2_guard as guard
 return guard.expected_refs(origin,registry,batch)''')
    source = replace_function(source, '_candidate', '''def _candidate(operation,fixture,model):
 from . import guarded_value_pilot_v2_native as native
 expected=s.DATA/('initial/model.json' if fixture and operation=='conformance' else 'training/model.json')
 path=Path(model).resolve() if model else expected
 need(path==expected.resolve(),'Candidate path is not the one prospective treatment')
 from .guarded_value_pilot_v2_sidecar import verify_export
 export=verify_export(path);provenance=export['model']['provenance']
 if fixture and operation=='conformance':
  need(provenance['kind']=='stepzero' and type(provenance['optimizer_steps']) is int and provenance['optimizer_steps']==0,
   'Initial conformance requires the exact stepzero export')
 else:
  from . import guarded_value_pilot_v2_train as training
  training.completed(path.parent)
  need(provenance['kind']=='fixed-final-step' and provenance['optimizer_steps']==400 and
   provenance['engineering_fixture'] is False,'Development execution requires final400 only')
 return native.candidate_binding(path)''')
    # Only two added metadata bindings surround unchanged identity creation.
    source = source.replace("s.admit_plan();need(operation in OPERATIONS and type(fixture) is bool,'Unknown executor scope');scope=classify(path,operation,fixture)",
        "s.admit_plan();need(operation in OPERATIONS and type(fixture) is bool,'Unknown executor scope');scope=classify(path,operation,fixture)\n from . import guarded_value_pilot_v2_data as cached\n need(Path(cache).resolve()==cached.DEFAULT.resolve(),'Executor cache path differs')\n cached.load_fit()")
    source = source.replace("scope=scope,policy_fixed=True,base_frozen=True,inherited_unknown_background=True)",
        "scope=scope,cached_fit_manifest=_ref(cached.FIT_DEFAULT/'manifest.json'),policy_fixed=True,base_frozen=True,inherited_unknown_background=True)")
    source = source.replace("identity=read(path/'identity.json');s.admit_plan()", "identity=read(path/'identity.json');s.admit_plan()\n from . import guarded_value_pilot_v2_data as cached\n cached.load_fit()\n need(identity['cache']==str(cached.DEFAULT.resolve()) and identity['cached_fit_manifest']==_ref(cached.FIT_DEFAULT/'manifest.json'),'Captured fit/cache binding changed')")
    source = replace_function(source, '_labels_rows', '''def _labels_rows(identity,data):
 raise ValueError('No new teacher-label operation exists in this pilot')''')
    source = replace_function(source, '_phase_summary', '''def _phase_summary(identity,data):
 held=data['hold'];operation=identity['operation'];expected=len(identity['schedule'])
 completed=len(data['games']) if operation=='arena' else len(data['results'])
 complete=not held and data['intent'] is None and data['pending'] is None and completed==expected
 overshoots=sum(r['body']['clock']['critical_ms']>25 for r in data['results'] if r['body']['clock'].get('requested_ms')==25)
 result=dict(schema=SCHEMA+'-'+operation+'-report',operation=operation,state='held' if held else 'complete' if complete else 'checkpoint',
  fixture=identity['fixture'],required_units=expected,completed_units=completed,queries=len(data['results']),
  generation=data['generation'],global_claims=len(data['claimed']),canonical_claims_sha256=data['claimed'].sha256,
  pending_attempts=int(data['intent'] is not None or data['pending'] is not None),hold=held,
  ownership_passed=not held and all(r['result']['passed'] is True for r in data['results']),
  critical_overshoots=overshoots,failed_returned_attempts=int(bool(held and held.get('failed_attempt_ref'))),
  failed_known_critical_overshoots=int(bool(held and held.get('known_critical_overshoot'))),
  policy_fixed=True,base_frozen=True,production_evaluation_admitted=False,
  inherited_unknown_background=True,historically_unseen=False)
 if operation=='arena':result.update(games=len(data['games']),terminal_games=len(data['games']),caps=0,fallbacks=0)
 return result''')
    source = replace_function(source, '_finalize', '''def _finalize(path,identity,data,pause=lambda point:None):
 report=_phase_summary(identity,data);need(report['state'] in ('complete','held'),'Cannot seal partial operation')
 _write_bytes(_claim_path(path),data['claimed'].records)
 atomic(path/'ownership-prefix.json',_prefix_document(path,identity,data,report))
 atomic(path/'report.json',report)
 catalog=dict(schema=SCHEMA+'-complete',identity_ref=_ref(path/'identity.json'),report_ref=_ref(path/'report.json'),
  ownership_prefix_ref=_ref(path/'ownership-prefix.json'),commits=s.common.pins(sorted((path/'commits').glob('*.json'))),fit_manifest_ref=None)
 pause('before_catalog');atomic(path/'catalog.json',catalog);pause('after_catalog')
 return dict(report=report,catalog=catalog,fit_manifest=None)''')
    source = replace_function(source, '_fit_manifest_document', '''def _fit_manifest_document(path,rows):
 raise ValueError('Captured fit manifest is immutable input, not executor output')''')
    # Remove the old labels-only closing branch; the new finite operations cannot label.
    source = source.replace(" if operation=='labels' and expected['state']=='complete':\n  rows=_labels_rows(identity,data);manifest=read(path/'fit-manifest.json')\n  need(_same(manifest,_fit_manifest_document(path,rows)) and _same(read(path/'fit-rows.json'),rows),\n   'Sealed fit rows/membership differ')\n", '')
    source = replace_function(source, 'completed_labels', '''def completed_labels(path=None):
 from . import guarded_value_pilot_v2_data as cached
 rows,manifest=cached.load_fit()
 return dict(report=dict(state='complete',roots=len(rows),complete_h8_roots=len(rows),fit_cases=32,
  ownership_passed=True,new_teacher_queries=0),fit_manifest=manifest,publication_gap=False)''')
    source = replace_function(source, 'admit_labels', '''def admit_labels(path=None):
 from . import guarded_value_pilot_v2_data as cached
 rows,manifest=cached.load_fit()
 need(len(rows)==128 and sum(row['is_case'] for row in rows)==32,'Captured fit membership differs')
 return dict(schema=s.SCHEMA+'-captured-label-admission',required_roots=128,completed_roots=128,
  fit_cases=32,horizon=8,all_legal_Q_complete=True,features_complete=True,ownership_passed=True,
  support192_disjoint=True,pending_attempts=0,hold=False,new_teacher_queries=0,
  catalog_ref=_ref(cached.FIT_DEFAULT/'catalog.json'),fit_manifest_ref=_ref(cached.FIT_DEFAULT/'manifest.json'))''')
    source = replace_function(source, 'load_fit', '''def load_fit(path=None):
 from . import guarded_value_pilot_v2_data as cached
 return cached.load_fit()[0]''')
    source = source.replace("capacity=16000000 if operation=='labels' else 2000000", "capacity=2000000")
    source = source.replace("      if kind=='teacher':transient=providers.teacher(state,cursor,origin,path/intent['attempt_sink'])\n      elif kind in ('features','prediction'):",
                            "      need(kind in ('features','prediction','decision'),'No new teacher-query kind is admitted')\n      if kind in ('features','prediction'):")
    source = source.replace("      if kind=='teacher':native.teacher_certificate(archived,state)\n", '')
    source = source.replace(" for folder in ('labels','initial-conformance','conformance','guards','arena'):",
                            " from . import guarded_value_pilot_v2_data as cached\n cached.load_fit()\n inputs.update(s.common.pins(p for p in cached.FIT_DEFAULT.iterdir() if p.is_file() and p.suffix=='.json'))\n for folder in ('initial-conformance','conformance','guards','arena'):")
    # Deleted labels cannot invoke the former subset-only no-new-claim shortcut.
    source = source.replace(" if identity['fixture'] and identity['operation']=='labels':\n  need(expected['stage_new_count']==0,'Recovery fixture discovered new owned states; campaign HOLD')\n", '')
    source = source.replace("def completed_job(output=DEFAULT,operation='conformance'):",
                            "def _completed_job_uncached(output=DEFAULT,operation='conformance'):",1)
    source = source.replace("\ndef has_completion(", '''
def completed_job(output=DEFAULT,operation='conformance'):
 from . import guarded_value_pilot_v2_job_views as views
 return views.completed(s.path_of(output),operation,lambda:_completed_job_uncached(output,operation))


def has_completion(''',1)
    source = replace_function(source, 'main', '''def main(argv=None):
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=('run','completed','status','publish'))
 parser.add_argument('--output',default=str(DEFAULT));parser.add_argument('--cache',default=str(s.DATA/'cache'))
 parser.add_argument('--lease');parser.add_argument('--operation',choices=OPERATIONS,default='conformance')
 parser.add_argument('--engineering-fixture',action='store_true');parser.add_argument('--model')
 parser.add_argument('--pause-at');parser.add_argument('--pause-marker');args=parser.parse_args(argv)
 if args.command=='run':
  if args.lease is None:
   need(has_completion(args.output,args.operation),'Contained native worker lease absent')
   answer=publish_completed(args.output,args.operation)
  else:
   from . import guarded_value_pilot_v2_resources as resources
   resources.bind_worker(args.lease)
   with s.worker_views():
    answer=run(args.output,args.cache,args.lease,args.operation,args.engineering_fixture,args.model,args.pause_at,args.pause_marker)
   resources.worker_complete(args.lease,dict(state=answer['report']['state'],operation=args.operation,
    completed_job_view=admission_receipt()))
 elif args.command=='completed':answer=completed_job(args.output,args.operation)
 elif args.command=='publish':answer=publish_completed(args.output,args.operation)
 else:
  path=s.path_of(args.output)
  if not(path/'identity.json').exists():answer=dict(state='not_started')
  else:
   identity=_validate_identity(path,args.operation);answer=_phase_summary(identity,_replay(path,identity))
 print(config.canonical(answer));return answer''')
    # Source projection receipt distinguishes unchanged journal/search logic from explicit role changes.
    old_nodes = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(original).body if isinstance(node, ast.FunctionDef)}
    new_nodes = {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    unchanged = sorted(name for name in old_nodes if old_nodes[name] == new_nodes.get(name))
    changed = sorted(name for name in old_nodes if name in new_nodes and name not in unchanged)
    expected_unchanged = {'_same','_ref','_salt','_events','_thin_result','_complete_data','_claim_path','_write_bytes','_prefix_document','_check_orphans','_ref_absolute','_fixed_view'}
    assert expected_unchanged <= set(unchanged), sorted(expected_unchanged - set(unchanged))
    NEW.write_text(source, encoding='utf-8', newline='\n')
    receipt = dict(schema='guarded-value-pilot-v2-executor-source-projection',
        original=dict(path=OLD.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(OLD.read_bytes()).hexdigest()),
        projected=dict(path=NEW.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(NEW.read_bytes()).hexdigest()),
        builder=dict(path=Path(__file__).relative_to(ROOT).as_posix(),sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        unchanged_function_AST=unchanged,changed_functions=changed,
        allowed_changes=['explicit new modules and namespaces','shared source-derived guard registry',
          'cached accepted fit input and no label worker','stepzero and final400 binding',
          'blank initial live claims then predecessor prefix chain',
          'actual worker binds before cold validation and finite full-byte-closed completed-job views'],
        search_algorithm_changed=False,optimizer_algorithm_changed=False,new_teacher_queries=0,
        models_fixed=True,policy_fixed=True)
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8',newline='\n')
    return receipt


if __name__ == '__main__':
    print(json.dumps(project(),sort_keys=True))

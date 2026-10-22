"""Author checks for full-root requested targets, protection and durable recovery."""
from contextlib import closing
import copy
from pathlib import Path
import sqlite3
import sys
import time
import unittest
import uuid
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from congklak import config,midgame_label_pilot as p
from congklak.native import Native


class MidgameLabelPilotAuthor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p.plan();cls.registry=p.read(p.REGISTRY);cls.settings=p.settings_for(True,2,1);cls.native=Native()
        cls.collection=ROOT/'build/midgame-data-pilot-author/cd6f8cdcf4f244828df759983d4bf2a0/fresh'
        p.collector().load_rows(cls.collection)
        cls.base=ROOT/'build/midgame-label-pilot-author'/uuid.uuid4().hex;cls.base.mkdir(parents=True)
        cls.path=cls.base/'fresh';p.run(cls.path,fixture=True,collection_path=cls.collection)
        cls.rows,cls.stores,cls.manifest=p.load_rows(cls.path)

    def test_01_literal_diagnostics_ownership_and_fixed_budget(self):
        prior=p.checked_baseline();self.assertEqual(prior['inputs_count'],2639)
        diagnostics=p.read(p.collector().REGISTRY)['diagnostics'];self.assertEqual(self.registry['diagnostics'],diagnostics)
        self.assertEqual(len(diagnostics),9);self.assertEqual(len(self.registry['old_h8_references']),1)
        for key,row in diagnostics.items():
            self.assertEqual(p.collector().raw_key(row['state']),key);self.assertEqual(row['state'],row['provenance']['raw_state'])
            self.assertFalse(row['provenance']['swapped']);self.assertEqual(row['cohort'],'diagnostic')
        self.assertEqual(sum(row['study_role']=='fresh' for row in self.rows.values()),1)
        self.assertEqual(sum(row['study_role']=='diagnostic' for row in self.rows.values()),2)
        self.assertEqual(p.CONTRACT['maximum_new_production_nodes'],503000000)
        self.assertEqual([p.CONTRACT[s]['node_budget'] for s in ('h8','h10','h12')],[2000000,5000000,10000000])
        with self.assertRaises(ValueError):p.settings_for(False,2,1)
        self.assertFalse(p.CONTRACT['H14_authorized'])

    def test_02_complete_full_Q_and_later_stage_scope(self):
        p.validate_tables(self.rows,self.stores,self.settings,self.stores['old_h8_references'],self.native,closed=True)
        self.assertEqual(len(self.stores['h8_attempts']),len(self.rows));self.assertEqual(len(self.stores['h10_attempts']),len(self.rows))
        self.assertEqual(len(self.stores['h12_attempts']),2)
        for stage in ('h8','h10','h12'):
            for key,attempt in self.stores[stage+'_attempts'].items():
                self.assertEqual([report['choice_depth'] for report in attempt['history']],list(range(1,attempt['completed_depth']+1)))
                if attempt['status']=='complete':
                    item=self.stores[stage+'_labels'][key]
                    self.assertEqual(item,p.item_for(attempt['history'][-1],self.rows[key]));self.assertEqual(item['label']['choice_depth'],self.settings[stage]['depth'])
                    self.assertEqual({r['pit'] for r in item['label']['action_scores']},{pit for pit in range(7) if self.rows[key]['legal_mask']&(1<<pit)})
                else:
                    self.assertNotIn(key,self.stores[stage+'_labels']);self.assertEqual(attempt['work']['nodes'],self.settings[stage]['node_budget'])
        for key,reference in self.stores['old_h8_references'].items():
            expected={name:reference['label'][name] for name in ('target_player0','best_moves','action_scores')}
            for stage in ('h8','h10','h12'):
                for report in self.stores[stage+'_attempts'][key]['history']:
                    if report['choice_depth']==8:self.assertEqual(p.q_signature(p.item_for(report,self.rows[key])),expected)

    def test_03_rehashed_corruption_and_nonpublication(self):
        key=next(iter(self.stores['h8_attempts']));attempt=self.stores['h8_attempts'][key]
        broken=copy.deepcopy(attempt);broken['history'].pop(0)
        with self.assertRaises(ValueError):p.validate_attempt(self.rows[key],broken,self.stores['h8_labels'].get(key),'h8',self.settings,self.native)
        zeroed=copy.deepcopy(attempt);zeroed['work']={name:0 for name in zeroed['work']}
        with self.assertRaises(ValueError):p.validate_attempt(self.rows[key],zeroed,self.stores['h8_labels'].get(key),'h8',self.settings,self.native)
        fake_label=p.item_for(attempt['history'][0],self.rows[key])
        with self.assertRaises(ValueError):p.validate_attempt(self.rows[key],attempt,fake_label,'h8',self.settings,self.native)
        wrong=copy.deepcopy(self.stores);fresh=next(k for k,r in self.rows.items() if r['study_role']=='fresh');wrong['h12_attempts'][fresh]=copy.deepcopy(attempt)
        with self.assertRaises(ValueError):p.validate_tables(self.rows,wrong,self.settings,wrong['old_h8_references'],self.native,closed=True)
        target=self.base/'bad-marker';target.mkdir()
        with closing(sqlite3.connect(self.path/'work.sqlite3')) as source,closing(sqlite3.connect(target/'work.sqlite3')) as dest:source.backup(dest)
        marker=copy.deepcopy(self.manifest);marker['stages']['h8']['labels']+=1
        with closing(sqlite3.connect(target/'work.sqlite3')) as db:
            with db:db.execute("UPDATE meta SET body=?,hash=? WHERE key='complete'",(config.canonical(marker),config.digest(marker)))
        with self.assertRaises(ValueError):p.run(target,fixture=True,collection_path=self.collection)
        self.assertFalse((target/'manifest.json').exists())

    def test_04_committed_resume_and_completed_zero_search(self):
        target=self.base/'resume'
        def interrupt(event):
            if event['event']=='midgame_label_h8_committed' and event['completed']==1:raise RuntimeError('owner durable interruption')
        with self.assertRaisesRegex(RuntimeError,'owner durable interruption'):p.run(target,fixture=True,collection_path=self.collection,on_event=interrupt)
        with closing(sqlite3.connect(target/'work.sqlite3')) as db:prefix=db.execute('SELECT state,body,hash FROM h8_attempts ORDER BY state').fetchall()
        result=p.run(target,fixture=True,collection_path=self.collection)
        with closing(sqlite3.connect(target/'work.sqlite3')) as db:self.assertEqual(prefix,db.execute('SELECT state,body,hash FROM h8_attempts ORDER BY state LIMIT 1').fetchall())
        rows,stores,_=p.load_rows(target);self.assertEqual(rows,self.rows)
        for stage in ('h8','h10','h12'):
            for key,attempt in stores[stage+'_attempts'].items():
                self.assertEqual(attempt['status'],self.stores[stage+'_attempts'][key]['status'])
                self.assertEqual([p.q_signature(p.item_for(r,rows[key])) for r in attempt['history']],[p.q_signature(p.item_for(r,rows[key])) for r in self.stores[stage+'_attempts'][key]['history']])
        def snapshot():return {file.name:(file.read_bytes(),file.stat().st_mtime_ns) for file in target.iterdir() if file.is_file() and not file.name.endswith(('-wal','-shm'))}
        before=snapshot();old=Native.analyze
        def prohibited(*args,**kwargs):raise AssertionError('Completed midgame label rerun searched')
        Native.analyze=prohibited
        try:self.assertEqual(p.run(target,fixture=True,collection_path=self.collection),result);p.load_rows(target)
        finally:Native.analyze=old
        self.assertEqual(snapshot(),before)


if __name__=='__main__':
    started=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MidgameLabelPilotAuthor))
    proof=dict(schema='midgame-label-pilot-author-acceptance-v1',passed=result.wasSuccessful(),tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),skips=len(result.skipped),
        methods=unittest.defaultTestLoader.getTestCaseNames(MidgameLabelPilotAuthor),duration_seconds=time.monotonic()-started,
        inputs={name:p.sha(ROOT/name) for name in (*p.SOURCES,'build/midgame-pilot-baseline.json','build/midgame-label-pilot-registry.json','build/midgame-label-pilot-plan.json',
            'build/midgame-data-pilot-registry.json','build/midgame-data-pilot-plan.json','build/midgame-data-pilot-implementation.json')})
    if hasattr(MidgameLabelPilotAuthor,'base'):proof['retained_directory']=MidgameLabelPilotAuthor.base.relative_to(ROOT).as_posix()
    p.teacher.atomic_json(ROOT/'build/midgame-label-pilot-author.json',proof);raise SystemExit(not result.wasSuccessful())

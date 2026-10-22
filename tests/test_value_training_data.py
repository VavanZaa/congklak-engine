import copy
import json
import os
from pathlib import Path
import sqlite3
import unittest
import uuid
from unittest import mock
from congklak import config, value_training_data as data


class TrainingDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path=Path(os.environ['VALUE_TRAINING_DATA_TEST_CACHE']).resolve() if os.environ.get('VALUE_TRAINING_DATA_TEST_CACHE') else data.ROOT/'build/value-training-v1'/('data-owner-tests-'+uuid.uuid4().hex[:8])
        cls.manifest=data.prepare(cls.path)

    def test_complete_reader_no_writer_or_extraction(self):
        with mock.patch.object(data,'Writer',side_effect=AssertionError('writer')), mock.patch.object(data,'_rows',side_effect=AssertionError('extraction')):
            self.assertEqual(data.prepare(self.path),self.manifest)

    def test_train_never_opens_fit_body(self):
        original=data.read
        def read(path):
            if Path(path).name=='fit-check.json':raise AssertionError('fit body read by train loader')
            return original(path)
        with mock.patch.object(data,'read',side_effect=read):rows,manifest=data.load_train(self.path)
        self.assertEqual(len(rows),512);self.assertEqual(len({r['component'] for r in rows}),512)
        self.assertTrue(all(r['reserved_role']=='train' and len(r['x'])==127 for r in rows))

    def test_fit_separate_and_odd_heuristic_targets(self):
        train,_=data.load_train(self.path);fit,_=data.load_fit(self.path)
        self.assertEqual(len(fit),128)
        self.assertFalse({r['component'] for r in train}&{r['component'] for r in fit})
        self.assertTrue(any(r['target_player0']%2 for r in train))
        self.assertTrue(all(r['target_z']==(r['target_player0']-r['D'])/r['r'] and -1<=r['target_z']<=1 for r in train+fit))

    def test_capped_completed_report_rejected(self):
        meta=copy.deepcopy(data.read(data.reservations.DEFAULT/'selection.json')['train'][0])
        with data.previous.ro('deeper-9000-v2') as source:
            root=source.execute('SELECT body,hash FROM roots WHERE state=?',(meta['state_key'],)).fetchone()
            item=json.loads(source.execute('SELECT body FROM labels WHERE state=?',(meta['state_key'],)).fetchone()[0])
            teacher=json.loads(source.execute("SELECT body FROM meta WHERE key='identity'").fetchone()[0])
        item['report']['completed_iteration']=False;digest=config.digest(item)
        meta['target_refs'][0]['sha256']=digest
        with sqlite3.connect(':memory:') as db:
            db.executescript('CREATE TABLE roots(state TEXT,body TEXT,hash TEXT); CREATE TABLE labels(state TEXT,body TEXT,hash TEXT);')
            db.execute('INSERT INTO roots VALUES(?,?,?)',(meta['state_key'],*root))
            db.execute('INSERT INTO labels VALUES(?,?,?)',(meta['state_key'],config.canonical(item),digest))
            with self.assertRaisesRegex(ValueError,'capped'):data._teacher(meta,db,teacher)


if __name__=='__main__':unittest.main()

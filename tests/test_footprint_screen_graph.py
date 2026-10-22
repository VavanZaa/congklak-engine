"""Disk closure against an independent bounded in-memory reference."""
import hashlib,sqlite3,unittest,uuid
from contextlib import closing
from pathlib import Path
from congklak import footprint_screen_graph as g,footprint_screen_reservations as r

def q(active,location=0):
    raw=bytearray(16);raw[location]=active;raw[7]=98-active
    return r.quotient(bytes(raw),0).hex()

class DiskClosureTests(unittest.TestCase):
    def setUp(self):
        self.folder=g.s.FOLDER/'fixtures/graph-tests'/uuid.uuid4().hex
        a,b,c,d=q(25),q(75),q(98),q(30,1)
        self.prior=dict(components={'A':dict(evaluation_blocked=False),'B':dict(evaluation_blocked=True),
                                   'isolated':dict(evaluation_blocked=False),'zero-query':dict(evaluation_blocked=True)},
            family_components={'rootA':'A','protected':'B','separate':'isolated','never-queried':'zero-query'},
            quotient_components={a:'A',b:'B',d:'isolated'})
        self.records=[(dict(family='rootA',origin='true/A'),dict(quotients=sorted((a,c)),source_refs=[],teacher_complete=True)),
                      (dict(family='cappedB',origin='true/B'),dict(quotients=sorted((b,c)),source_refs=[],teacher_complete=False))]
        self.identity=dict(schema='pure-graph-test',source_sha256=g.s.sha(Path(g.__file__)),fixture=True)

    def test_disk_partition_and_full_footprints_match_bounded_literal_graph(self):
        g.build(self.folder,self.prior,self.records,self.identity)
        disk=g.load(self.folder,self.prior,self.records,self.identity)
        reference=r._acquired_graph(self.prior,self.records)
        families=list(disk['family_components'])
        for a in families:
            for b in families:
                self.assertEqual(disk['family_components'][a]==disk['family_components'][b],
                    reference['family_components'][a]==reference['family_components'][b])
            actual=disk['components'][disk['family_components'][a]]
            expected=reference['components'][reference['family_components'][a]]
            self.assertEqual(actual['evaluation_blocked'],expected['evaluation_blocked'])
            self.assertEqual(actual['prior_components'],expected['prior_components'])
        self.assertEqual(disk['quotient_component_map']['quotient_count'],4)
        self.assertEqual(disk['quotient_component_map']['incidence_count'],4)
        self.assertTrue(disk['components'][disk['family_components']['cappedB']]['acquisition_exposed'])
        self.assertEqual(disk['components'][disk['family_components']['never-queried']]['quotient_count'],0)

    def test_cursor_resume_preserves_committed_root_and_discards_uncommitted_transaction(self):
        calls=0
        def stop():
            nonlocal calls
            calls+=1;return calls==2
        with self.assertRaises(g.Paused) as exception:g.build(self.folder,self.prior,self.records,self.identity,stop)
        self.assertEqual(exception.exception.cursor,1)
        work=self.folder/'work.sqlite3'
        with closing(sqlite3.connect(work)) as db:
            before=db.execute('SELECT * FROM roots').fetchall();self.assertEqual(len(before),1)
        db=sqlite3.connect(work);db.execute('BEGIN IMMEDIATE')
        db.execute('INSERT INTO incidence VALUES(999,?)',(bytes.fromhex(q(40)),));db.close()
        g.build(self.folder,self.prior,self.records,self.identity)
        disk=g.load(self.folder,self.prior,self.records,self.identity)
        with sqlite3.connect(self.folder/g.MAP_NAME) as db:self.assertEqual(db.execute('SELECT * FROM roots WHERE idx=0').fetchall(),before)
        self.assertEqual(disk['cursor'],2)

    def test_completed_reader_is_readonly_and_same_bytes_repeat(self):
        g.build(self.folder,self.prior,self.records,self.identity);path=self.folder/g.MAP_NAME
        before=(g.s.sha(path),path.stat().st_size,path.stat().st_mtime_ns)
        first=g.load(self.folder,self.prior,self.records,self.identity)
        second=g.load(self.folder,self.prior,self.records,self.identity)
        self.assertEqual(first,second);self.assertEqual(before,(g.s.sha(path),path.stat().st_size,path.stat().st_mtime_ns))
        with self.assertRaisesRegex(ValueError,'source identity'):
            g.load(self.folder,self.prior,self.records,dict(self.identity,fixture=False))

    def test_extra_quotient_cannot_be_resealed_into_certified_lookup(self):
        g.build(self.folder,self.prior,self.records,self.identity)
        with sqlite3.connect(self.folder/g.MAP_NAME) as db:
            db.execute('INSERT INTO keys VALUES(?,?,?,?)',(bytes.fromhex(q(45)),1,0,1))
        with self.assertRaisesRegex(ValueError,'invented or wrong first-source'):
            g.load(self.folder,self.prior,self.records,self.identity)

    def test_corrupt_parent_cycle_and_low_alias_source_substitution_are_refused(self):
        g.build(self.folder,self.prior,self.records,self.identity)
        with sqlite3.connect(self.folder/g.MAP_NAME) as db:db.execute('UPDATE owners SET parent=id+1 WHERE id=1');db.execute('UPDATE owners SET parent=1 WHERE id=2')
        with self.assertRaisesRegex(ValueError,'cycle|partition differs'):
            g.load(self.folder,self.prior,self.records,self.identity)

    def test_shared_low_cannot_enter_disk_connector_tables(self):
        changed=list(self.records);changed[0]=(changed[0][0],dict(changed[0][1],quotients=[q(24)]))
        with self.assertRaisesRegex(ValueError,'Shared low'):
            g.build(self.folder,self.prior,changed,self.identity)

if __name__=='__main__':unittest.main()

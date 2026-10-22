"""CUDA recovery for the versioned depth-six training lineage."""
from contextlib import closing
import sqlite3
from pathlib import Path
import tempfile
import unittest
import torch
from congklak import training_v2 as tr
from congklak.native import ROOT

def saved(path):
    with closing(sqlite3.connect(path/"catalog.sqlite3")) as db: name=db.execute("SELECT name FROM checkpoints ORDER BY id DESC LIMIT 1").fetchone()[0]
    return torch.load(path/name,map_location="cpu",weights_only=True)
def equal(a,b):
    if torch.is_tensor(a): return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)): return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

class CurriculumAcceptance(unittest.TestCase):
    def test_01_cuda_restore_and_isolated_depth_six_loader(self):
        self.assertTrue(torch.cuda.is_available())
        _,data,_=tr.load_data()
        self.assertEqual({s:len(r) for s,r in data.items()},{"train":8000,"development":1000})
        self.assertTrue(all(r["label"]["choice_depth"]==6 for rows in data.values() for r in rows))
        work=ROOT/"build/curriculum-acceptance"; work.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=work) as temp:
            path=Path(temp); args={"steps":40,"seed":7,"width":256,"batch_size":32,"checkpoint_every":10,"device":"cuda"}
            fresh=tr.train(path/"fresh",**args); stop=[False]
            def event(e):
                if e["event"]=="checkpoint_committed" and e["step"]==10: stop[0]=True
            interrupted=tr.train(path/"resumed",**args,on_event=event,cancel=lambda:stop[0])
            self.assertEqual(interrupted["status"],"interrupted")
            resumed=tr.train(path/"resumed",**args)
            self.assertEqual(resumed["weights_sha256"],fresh["weights_sha256"])
            self.assertTrue(equal(saved(path/"resumed"),saved(path/"fresh")))

if __name__=="__main__": unittest.main()

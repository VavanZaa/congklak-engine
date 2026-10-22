"""Product finder acceptance: HTTP isolation and real killed-server recovery."""
from __future__ import annotations

import hashlib
import http.client
import json
import os
from pathlib import Path
import queue
import sqlite3
import subprocess
import sys
import threading
import time
import unittest
import uuid

from congklak.dakon_opening import ROOT, Run, envelope, recipe
from congklak.dakon_opening_verify import canonical, read_certificate, verify
from congklak.product_finder_v1 import Finder
from congklak.product_v2 import Product, make_server

RECEIPTS = ROOT/'build'/'product-finder-v1'
BASE = ROOT/'build'/'dakon-opening-v1'/'browser-tests'


def fresh():
    return BASE/uuid.uuid4().hex


def settled(finder, seconds=20):
    end = time.monotonic()+seconds
    while time.monotonic() < end:
        state = finder.snapshot()
        if state['active_run'] is None:
            return state
        time.sleep(.01)
    raise AssertionError('Finder did not settle')


def request(port, route='/api/finder', payload=None, token=None, headers=None):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=20)
    hdr = dict(headers or {})
    if payload is None:
        connection.request('GET', route, headers=hdr)
    else:
        hdr.update({'Content-Type':'application/json', 'X-Congklak-Token':token or ''})
        connection.request('POST', route, json.dumps(payload), hdr)
    response = connection.getresponse(); raw = response.read()
    result = (response.status, raw)
    connection.close()
    return result


def logical(path):
    result = {}
    for relative in ('boards.sqlite3', 'analysis/analysis.sqlite3'):
        with sqlite3.connect(path/relative) as db:
            for (name,) in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"):
                rows = [list(row) for row in db.execute('SELECT * FROM "'+name+'"')]
                result[relative+':'+name] = sorted(rows, key=canonical)
    return result


class BrowserFinder(unittest.TestCase):
    def test_identity_only_initialization_recovers_known_opening(self):
        root=fresh();rid=uuid.uuid4().hex;path=root/rid;path.mkdir(parents=True)
        (path/'identity.json').write_text(canonical(envelope(recipe(7,'majority',0,10000))))
        finder=Finder(root)
        try:
            saved=finder.snapshot()['runs'][0]
            self.assertEqual(saved['simulated_actions'],0)
            self.assertTrue(saved['resumable'])
            self.assertIsNone(finder.snapshot()['active_run'])
            finder.command(dict(op='resume',run_id=rid))
            self.assertEqual(settled(finder)['runs'][0]['outcome'],'guaranteed_win')
        finally:finder.close()

    def test_strict_commands_and_one_writer(self):
        finder = Finder(fresh(), search_options={'max_actions':1})
        try:
            for n in (0,65,True,7.0,'7',None):
                with self.assertRaises(ValueError):
                    finder.command(dict(op='start',n=n,goal='majority',starter=0))
            for req in ({},[],dict(op='start',n=7,goal='../bad',starter=0),
                        dict(op='start',n=7,goal='majority',starter=2),
                        dict(op='start',n=7,goal='majority',starter=0,path='bad'),
                        dict(op='resume',run_id='../bad')):
                with self.assertRaises((ValueError,TypeError)):finder.command(req)
            self.assertEqual(finder.snapshot()['runs'],[])
            result = finder.command(dict(op='start',n=7,goal='majority',starter=0))
            rid = result['active_run']; settled(finder)
            with Run(finder.root/rid):
                finder.command(dict(op='resume',run_id=rid)); result=settled(finder)
                self.assertIn('Another finder owns',result['runs'][0]['error'])
            finder.command(dict(op='resume',run_id=rid)); self.assertIsNone(settled(finder)['runs'][0]['error'])
        finally:finder.close()

    def test_bounded_work_restart_and_verified_replay(self):
        root = fresh(); finder = Finder(root,search_options={'max_actions':5})
        first = finder.command(dict(op='start',n=7,goal='majority',starter=0))
        rid = first['active_run']; before=settled(finder)['runs'][0]
        self.assertEqual(before['simulated_actions'],5)
        self.assertEqual(before['outcome'],'unknown')
        with self.assertRaisesRegex(ValueError,'no completed'):finder.certificate(rid)
        finder.close()
        finder = Finder(root)
        try:
            restored=finder.snapshot()
            self.assertIsNone(restored['active_run'])
            self.assertEqual(restored['runs'][0]['simulated_actions'],5)
            finder.command(dict(op='resume',run_id=rid)); result=settled(finder)['runs'][0]
            self.assertEqual(result['outcome'],'guaranteed_win')
            self.assertEqual(result['simulated_actions'],126)
            self.assertNotIn('certificate',result)
            exported=finder.certificate(rid)
            self.assertTrue(exported['verification']['passed'])
            self.assertEqual(exported['verification']['starter_store'],54)
            self.assertEqual(exported['certificate'],read_certificate(ROOT/'data/dakon-opening-v1/n7-majority-A/certificate.json'))
            files={p.name:p.read_bytes() for p in (root/rid).glob('*.json')}
            finder.command(dict(op='resume',run_id=rid))
            self.assertEqual(files,{p.name:p.read_bytes() for p in (root/rid).glob('*.json')})
            bad=exported['certificate'];bad['body']['actions'][0]=6
            (root/rid/'certificate.json').write_text(canonical(envelope(bad['body'])))
            with self.assertRaises(ValueError):finder.certificate(rid)
            self.assertEqual(finder.snapshot()['runs'][0]['outcome'],'unknown')
        finally:finder.close()

    def test_pause_active_work_and_shutdown_checkpoint(self):
        reached=threading.Event(); release=threading.Event()
        def hook(stage):
            if stage == 'after_checkpoint_replace' and not reached.is_set():
                reached.set(); release.wait(10)
        finder=Finder(fresh(),search_options={'checkpoint_actions':1},hook=hook)
        try:
            result=finder.command(dict(op='start',n=12,goal='majority',starter=1))
            rid=result['active_run']; self.assertTrue(reached.wait(5))
            with self.assertRaisesRegex(ValueError,'Pause the current'):
                finder.command(dict(op='start',n=7,goal='majority',starter=0))
            started=time.monotonic(); paused=finder.command(dict(op='pause',run_id=rid))
            self.assertLess(time.monotonic()-started,.2)
            self.assertTrue(paused['runs'][0]['pausing'])
            release.set(); result=settled(finder)['runs'][0]
            self.assertEqual(result['stop_reason'],'cancelled')
            self.assertEqual(result['simulated_actions'],1)
            finder.command(dict(op='resume',run_id=rid));finder.close()
            restored=Finder(finder.root)
            try:
                self.assertIsNone(restored.snapshot()['active_run'])
                self.assertGreaterEqual(restored.snapshot()['runs'][0]['simulated_actions'],1)
            finally:restored.close()
        finally:release.set();finder.close()

    def test_both_players_and_stronger_target_and_no_witness(self):
        finder=Finder(fresh())
        try:
            for n,goal,starter,store in ((7,'majority',1,54),(7,'refill_threshold',0,92)):
                rid=finder.command(dict(op='start',n=n,goal=goal,starter=starter))['active_run']
                self.assertEqual(settled(finder)['runs'][0]['outcome'],'guaranteed_win')
                cert=finder.certificate(rid)
                self.assertEqual(cert['verification']['starter_store'],store)
                self.assertEqual(cert['certificate']['body']['steps'][-1]['cells'][starter*(n+1)+n],store)
            rid=finder.command(dict(op='start',n=1,goal='majority',starter=0))['active_run']
            done=settled(finder)['runs'][0]
            self.assertEqual(done['outcome'],'no_first_turn_witness')
            with self.assertRaises(ValueError):finder.certificate(rid)
        finally:finder.close()

    def test_corrupt_latest_checkpoint_falls_back_without_winning_claim(self):
        root=fresh(); finder=Finder(root,search_options={'max_actions':5,'checkpoint_actions':2})
        rid=finder.command(dict(op='start',n=7,goal='majority',starter=0))['active_run']
        prior=settled(finder)['runs'][0];finder.close()
        slots=list((root/rid).glob('checkpoint-*.json'))
        latest=max(slots,key=lambda p:json.loads(p.read_text())['body']['generation'])
        latest.write_text('broken')
        finder=Finder(root)
        try:
            restored=finder.snapshot()['runs'][0]
            self.assertTrue(restored['recovered_from_older_checkpoint'])
            self.assertLess(restored['generation'],prior['generation'])
            self.assertEqual(restored['outcome'],'unknown')
            finder.command(dict(op='resume',run_id=rid));self.assertEqual(settled(finder)['runs'][0]['outcome'],'guaranteed_win')
        finally:finder.close()

    def test_http_security_and_existing_games_unchanged(self):
        path=ROOT/'data/product-v2-fixtures'/('finder-'+uuid.uuid4().hex)
        product=Product(path)
        product.command(dict(op='apply',revision=0,action=0))
        product.command(dict(op='save',name='Preserved before finder'))
        before=logical(path)
        server=make_server(product,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        port=server.server_port
        try:
            for headers in ({'Host':'evil.example'},{'Origin':'https://evil.example'}):
                self.assertEqual(request(port,headers=headers)[0],403)
            self.assertEqual(request(port,'/finder.js')[0],200)
            self.assertIn(b'Opening finder',request(port,'/')[1])
            self.assertEqual(request(port,'/api/finder?path=bad')[0],400)
            self.assertEqual(request(port,'/api/finder/../certificate')[0],400)
            payload=dict(op='start',n=7,goal='majority',starter=0)
            self.assertEqual(request(port,'/api/finder/command',payload)[0],403)
            code,raw=request(port);self.assertEqual(code,200); token=json.loads(raw)['token']
            code,raw=request(port,'/api/finder/command',payload,token)
            self.assertEqual(code,200);rid=json.loads(raw)['active_run']
            settled(product.finder())
            code,raw=request(port,f'/api/finder/{rid}/certificate')
            self.assertEqual(code,200);self.assertTrue(json.loads(raw)['verification']['passed'])
            self.assertEqual(logical(path),before)
        finally:server.shutdown();server.server_close();thread.join(5);product.close()
        product=Product(path)
        try:
            self.assertEqual(logical(path),before)
            self.assertEqual(product.snapshot()['sessions'][0]['name'],'Preserved before finder')
            self.assertEqual(product.finder().snapshot()['runs'][0]['outcome'],'guaranteed_win')
        finally:product.close()


class FinderProcessRecovery(unittest.TestCase):
    def test_actual_server_kills_at_checkpoint_and_certificate_boundaries(self):
        RECEIPTS.mkdir(parents=True,exist_ok=True)
        receipts=[]
        for stage in ('before_checkpoint_replace','after_checkpoint_replace','after_certificate_publish'):
            path=ROOT/'data/product-v2-fixtures'/('finder-kill-'+uuid.uuid4().hex)
            marker=RECEIPTS/(path.name+'.marker')
            children=[]
            def launch(trap=False):
                # Only this test bootstrap adds a deterministic crash boundary.
                code='''
import sys, threading
from pathlib import Path
from congklak.product_finder_v1 import Finder
from congklak.product_v2 import main
stage, marker, data, trap = sys.argv[1:]
if trap == 'yes':
 original = Finder.__init__
 counts = {}
 def hook(where):
  if where == stage:
   counts[where] = counts.get(where, 0)+1
   if where != 'after_certificate_publish' and counts[where] < 2: return
   Path(marker).write_text(where)
   threading.Event().wait()
 def init(self, root): original(self, root, search_options={'checkpoint_actions':4}, hook=hook)
 Finder.__init__ = init
sys.argv = ['product', '--port', '0', '--data-dir', data]
raise SystemExit(main())
'''
                child=subprocess.Popen([sys.executable,'-B','-X','utf8','-u','-c',code,stage,str(marker),str(path),'yes' if trap else 'no'],
                    cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                children.append(child);q=queue.Queue()
                threading.Thread(target=lambda:q.put(child.stdout.readline()),daemon=True).start()
                ready=json.loads(q.get(timeout=20));self.assertEqual(ready['event'],'product_ready',ready)
                return child,int(ready['url'].split(':')[-1].rstrip('/'))
            try:
                child,port=launch(True)
                state=json.loads(request(port,'/api/state')[1])
                for payload in (dict(op='apply',revision=state['revision'],action=0),dict(op='save',name='Keep through kill')):
                    self.assertEqual(request(port,'/api/command',payload,state['token'])[0],200)
                before=logical(path)
                payload=dict(op='start',n=7,goal='majority',starter=0)
                code,raw=request(port,'/api/finder/command',payload,state['token']);self.assertEqual(code,200)
                rid=json.loads(raw)['active_run']
                deadline=time.monotonic()+15
                while not marker.exists() and time.monotonic()<deadline:time.sleep(.01)
                self.assertTrue(marker.exists(),stage)
                child.kill();child.wait(5)
                self.assertEqual(logical(path),before)
                child,port=launch(False)
                code,raw=request(port);saved=json.loads(raw)
                self.assertEqual(code,200);self.assertIsNone(saved['active_run'])
                checkpoint=saved['runs'][0]
                self.assertEqual(checkpoint['outcome'],'unknown')
                self.assertGreater(checkpoint['generation'],0)
                self.assertEqual(logical(path),before)
                self.assertEqual(request(port,'/api/finder/command',dict(op='resume',run_id=rid),saved['token'])[0],200)
                deadline=time.monotonic()+20
                while time.monotonic()<deadline:
                    saved=json.loads(request(port)[1])
                    if saved['active_run'] is None:break
                    time.sleep(.02)
                self.assertEqual(saved['runs'][0]['outcome'],'guaranteed_win',saved)
                exported=json.loads(request(port,f'/api/finder/{rid}/certificate')[1])
                self.assertTrue(verify(exported['certificate'])['passed'])
                reference=read_certificate(ROOT/'data/dakon-opening-v1/n7-majority-A/certificate.json')
                self.assertEqual(exported['certificate'],reference)
                self.assertEqual(logical(path),before)
                receipts.append(dict(stage=stage,passed=True,run_id=rid,
                    recovered_generation=checkpoint['generation'],
                    recovered_actions=checkpoint['simulated_actions'],
                    identical_certificate=True,saved_tables_unchanged=True))
            finally:
                for child in children:
                    if child.poll() is None:child.kill();child.wait(5)
                    child.stdout.close();child.stderr.close()
        (RECEIPTS/'recovery.json').write_text(json.dumps(dict(passed=True,cases=receipts),indent=2)+'\n')


if __name__ == '__main__':unittest.main()

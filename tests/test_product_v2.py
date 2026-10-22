"""Bounded local product acceptance; all user fixtures are non-training data."""
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

from congklak import config
from congklak.native import ROOT
from congklak.product_v2 import Product, make_server

FIXTURES = ROOT/'data'/'product-v2-fixtures'
RECEIPTS = ROOT/'build'/'product-v2'
SMALL = dict(pits=[[0,1,0,0,0,0,0],[0,0,0,0,0,1,0]], stores=[48,48],controller=0)
PASS = dict(pits=[[0]*7,[0]*6+[1]],stores=[49,48],controller=0)


def limits(depth=4,ms=100):
    return dict(max_depth=depth,time_ms=ms,node_budget=1000000,memory_mb=512,tt_mb=16,use_pvs=True)


def wait_done(product, seconds=5):
    end=time.monotonic()+seconds
    while time.monotonic()<end:
        state=product.snapshot()
        if not state['active']:
            return state
        time.sleep(.01)
    raise AssertionError('Product analysis did not stop')


class ProductAcceptance(unittest.TestCase):
    def setUp(self):
        self.path=FIXTURES/uuid.uuid4().hex
        self.product=Product(self.path)

    def tearDown(self):
        self.product.close()

    def cmd(self,op,**extra):
        return self.product.command(dict(op=op,revision=self.product.revision,**extra))

    def test_isolated_paths_and_strict_import(self):
        for path in (ROOT/'data'/'prospective-benchmark-v5',ROOT/'data'/'product-v2-fixtures'/'..'/'pilot-10000'):
            with self.assertRaisesRegex(config.InputError,'Product data must'):
                Product(path)
        old=self.product.snapshot()['position']
        for bad in (None,{},dict(pits=[[7]*7,[7]*7],stores=[1,0],controller=0)):
            with self.assertRaises(config.InputError):
                self.cmd('import',position=bad)
        self.assertEqual(old,self.product.position)
        with self.assertRaises(config.InputError):
            self.product.command(dict(op='opening',revision=-1))

    def test_pass_and_terminal_board_play(self):
        value=self.cmd('import',position=PASS)
        self.assertTrue(value['forced_pass'])
        self.assertEqual(value['legal_actions'],[])
        with self.assertRaisesRegex(config.InputError,'Pass before'):
            self.cmd('apply',action=6)
        value=self.cmd('apply',action='pass')
        self.assertEqual(value['position']['controller'],1)
        value=self.cmd('apply',action=6)
        self.assertTrue(value['terminal'])
        self.assertEqual(value['position']['stores'],[49,49])
        with self.assertRaises(config.InputError):
            self.cmd('analyze',limits=limits())

    def test_saved_board_and_job_survive_restart(self):
        self.cmd('import',position=SMALL)
        self.cmd('save',name='Two-seed test')
        self.cmd('analyze',limits=limits())
        value=wait_done(self.product)
        self.assertIsNotNone(value['analysis']['report'])
        job=value['last_job']; expected=value['analysis']['report']
        self.product.close(); self.product=Product(self.path)
        recovered=self.product.snapshot()
        self.assertEqual(recovered['analysis']['report'],expected)
        self.assertEqual(recovered['position'],SMALL)
        self.cmd('opening')
        value=self.cmd('load_session',session_id=recovered['sessions'][0]['id'])
        self.assertEqual(value['position'],SMALL)
        value=self.cmd('load_job',job_id=job)
        self.assertTrue(value['analysis']['current'])
        self.cmd('resume',job_id=job,limits=limits())
        self.assertEqual(wait_done(self.product)['analysis']['report'],expected)

    def test_current_recommendation_and_stale_guard(self):
        self.cmd('import',position=SMALL)
        self.cmd('analyze',limits=limits())
        value=wait_done(self.product)
        self.assertTrue(value['analysis']['report']['best_moves'])
        self.cmd('opening')
        with self.assertRaisesRegex(config.InputError,'No recommendation belongs'):
            self.cmd('apply_best')
        self.assertFalse(self.product.snapshot()['analysis']['current'])

    def test_async_stop_and_board_race(self):
        self.cmd('live')
        self.assertTrue(self.product.snapshot()['active'])
        # A user move interrupts analysis, without accidentally playing its best move.
        started=time.monotonic()
        value=self.cmd('apply',action=0)
        self.assertFalse(value['active'])
        self.assertEqual(value['revision'],1)
        self.assertEqual(value['position']['pits'][0][0],0)
        self.assertEqual(len(value['history']['entries']),1)
        self.assertTrue(value['analysis'] is None or not value['analysis']['current'])
        self.assertLess(time.monotonic()-started,3)
        live=self.cmd('live')
        self.assertTrue(live['active'])
        self.assertEqual(self.product.store.load(live['last_job'])['payload']['position'],value['position'])
        started=time.monotonic(); self.cmd('stop')
        value=wait_done(self.product)
        self.assertLess(time.monotonic()-started,3)
        self.assertFalse(value['active'])
        self.assertEqual(len(value['history']['entries']),1)

    def test_bonus_turn_and_regular_turn_have_explanations(self):
        value=self.cmd('apply',action=0)
        self.assertEqual(value['position']['controller'],0)
        self.assertIn("A's store",value['history']['turn_message'])
        self.assertIn('another move',value['history']['turn_message'])
        self.assertEqual(value['history']['entries'][0]['action'],0)
        # Find a genuine turn change from legal native transitions.
        for _ in range(20):
            normal=self.product.native.normalize(__import__('congklak.native',fromlist=['flatten']).flatten(self.product.position),actor=self.product.position['controller'],total=98)
            if normal['terminal']:break
            old=self.product.position['controller']
            value=self.cmd('apply',action='pass' if normal['passes'] else normal['legal_actions'][0])
            if value['position']['controller']!=old:
                self.assertIn('moves next',value['history']['turn_message'])
                break
        else:self.fail('No normal turn change observed')

    def test_history_save_load_restart_and_illegal_moves(self):
        value=self.cmd('apply',action=0)
        history=value['history']; position=value['position']
        self.cmd('save',name='Bonus move')
        saved=self.product.snapshot()['sessions'][0]['id']
        for action in (0,-1,7,True):
            with self.assertRaises(config.InputError):self.cmd('apply',action=action)
        with self.assertRaises(config.InputError):
            self.product.command(dict(op='apply',revision=0,action=1))
        self.assertEqual(self.product.snapshot()['history'],history)
        self.product.close();self.product=Product(self.path)
        self.assertEqual(self.product.snapshot()['history'],history)
        self.cmd('opening')
        self.assertEqual(self.product.snapshot()['history']['entries'],[])
        loaded=self.cmd('load_session',session_id=saved)
        self.assertEqual(loaded['history']['entries'],history['entries'])
        self.assertEqual(loaded['position'],position)
        self.product.close();self.product=Product(self.path)
        self.assertEqual(self.product.snapshot()['history']['entries'],history['entries'])

    def test_pass_history_and_automatic_opponent_pass(self):
        self.cmd('import',position=PASS)
        value=self.cmd('apply',action='pass')
        entry=value['history']['entries'][0]
        self.assertEqual(entry['action'],'pass')
        self.assertEqual(entry['actor'],0)
        self.assertIn('row is empty',entry['explanation'])
        # A lands in an empty opponent pit. B must pass and A continues.
        p=dict(pits=[[0,0,0,0,0,0,2],[0]*7],stores=[48,48],controller=0)
        self.cmd('import',position=p)
        value=self.cmd('apply',action=6)
        self.assertEqual(value['position']['controller'],1)  # B now has a seed and can play.
        # Own-empty landing with an empty opposite, while B has no seeds, passes B back to A.
        p=dict(pits=[[1,0,1,0,0,0,0],[0]*7],stores=[48,48],controller=0)
        self.cmd('import',position=p)
        value=self.cmd('apply',action=0)
        self.assertEqual(value['position']['controller'],0)
        self.assertEqual(value['history']['entries'][0]['passes'],[dict(**{'from':1,'to':0})])
        self.assertIn('B\'s row is empty',value['history']['turn_message'])

    def test_migrate_v1_board_without_fabricating_old_moves(self):
        self.cmd('apply',action=0)
        position=self.product.position
        self.product.close()
        with sqlite3.connect(self.path/'boards.sqlite3') as db:
            db.execute('DROP TABLE play_history')
        self.product=Product(self.path)
        state=self.product.snapshot()
        self.assertEqual(state['position'],position)
        self.assertEqual(state['history']['entries'],[])
        self.assertIn('Earlier moves were not recorded',state['history']['note'])

    def test_live_resume_and_autoplay_race(self):
        self.cmd('live')
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            value=self.product.snapshot()
            if value['analysis'] and value['analysis']['report']:break
            time.sleep(.01)
        self.assertIsNotNone(value['analysis']['report'])
        job=value['last_job'];depth=value['analysis']['report']['choice_depth']
        self.cmd('stop');wait_done(self.product)
        value=self.cmd('live')
        self.assertEqual(value['last_job'],job)
        self.cmd('stop');value=wait_done(self.product)
        self.assertGreaterEqual(value['analysis']['report']['choice_depth'],depth)
        # Completed engine_turn event is queued but not yet drained. A new board action wins.
        self.cmd('import',position=SMALL)
        self.cmd('engine_turn',limits=limits(4,100))
        self.product.coord.join()
        value=self.cmd('opening')
        self.assertEqual(value['position'],config.position())
        self.assertEqual(value['history']['entries'],[])

    def test_engine_turn_applies_only_a_current_completed_move(self):
        self.cmd('import',position=SMALL)
        old=self.product.revision
        self.cmd('engine_turn',limits=limits())
        value=wait_done(self.product)
        self.assertEqual(value['revision'],old+1)
        self.assertEqual(sum(value['position']['stores'])+sum(map(sum,value['position']['pits'])),98)
        self.assertFalse(value['analysis']['current'])

    def test_resource_configuration_fixed(self):
        for extra in ({'memory_mb':24576},{'tt_mb':0},{'use_pvs':False},{'time_ms':0},{'max_depth':129}):
            settings=limits(); settings.update(extra)
            with self.assertRaises(config.InputError):
                self.cmd('analyze',limits=settings)

    def test_http_security_assets_and_api(self):
        server=make_server(self.product,0)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        port=server.server_port
        def request(path='/',method='GET',body=None,headers=None):
            conn=http.client.HTTPConnection('127.0.0.1',port,timeout=5)
            conn.request(method,path,body,headers or {})
            response=conn.getresponse(); raw=response.read(); result=(response.status,dict(response.getheaders()),raw);conn.close();return result
        try:
            status,headers,raw=request()
            self.assertEqual(status,200);self.assertIn(b'Find your next move',raw)
            self.assertIn('frame-ancestors',headers['Content-Security-Policy'])
            self.assertEqual(request('/../README.md')[0],404)
            self.assertEqual(request('/api/state',headers={'Host':'evil.example'})[0],403)
            self.assertEqual(request('/api/state',headers={'Origin':'https://evil.example'})[0],403)
            state=json.loads(request('/api/state')[2])
            self.assertFalse(state['research_training_eligible'])
            self.assertEqual(state['identity']['evaluator'],'H_v1')
            self.assertEqual(state['identity']['rules_digest'],config.RULES_DIGEST)
            headers={'Content-Type':'application/json','Origin':f'http://127.0.0.1:{port}'}
            self.assertEqual(request('/api/command','POST','{}',headers)[0],403)
            headers['X-Congklak-Token']=state['token']
            self.assertEqual(request('/api/command','POST','x'*65537,headers)[0],400)
            self.assertEqual(request('/api/command','POST','['*1500+'0'+']'*1500,headers)[0],400)
            self.assertEqual(request('/api/command','POST','{"op":"opening","revision":0}',headers)[0],200)
            self.assertNotIn('token',json.loads(request('/api/export')[2]))
            self.assertIn(b"command('stop')",request('/app.js')[2])
        finally:
            server.shutdown();server.server_close();thread.join(3)


class ProcessRecovery(unittest.TestCase):
    def test_actual_product_kill_restart_resume(self):
        path=FIXTURES/('recovery-'+uuid.uuid4().hex)
        commands=[]; children=[]
        def launch():
            command=[sys.executable,'-B','-u','-m','congklak.product_v2','--port','0','--data-dir',str(path)]
            child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            children.append(child);commands.append(command)
            q=queue.Queue();threading.Thread(target=lambda:q.put(child.stdout.readline()),daemon=True).start()
            line=q.get(timeout=15);ready=json.loads(line)
            self.assertEqual(ready['event'],'product_ready',ready)
            port=int(ready['url'].split(':')[-1].rstrip('/'))
            return child,port
        def request(port,payload=None):
            connection=http.client.HTTPConnection('127.0.0.1',port,timeout=5)
            if payload is None:
                connection.request('GET','/api/state')
            else:
                current=request(port)
                data=dict(payload,revision=current['revision'])
                connection.request('POST','/api/command',json.dumps(data),{'Content-Type':'application/json','Origin':f'http://127.0.0.1:{port}','X-Congklak-Token':current['token']})
            response=connection.getresponse(); value=json.loads(response.read());connection.close()
            self.assertEqual(response.status,200,value)
            return value
        try:
            child,port=launch()
            request(port,dict(op='apply',action=0))
            request(port,dict(op='live'))
            deadline=time.monotonic()+10
            while time.monotonic()<deadline:
                before=request(port)
                if before['active'] and before['analysis'] and before['analysis']['report']:
                    break
                time.sleep(.01)
            else:
                self.fail('Could not observe a live analysis with a completed checkpoint')
            child.kill(); killed_code=child.wait(timeout=10)
            self.assertNotEqual(killed_code,0)
            # A newer complete iteration may commit between HTTP observation
            # and termination. Compare the actual durable post-kill report.
            with sqlite3.connect((path/'analysis'/'analysis.sqlite3').as_uri()+'?mode=ro',uri=True) as checkpoint:
                row=checkpoint.execute('SELECT body,body_hash FROM reports WHERE job_id=? ORDER BY depth DESC LIMIT 1',(before['last_job'],)).fetchone()
                expected=json.loads(row[0])
                self.assertEqual(config.digest(expected),row[1])
            restarted,port=launch()
            after=request(port)
            self.assertEqual(after['position'],before['position'])
            self.assertEqual(after['last_job'],before['last_job'])
            self.assertEqual(after['history'],before['history'])
            self.assertEqual(len(after['history']['entries']),1)
            self.assertEqual(after['analysis']['report'],expected)
            depth=after['analysis']['report']['choice_depth']
            resumed=request(port,dict(op='resume',job_id=after['last_job'],limits=limits(depth,100)))
            self.assertFalse(resumed['active'])
            self.assertEqual(resumed['analysis']['report'],expected)
            request(port,dict(op='resume',job_id=after['last_job'],limits=limits(depth+1,100)))
            end=time.monotonic()+5
            while time.monotonic()<end:
                complete=request(port)
                if not complete['active']:break
                time.sleep(.02)
            self.assertFalse(complete['active'])
            self.assertGreaterEqual(complete['analysis']['report']['choice_depth'],depth)
            receipt=dict(schema='congklak-product-v2-process-recovery',passed=True,actual_process_kill=True,actual_process_kills=1,
                         killed_pid=child.pid,restarted_pid=restarted.pid,killed_exit_code=killed_code,
                         saved_position_equal=True,saved_job_equal=True,saved_report_equal=True,saved_history_equal=True,
                         original_completed_depth=depth,resumed_completed_depth=complete['analysis']['report']['choice_depth'],
                         observed_before_kill_depth=before['analysis']['report']['choice_depth'],durable_killed_report_sha256=row[1],
                         same_horizon_resume_no_search=True,further_resume_stopped_cleanly=True,
                         product_data_dir=str(path.relative_to(ROOT)).replace('\\','/'),
                         research_training_eligible=False,models_and_policy_fixed=True,commands=commands,
                         sources={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (ROOT/'congklak'/'product_v2.py',Path(__file__))})
            RECEIPTS.mkdir(parents=True,exist_ok=True)
            if (RECEIPTS/'recovery.json').exists():
                history=RECEIPTS/'recovery-history';history.mkdir(exist_ok=True)
                old=(RECEIPTS/'recovery.json').read_bytes()
                (history/(hashlib.sha256(old).hexdigest()+'.json')).write_bytes(old)
            (RECEIPTS/'recovery.json').write_text(config.canonical(receipt)+'\n',encoding='utf-8')
        finally:
            for child in children:
                if child.poll() is None:child.kill();child.wait(timeout=10)
                for pipe in (child.stdout,child.stderr):
                    if pipe:pipe.close()


if __name__ == '__main__':
    unittest.main()

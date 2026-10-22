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
from congklak.product_v1 import Product, make_server

FIXTURES = ROOT/'data'/'product-v1-fixtures'
RECEIPTS = ROOT/'build'/'product-v1'
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
        for path in (ROOT/'data'/'prospective-benchmark-v5',ROOT/'data'/'product-v1-fixtures'/'..'/'pilot-10000'):
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
        self.cmd('analyze',limits=limits(32,3000))
        self.assertTrue(self.product.snapshot()['active'])
        with self.assertRaisesRegex(config.InputError,'Analysis is running'):
            self.cmd('opening')
        started=time.monotonic(); self.cmd('stop')
        value=wait_done(self.product)
        self.assertLess(time.monotonic()-started,3)
        self.assertFalse(value['active'])
        self.assertEqual(value['position'],config.position())

    def test_engine_turn_applies_only_a_current_completed_move(self):
        self.cmd('import',position=SMALL)
        old=self.product.revision
        self.cmd('engine_turn',limits=limits())
        value=wait_done(self.product)
        self.assertEqual(value['revision'],old+1)
        self.assertEqual(sum(value['position']['stores'])+sum(map(sum,value['position']['pits'])),98)
        self.assertFalse(value['analysis']['current'])

    def test_resource_configuration_fixed(self):
        for extra in ({'memory_mb':24576},{'tt_mb':0},{'use_pvs':False},{'time_ms':0},{'max_depth':128}):
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
            command=[sys.executable,'-B','-u','-m','congklak.product_v1','--port','0','--data-dir',str(path)]
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
            request(port,dict(op='analyze',limits=limits(32,10000)))
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
            receipt=dict(schema='congklak-product-v1-process-recovery',passed=True,actual_process_kill=True,actual_process_kills=1,
                         killed_pid=child.pid,restarted_pid=restarted.pid,killed_exit_code=killed_code,
                         saved_position_equal=True,saved_job_equal=True,saved_report_equal=True,
                         original_completed_depth=depth,resumed_completed_depth=complete['analysis']['report']['choice_depth'],
                         observed_before_kill_depth=before['analysis']['report']['choice_depth'],durable_killed_report_sha256=row[1],
                         same_horizon_resume_no_search=True,further_resume_stopped_cleanly=True,
                         product_data_dir=str(path.relative_to(ROOT)).replace('\\','/'),
                         research_training_eligible=False,models_and_policy_fixed=True,commands=commands,
                         sources={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (ROOT/'congklak'/'product_v1.py',Path(__file__))})
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

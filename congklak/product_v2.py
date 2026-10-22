"""Local Betawi board client over the unchanged H_v1 coordinator.

User games live in a separate product directory and are not research examples.
No learning, neural runtime, benchmark admission or experimental modules load here.
"""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import secrets
import signal
import sqlite3
import threading
import time
import uuid
from urllib.parse import urlsplit

from . import config
from .__main__ import Coordinator
from .explanations import explain
from .native import Native, ROOT, flatten
from .storage import Store
from . import product_rules_v1 as product_rules

ASSETS = ROOT / "product_v2"
MAX_BODY = 65536
VERSION = "congklak-local-product-v2"
LIVE_LIMITS = dict(max_depth=128, time_ms=3600000, node_budget=0,
                   memory_mb=512, tt_mb=16, use_pvs=True)


def need(condition, message, code="invalid_request"):
    if not condition:
        raise config.InputError(code, message)


class EventsCoordinator(Coordinator):
    def __init__(self, native, store, events):
        super().__init__(native, store)
        self.events = events

    def emit(self, event, request_id=None, **values):
        self.events.put(dict(event=event, **values))


class Product:
    def __init__(self, data_dir):
        config.verify_rules()
        self.path = Path(data_dir).resolve()
        need(self.path == ROOT.resolve()/'data'/'product-v1' or
             self.path.is_relative_to(ROOT.resolve()/'data'/'product-v2-fixtures'),
             "Product data must use data/product-v1 or data/product-v2-fixtures.", 'unsafe_data_directory')
        self.path.mkdir(parents=True, exist_ok=True)
        self.native = Native()
        self.store = Store(self.path / "analysis", config.identity(self.native.build_id), native=self.native)
        self.events = queue.SimpleQueue()
        self.coord = EventsCoordinator(self.native, self.store, self.events)
        self.lock = threading.RLock()
        self.token = secrets.token_urlsafe(32)
        self._finder = None
        self.latest = None
        self.message = "Ready. Select a pit or analyze this position."
        self.auto_apply = None
        self.pending_context = None
        self.search_mode = None
        self.search_started = None
        self.search_work = None
        self.rules_ready = False
        self.timeline = None  # Kept separately so the v1 current-board format remains readable.
        self.db = sqlite3.connect(self.path / "boards.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS current(id INTEGER PRIMARY KEY CHECK(id=1), body TEXT NOT NULL, sha TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, name TEXT NOT NULL, body TEXT NOT NULL, sha TEXT NOT NULL, updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS play_history(id TEXT PRIMARY KEY, body TEXT NOT NULL, sha TEXT NOT NULL);
        """)
        row = self.db.execute("SELECT body,sha FROM current WHERE id=1").fetchone()
        restored_board = row is not None
        if row:
            value = json.loads(row['body'])
            need(config.digest(value) == row['sha'], "Saved board checksum differs", "corrupt_saved_board")
            need(set(value) == {'position', 'revision', 'last_job', 'research_training_eligible'}, "Saved board schema differs")
            need(value['research_training_eligible'] is False, "User board eligibility differs")
            self.position = config.position(value['position'])
            self.revision = config.integer(value['revision'], 'revision', 0, 2**63-1)
            self.last_job = value['last_job']
        else:
            self.position = config.position()
            self.revision = 0
            self.last_job = None
            self._persist()

        try:
            product_rules.migrate(self.db)
            self.rules_ready = True
        except Exception:
            self.close()
            raise
        if self.last_job:
            self._load_report(self.last_job)
            self.message = "Saved board restored. Resume its analysis when ready."
        row = self.db.execute("SELECT body,sha FROM play_history WHERE id='current'").fetchone()
        if row:
            self.timeline = self._read_timeline(row)
            need(self.timeline['board_key'] == self.board_key and self.timeline['revision'] == self.revision,
                 "Saved move history does not match the board", "corrupt_saved_history")
        else:
            self.timeline = self._new_timeline("History begins at the board present when move recording was enabled. Earlier moves were not recorded."
                                               if restored_board else "New game. Every chosen pit and forced pass is recorded below.")
            self._persist()

    def _new_timeline(self, note):
        return dict(start=self.position, entries=[], note=note, turn_message="",
                    board_key=self.board_key, revision=self.revision)

    @staticmethod
    def _read_timeline(row):
        body = json.loads(row['body'])
        need(config.digest(body) == row['sha'], "Saved move history checksum differs", "corrupt_saved_history")
        need(isinstance(body.get('entries'), list), "Invalid saved move history")
        return body

    @property
    def board_key(self):
        return config.digest(self.position)

    def _persist(self):
        body = dict(position=self.position, revision=self.revision, last_job=self.last_job, research_training_eligible=False)
        with self.db:
            if self.rules_ready:
                product_rules.require(self.db, 'current')
            self.db.execute("INSERT OR REPLACE INTO current VALUES(1,?,?)", (config.canonical(body), config.digest(body)))
            if self.timeline is not None:
                self.db.execute("INSERT OR REPLACE INTO play_history VALUES('current',?,?)",
                                (config.canonical(self.timeline), config.digest(self.timeline)))

    def _load_report(self, job_id):
        saved = self.store.load(job_id)
        product_rules.engine_identity(saved['payload']['identity'])
        self.latest = dict(job_id=job_id, board_key=config.digest(saved['payload']['position']),
                           report=saved['report'], explanations=explain(saved['report']),
                           stop_reason=saved['stop_reason'], compatible=saved['compatible'],
                           identity=saved['payload']['identity'],
                           limits=saved['runs'][-1]['settings'] if saved['runs'] else saved['payload']['initial_limits'])
        return saved

    def _idle(self):
        need(self.coord.active is None, "Analysis is running. Stop it and wait before changing the board.", "busy")

    def _revision(self, req):
        need(type(req.get('revision')) is int and req['revision'] == self.revision,
             "The board changed. Refresh before applying this action.", "stale_board")

    def _change(self, p, message, entry=None, history=None):
        self.position = config.position(p)
        self.revision += 1
        self.last_job = None
        self.auto_apply = None
        self.message = message
        if history is not None:
            self.timeline = dict(history, board_key=self.board_key, revision=self.revision)
        elif entry is None:
            self.timeline = self._new_timeline(message + " Move recording starts here.")
        else:
            # Board and history are committed in the same FULL-synchronous SQLite transaction.
            self.timeline = dict(self.timeline, entries=[*self.timeline['entries'], entry],
                                 board_key=self.board_key, revision=self.revision, turn_message=entry['explanation'])
        self._persist()

    def _interrupt(self):
        # Disable autoplay before draining a completion racing with a user move.
        self.auto_apply = None
        self.coord.cancel()
        self.coord.join()
        self._drain()
        self._idle()

    def _apply(self, action):
        normal = self.native.normalize(flatten(self.position), actor=self.position['controller'], total=98)
        need(normal.get('ok'), "Position could not be normalized")
        if action == 'pass':
            need(bool(normal['passes']), "Pass is legal only when the current row is empty.", "illegal_pass")
            actor = self.position['controller']
            explanation = f"Player {'AB'[actor]}'s row is empty; player {'AB'[normal['position']['controller']]} moves next."
            entry = self._move_entry(actor, 'pass', normal['position'], explanation, normal['passes'], 'empty_row')
            self._change(normal['position'], explanation, entry)
            return
        action = config.integer(action, 'pit', 0, 6)
        need(not normal['passes'], "Pass before selecting the other player's pit.", "forced_pass")
        result = self.native.apply(flatten(self.position), actor=self.position['controller'], action=action, total=98, trace=True)
        need(result.get('ok'), result.get('error', {}).get('message', 'Illegal pit'), 'illegal_action')
        actor = self.position['controller']
        label = 'AB'[actor]
        if result['terminal']:
            explanation = "Both rows are empty. The round is complete."
        elif result['passes']:
            explanation = " ".join(f"Player {'AB'[p['from']]}'s row is empty and must pass to {'AB'[p['to']]}." for p in result['passes'])
            if result['position']['controller'] == actor:
                explanation += f" Player {label} plays again."
        elif result['reason'] == 'own_store':
            explanation = f"The last seed landed in {label}'s store. Player {label} gets another move."
        else:
            explanation = f"The turn ends. Player {'AB'[result['position']['controller']]} moves next."
        entry = self._move_entry(actor, action, result['position'], explanation, result['passes'], result['reason'])
        entry.update(relays=result['relays'], drops=result['drops'],
                     captured=sum(e.get('count', 0) for e in result['events'] if e['type'] == 'capture'))
        self._change(result['position'], f"Player {label} played pit {action+1}. " + explanation, entry)

    def _move_entry(self, actor, action, after, explanation, passes, reason):
        return dict(number=len(self.timeline['entries'])+1, actor=actor, action=action,
                    before=self.position, after=after, before_key=self.board_key, after_key=config.digest(after),
                    explanation=explanation, passes=passes, reason=reason,
                    store_delta=[after['stores'][i]-self.position['stores'][i] for i in (0,1)])

    def _best(self):
        need(self.latest is not None and self.latest['board_key'] == self.board_key, "No recommendation belongs to this board.", 'stale_analysis')
        need(self.latest['compatible'], "This saved analysis uses a different engine build. Analyze the board again.", 'incompatible_saved_data')
        report = self.latest['report']
        need(report is not None and report['choice_depth'] > 0 and report['best_moves'], "No completed move ranking is available.", 'no_ranking')
        need(not report['forced_passes'], "Apply the mandatory pass first.", 'forced_pass')
        self._apply(report['best_moves'][0])

    def _drain(self):
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind = event['event']
            if kind == 'job_started':
                self.search_started = time.monotonic()
                self.search_work = None
                self.last_job = event['job_id']
                self._persist()
                self.message = "Analyzing. Completed depths will appear below."
            elif kind in ('iteration', 'analysis_done'):
                job = event['job_id']
                if job == self.last_job:
                    self._load_report(job)
                    if kind == 'analysis_done':
                        self.search_work = event.get('work')
                    self.message = ("Analysis running." if kind == 'iteration' else
                                    "Analysis stopped: " + str(event.get('stop_reason', 'complete')).replace('_', ' ') + ".")
                if kind == 'analysis_done' and self.auto_apply:
                    wanted = self.auto_apply
                    self.auto_apply = None
                    if wanted == self.board_key and job == self.last_job and event.get('ok'):
                        try:
                            self._best()
                        except config.InputError as exc:
                            self.message = str(exc)
                if event.get('error'):
                    self.message = event['error'].get('message', 'Analysis error')

    def snapshot(self):
        with self.lock:
            product_rules.require(self.db, 'current')
            self._drain()
            normal = self.native.normalize(flatten(self.position), actor=self.position['controller'], total=98)
            need(normal.get('ok'), "Position could not be validated")
            with self.store.mutex:
                jobs = [dict(id=r['id'], status=r['status'], depth=r['latest_depth']) for r in
                        self.store.db.execute("SELECT id,status,latest_depth FROM jobs ORDER BY updated DESC LIMIT 20")]
            sessions = [dict(id=r['id'], name=r['name']) for r in self.db.execute("SELECT id,name FROM sessions ORDER BY updated DESC LIMIT 20")]
            latest = dict(self.latest, current=self.latest['board_key'] == self.board_key) if self.latest else None
            return dict(ok=True, version=VERSION, token=self.token, position=self.position,
                        revision=self.revision, board_key=self.board_key, active=self.coord.active is not None,
                        legal_actions=[] if normal['passes'] else normal['legal_actions'],
                        forced_pass=bool(normal['passes']), terminal=normal['terminal'],
                        analysis=latest, last_job=self.last_job, message=self.message,
                        jobs=jobs, sessions=sessions, profile=config.PROFILE, engine='H_v1',
                        identity=config.identity(self.native.build_id),
                        rule_contract=dict(product_rules.CONTRACT),
                        rules_source=product_rules.SOURCE,
                        history=self.timeline,
                        search=dict(mode=self.search_mode, elapsed_ms=int((time.monotonic()-self.search_started)*1000)
                                    if self.coord.active is not None and self.search_started is not None else None,
                                    work=self.search_work),
                        research_training_eligible=False)

    @staticmethod
    def search_limits(req, engine=False):
        value = req.get('limits', {})
        need(isinstance(value, dict), "Search limits must be an object")
        limits = config.limits(value)
        need(1 <= limits['max_depth'] <= 128, "Choose a depth from 1 to 128")
        need(20 <= limits['time_ms'] <= 10000, "Choose analysis time from 20 to 10000 ms")
        need(1 <= limits['node_budget'] <= 2000000, "Choose a node limit from 1 to 2000000")
        need(limits['memory_mb'] == 512 and limits['tt_mb'] == 16 and limits['use_pvs'] is True,
             "This product uses the fixed H_v1 configuration: 512 MiB, 16 MiB TT and PVS.")
        return limits

    def command(self, req):
        need(isinstance(req, dict), "Command must be a JSON object")
        need(set(req) <= {'op','revision','position','action','limits','job_id','session_id','name','rule_contract'}, "Unsupported command fields")
        if 'rule_contract' in req:
            product_rules.validate(req['rule_contract'])
        with self.lock:
            product_rules.require(self.db, 'current')
            op = req.get('op')
            if op == 'stop':
                self.auto_apply = None
            changing = op in ('import','opening','apply','apply_best','load_session','load_job')
            if changing:
                self._revision(req)
                self._interrupt()
            else:
                self._drain()
            if op == 'stop':
                self.auto_apply = None
                self.coord.cancel()
                self.message = "Stop requested. Keeping completed analysis."
            elif op in ('import','opening','apply','apply_best'):
                self._revision(req); self._idle()
                if op == 'import':
                    need(isinstance(req.get('position'), dict), "Import requires a position object")
                    p = config.position(req.get('position'))
                    need('position' in req, "Import requires a position")
                    self._change(p, "Position imported.")
                elif op == 'opening':
                    self._change(config.position(), "Opening position restored.")
                elif op == 'apply':
                    self._apply(req.get('action'))
                else:
                    self._best()
            elif op in ('analyze','engine_turn','resume','live'):
                self._revision(req); self._idle()
                limits = dict(LIVE_LIMITS) if op == 'live' else self.search_limits(req, op == 'engine_turn')
                if op == 'live':
                    need('limits' not in req, "Live analysis uses its fixed bounded limits")
                self.search_mode = op
                normal = self.native.normalize(flatten(self.position), actor=self.position['controller'], total=98)
                need(not normal['terminal'], "The game has ended.", 'terminal_position')
                need(not normal['passes'], "Apply the mandatory pass first.", 'forced_pass')
                # Resume the matching durable live root after a disconnect or restart.
                if op == 'live' and self.last_job and self.latest and self.latest['board_key'] == self.board_key and self.latest['compatible']:
                    self.auto_apply = None
                    self.coord.handle(dict(op='resume', job_id=self.last_job, limits=limits))
                elif op == 'resume':
                    job = req.get('job_id', self.last_job)
                    need(isinstance(job, str), "Select a saved analysis first")
                    saved = self.store.load(job, compatible=True)
                    need(config.digest(saved['payload']['position']) == self.board_key, "Saved analysis belongs to another board. Load it first.", 'stale_analysis')
                    self.last_job = job
                    self._persist()
                    self.coord.handle(dict(op='resume', job_id=job, limits=limits))
                else:
                    with self.store.mutex:
                        need(self.store.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] < 1000,
                             "The product directory has reached its 1000-analysis limit.", 'product_capacity')
                    self.auto_apply = self.board_key if op == 'engine_turn' else None
                    self.coord.handle(dict(op='analyze', position=self.position, limits=limits))
                self._drain()
            elif op == 'save':
                name = req.get('name', 'Saved board')
                need(isinstance(name, str) and 1 <= len(name.strip()) <= 80, "Use a board name of 1 to 80 characters")
                need(self.db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] < 100,
                     "The product directory has reached its 100-saved-board limit.", 'product_capacity')
                body = dict(position=self.position, research_training_eligible=False)
                with self.db:
                    session_id = uuid.uuid4().hex
                    self.db.execute("INSERT INTO sessions VALUES(?,?,?,?,?)", (session_id,name.strip(),config.canonical(body),config.digest(body),time.time()))
                    self.db.execute("INSERT INTO play_history VALUES(?,?,?)", (session_id, config.canonical(self.timeline), config.digest(self.timeline)))
                    product_rules.bind(self.db, session_id, 'new_save')
                self.message = "Board saved."
            elif op == 'load_session':
                self._revision(req); self._idle()
                name = req.get('session_id')
                need(isinstance(name,str) and len(name) == 32, "Select a saved board")
                row = self.db.execute("SELECT body,sha FROM sessions WHERE id=?", (name,)).fetchone()
                need(row is not None, "Saved board was not found")
                product_rules.require(self.db, name)
                body = json.loads(row['body'])
                need(config.digest(body) == row['sha'] and body['research_training_eligible'] is False, "Saved board checksum differs")
                history = self.db.execute("SELECT body,sha FROM play_history WHERE id=?", (name,)).fetchone()
                restored = None
                if history:
                    restored = self._read_timeline(history)
                    need(restored['board_key'] == config.digest(body['position']), "Saved history belongs to another board")
                self._change(body['position'], "Saved board loaded.", history=restored)
            elif op == 'load_job':
                self._revision(req); self._idle()
                job = req.get('job_id')
                need(isinstance(job, str) and len(job) == 32, "Select a saved analysis")
                saved = self._load_report(job)
                self._change(saved['payload']['position'], "Saved analysis loaded.")
                self.last_job = job
                self._persist()
            else:
                raise config.InputError('unknown_operation', 'Unknown product command')
            return self.snapshot()

    def finder(self):
        with self.lock:
            if self._finder is None:
                from .product_finder_v1 import Finder
                suffix = ('browser' if self.path == ROOT.resolve()/'data'/'product-v1'
                          else 'product-fixtures/' + str(self.path.relative_to(ROOT.resolve()/'data'/'product-v2-fixtures')))
                base = ROOT/('data' if suffix == 'browser' else 'build')/'dakon-opening-v1'
                self._finder = Finder(base/suffix)
            return self._finder

    def close(self):
        if self._finder is not None:
            self._finder.close()
        self.coord.cancel()
        self.coord.join()
        self.db.close()
        self.store.close()


def make_server(product, port=18767):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.0'

        def log_message(self, *_):
            pass

        def send(self, status, body, content_type='application/json; charset=utf-8', attachment=False):
            raw = body if isinstance(body,bytes) else config.canonical(body).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
            if attachment:
                self.send_header('Content-Disposition', 'attachment; filename="congklak-analysis.json"')
            self.end_headers()
            self.wfile.write(raw)

        def admit(self, post=False):
            host = self.headers.get('Host', '')
            need(host in {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}, 'Unrecognized local host', 'forbidden')
            origin = self.headers.get('Origin')
            need(origin is None or origin == 'http://' + host, 'Cross-origin requests are refused', 'forbidden')
            if post:
                need(self.headers.get('X-Congklak-Token') == product.token, 'Refresh this page before sending a command', 'forbidden')

        def do_GET(self):
            try:
                self.admit()
                route = urlsplit(self.path)
                need(not route.query, 'Unexpected query parameters')
                if route.path in ('/api/state', '/api/export'):
                    value = product.snapshot()
                    if route.path == '/api/export':
                        value.pop('token', None)
                    self.send(200, value, attachment=route.path == '/api/export')
                elif route.path == '/api/finder':
                    self.send(200, dict(product.finder().snapshot(), token=product.token))
                elif route.path.startswith('/api/finder/') and route.path.endswith('/certificate'):
                    parts = route.path.split('/')
                    need(len(parts) == 5, 'Unknown finder route')
                    self.send(200, product.finder().certificate(parts[3]))
                elif route.path in ('/', '/app.js', '/finder.js', '/style.css'):
                    name = {'/':'index.html','/app.js':'app.js','/finder.js':'finder.js','/style.css':'style.css'}[route.path]
                    typ = 'text/html' if name == 'index.html' else 'text/css' if name.endswith('.css') else 'application/javascript'
                    self.send(200, (ASSETS/name).read_bytes(), typ + '; charset=utf-8')
                else:
                    self.send(404, dict(ok=False,error=dict(code='not_found',message='Page not found')))
            except (config.InputError, ValueError, OSError, RuntimeError, TypeError, KeyError) as exc:
                self.send(403 if getattr(exc,'code','') == 'forbidden' else 400, dict(ok=False,error=dict(code=getattr(exc,'code','request_error'),message=str(exc))))

        def do_POST(self):
            try:
                self.admit(True)
                need(self.path in ('/api/command', '/api/finder/command'), 'Unknown command route')
                need(self.headers.get('Transfer-Encoding') is None, 'Chunked requests are unsupported')
                need(self.headers.get_content_type() == 'application/json', 'Send application/json')
                length = int(self.headers.get('Content-Length', '-1'))
                need(0 <= length <= MAX_BODY, 'Command exceeds 64 KiB')
                self.connection.settimeout(3)
                body = self.rfile.read(length)
                need(len(body) == length, 'Incomplete JSON command')
                req = json.loads(body,parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite number')))
                self.send(200, product.command(req) if self.path == '/api/command' else
                          dict(product.finder().command(req), token=product.token))
            except (config.InputError, ValueError, TypeError, KeyError, OSError, RuntimeError, RecursionError, sqlite3.Error) as exc:
                self.send(403 if getattr(exc,'code','') == 'forbidden' else 400, dict(ok=False,error=dict(code=getattr(exc,'code','request_error'),message=str(exc))))

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = False

    return Server(('127.0.0.1', port), Handler)


def main():
    parser = argparse.ArgumentParser(description='Local Congklak Betawi analyze and play app (H_v1 CPU)')
    parser.add_argument('--port',type=int,default=18767)
    parser.add_argument('--data-dir',default=str(ROOT/'data'/'product-v1'))
    parser.add_argument('--open-browser', action='store_true')
    args = parser.parse_args()
    product = None
    server = None
    try:
        need(0 <= args.port <= 65535, 'Invalid local port')
        product = Product(args.data_dir)
        server = make_server(product,args.port)
        def stop(*_):
            product.coord.cancel()
            threading.Thread(target=server.shutdown,daemon=True).start()
        signal.signal(signal.SIGINT,stop)
        if hasattr(signal,'SIGTERM'):
            signal.signal(signal.SIGTERM,stop)
        print(config.canonical(dict(event='product_ready',url=f'http://127.0.0.1:{server.server_port}/',version=VERSION,engine='H_v1')),flush=True)
        if args.open_browser:
            import webbrowser
            webbrowser.open(f'http://127.0.0.1:{server.server_port}/')
        server.serve_forever(poll_interval=.2)
        return 0
    except Exception as exc:
        print(config.canonical(dict(event='product_error',error=str(exc))),flush=True)
        return 2
    finally:
        if server:
            server.server_close()
        if product:
            product.close()


if __name__ == '__main__':
    raise SystemExit(main())

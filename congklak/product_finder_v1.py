"""Browser adapter for the version-pinned finder. Never opens a game database."""
from __future__ import annotations

import re
import threading
import time
import uuid

from .dakon_opening import Run, read_recipe, recipe, run_path, status
from .dakon_opening_verify import canonical, read_certificate, verify

RUN_ID = re.compile(r'[0-9a-f]{32}\Z')


class Finder:
    """One cooperative worker, durable run directories, read-only replay exports.

    On process restart all unfinished work is paused. Polling uses the last
    committed projection; it never reconstructs the DFS or starts a worker.
    """

    def __init__(self, root, *, search_options=None, hook=None):
        self.root = run_path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.records = {}
        self.thread = None
        self.active = None
        self.cancel = threading.Event()
        self.closed = False
        self.started = None
        self.options = dict(max_actions=0, max_seconds=600, checkpoint_actions=64,
                            progress_seconds=.5)
        self.options.update(search_options or {})
        self.hook = hook
        for path in sorted(self.root.iterdir()):
            if path.is_dir() and RUN_ID.fullmatch(path.name):
                self.records[path.name] = self._restore(path)

    def _path(self, run_id):
        if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
            raise ValueError('Choose a saved finder run')
        path = run_path(self.root/run_id)
        if path.parent != self.root:
            raise ValueError('Finder run is outside its library')
        return path

    @staticmethod
    def _clean(value):
        return {key: val for key, val in value.items() if key != 'certificate'}

    def _restore(self, path):
        record = dict(run_id=path.name, created=path.stat().st_mtime, error=None,
                      resumable=True)
        try:
            record['created'] = (path/'identity.json').stat().st_mtime
            if not list(path.glob('checkpoint-[01].json')) and not (path/'certificate.json').exists():
                # A crash after publishing identity but before the first frontier
                # has a single reconstructible state: the untouched opening.
                read_recipe(path)
                with Run(path) as initial:
                    record.update(self._clean(initial.summary()))
            record.update(self._clean(status(path)))
        except (ValueError, OSError, TypeError, KeyError, RuntimeError) as error:
            record.update(state='unavailable', outcome='unknown', error=str(error),
                          resumable=False)
        return record

    def snapshot(self):
        with self.lock:
            runs = []
            for record in sorted(self.records.values(), key=lambda r: r['created'], reverse=True):
                item = dict(record)
                item['active'] = self.active == item['run_id']
                item['pausing'] = item['active'] and self.cancel.is_set()
                if item['active']:
                    item['session_seconds'] = round(time.monotonic()-self.started, 1)
                runs.append(item)
            return dict(ok=True, runs=runs, active_run=self.active, max_n=64,
                        run_limit=100, session_limit_seconds=600,
                        research_training_eligible=False)

    def command(self, req):
        if not isinstance(req, dict):
            raise ValueError('Expected a finder command object')
        op = req.get('op')
        keys = {'op', 'n', 'goal', 'starter'} if op == 'start' else {'op', 'run_id'}
        if set(req) != keys or op not in ('start', 'pause', 'resume'):
            raise ValueError('Unknown finder command or fields')
        with self.lock:
            if self.closed:
                raise ValueError('Finder is shutting down')
            if op == 'pause':
                run_id = req['run_id']
                self._path(run_id)
                if run_id not in self.records:
                    raise ValueError('Saved finder run was not found')
                if self.active == run_id:
                    self.cancel.set()
                return self.snapshot()
            if self.active is not None:
                raise ValueError('Pause the current finder before starting or resuming another run')
            if op == 'start':
                prepared = recipe(req['n'], req['goal'], req['starter'], 10000)
                if len(self.records) >= 100:
                    raise ValueError('Finder library has reached its 100-run limit')
                run_id = uuid.uuid4().hex
                path = self._path(run_id)
                with Run(path, create={key: prepared[key] for key in ('n','goal','starter','cache_entries')}) as run:
                    record = dict(self._clean(run.summary()), run_id=run_id,
                                  created=(path/'identity.json').stat().st_mtime,
                                  error=None, resumable=True)
                self.records[run_id] = record
            else:
                run_id = req['run_id']
                self._path(run_id)
                if run_id not in self.records:
                    raise ValueError('Saved finder run was not found')
                if not self.records[run_id]['resumable']:
                    raise ValueError('This run needs repair: '+self.records[run_id]['error'])
                if self.records[run_id]['state'] in ('found', 'exhausted'):
                    return self.snapshot()
            self.active = run_id
            self.started = time.monotonic()
            self.cancel = threading.Event()
            self.records[run_id]['error'] = None
            self.thread = threading.Thread(target=self._work, args=(run_id,),
                                           name='dakon-opening-finder', daemon=True)
            self.thread.start()
            return self.snapshot()

    def _progress(self, run_id, summary):
        with self.lock:
            self.records[run_id].update(self._clean(summary))

    def _work(self, run_id):
        try:
            with Run(self._path(run_id), hook=self.hook) as run:
                # No winning claim until this invocation verifies the certificate.
                self._progress(run_id, dict(self._clean(run.summary()), outcome='unknown'))
                result = run.search(**self.options, cancel=self.cancel.is_set,
                                    emit=lambda summary: self._progress(run_id, summary))
                self._progress(run_id, result)
        except Exception as error:
            with self.lock:
                self.records[run_id].update(error=str(error), outcome='unknown')
        finally:
            with self.lock:
                self.active = None
                self.started = None

    def certificate(self, run_id):
        with self.lock:
            path = self._path(run_id)
            record = self.records.get(run_id)
            if not record or record.get('outcome') != 'guaranteed_win' or self.active == run_id:
                raise ValueError('This run has no completed, verified winning sequence')
        # Validate again at export, including recipe and durable witness binding.
        try:
            current = status(path)
            cert = read_certificate(path/'certificate.json')
            checked = verify(cert)
            if (current['state'] != 'found' or
                canonical(cert['body']['recipe']) != canonical(read_recipe(path)) or
                [action+1 for action in cert['body']['actions']] != current['line_right_store']):
                raise ValueError('Certificate differs from the saved finder run')
        except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
            with self.lock:
                self.records[run_id].update(outcome='unknown', error=str(error), resumable=False)
            raise
        return dict(ok=True, run_id=run_id, certificate=cert, verification=checked,
                    research_training_eligible=False)

    def close(self):
        with self.lock:
            self.closed = True
            self.cancel.set()
            thread = self.thread
        if thread:
            thread.join()

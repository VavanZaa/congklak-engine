"""Observations inside the full decision clock; no GC or scheduler policy change."""
import gc,time

class GcObserver:
    def __init__(self):
        self.active=None;self.events=[];self._starts={};self.closed=False
        self._callback=self._observe;gc.callbacks.append(self._callback)
    def _observe(self,phase,info):
        if self.active is None:return
        generation=info['generation']
        if phase=='start':self._starts[generation]=time.perf_counter_ns()
        else:
            self.events.append(dict(decision_id=self.active,generation=generation,
              elapsed_ns=time.perf_counter_ns()-self._starts.pop(generation,time.perf_counter_ns()),
              collected=info['collected'],uncollectable=info['uncollectable']))
    def begin(self,decision_id,started_ns=None):
        if self.closed or self.active is not None:raise ValueError('Timing observer is closed or already active')
        self.active=decision_id
        return CriticalClock(self,decision_id,started_ns)
    def close(self):
        if self.active is not None:raise ValueError('A full decision clock remains open')
        gc.callbacks.remove(self._callback);self.closed=True
        return dict(schema='prospective-benchmark-v2-gc-observer',gc_policy_changed=False,events=len(self.events))

class CriticalClock:
    def __init__(self,observer,decision_id,started_ns=None):
        self.started_ns=time.perf_counter_ns() if started_ns is None else started_ns
        self.started_thread_ns=time.thread_time_ns();self.started_process_ns=time.process_time_ns()
        self.observer=observer;self.decision_id=decision_id;self.event_offset=len(observer.events)
        self.gc_before=tuple(x['collections'] for x in gc.get_stats());self.points={};self.closed=False
    @property
    def started(self):return self.started_ns/1e9
    def mark(self,name):
        if self.closed or name in self.points:raise ValueError('Timing phase is closed or duplicated')
        self.points[name]=(time.perf_counter_ns()-self.started_ns)/1e6
    def finish(self):
        if self.closed:raise ValueError('Full decision timer closed twice')
        gc_after=tuple(x['collections'] for x in gc.get_stats())
        result=dict(schema='prospective-benchmark-v2-full-decision-clock',phase_end_ms=dict(self.points),
          gc_collections_before=self.gc_before,gc_collections_after=gc_after,
          gc_events=list(self.observer.events[self.event_offset:]),gc_enabled=gc.isenabled(),gc_policy_changed=False,
          thread_cpu_ms=(time.thread_time_ns()-self.started_thread_ns)/1e6,
          process_cpu_ms=(time.process_time_ns()-self.started_process_ns)/1e6,
          boundary='both-seat reset, complete native search/trace/hash/guard/witness, literal legality/transition equality and CPU verification',
          archive_encoding_outside_clock=True)
        self.observer.active=None;self.closed=True
        result['critical_ms']=(time.perf_counter_ns()-self.started_ns)/1e6
        return result

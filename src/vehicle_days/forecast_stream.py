"""Bounded multicore forecasting for the immutable reference replay.

Each job observes only its own logical snapshot. The presentation buffer may
prepare later snapshots of the recorded reference, but never exposes them before
that time. A live, unrecorded world would submit observed snapshots without this
reference-replay lookahead. Neither path owns the viewer's clock.
"""
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import math
import multiprocessing
import os
import threading
import time

_WORKER_CONTEXT = None
_WORKER_REASONER = None


def _init_worker():
    global _WORKER_CONTEXT, _WORKER_REASONER
    from .encounters import EncounterService
    from .reasoning import RollingContext, TrafficReasoner
    _WORKER_CONTEXT = RollingContext()
    _WORKER_REASONER = TrafficReasoner(EncounterService(context=_WORKER_CONTEXT))


def _assess_job(owner, at):
    began=time.perf_counter()
    result = _WORKER_REASONER.assess(owner, at)
    index = result['snapshot']
    forecasts = [(p['id'], _WORKER_CONTEXT.forecast(index,p)) for p in result['candidates']]
    return {'assessment':result, 'forecasts':forecasts,'worker_seconds':time.perf_counter()-began,'worker_pid':os.getpid()}


class ForecastStream:
    def __init__(self, encounters, reasoner, workers=None, executor=None):
        self.encounters, self.reasoner = encounters, reasoner
        self.workers = workers or min(16,max(2,(os.cpu_count() or 4)-2))
        self.executor = executor or ProcessPoolExecutor(max_workers=self.workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=_init_worker)
        self.lock = threading.RLock()
        self.jobs, self.ready = {}, OrderedDict()
        self.active_owner = None
        self.anchor = None
        self.closed = False
        self.cost = 1.
        self.completed, self.discarded, self.failed = 0, 0, 0
        self.compute_seconds = 0.
        self.worker_pids=set()
        self.peak_pending = 0

    def _key(self, owner, at):
        return (owner,self.encounters.context.frame_index(at))

    def _complete(self, key, future):
        try:
            payload = future.result()
            result = payload['assessment']
            if result['source'] != self.encounters.context.source:
                raise ValueError('The reference recording changed')
            with self.lock:
                if self.closed or key[0]!=self.active_owner:
                    self.jobs.pop(key,None);self.discarded+=1;return
            # Adopt completed immutable calculations, never a viewer clock/state.
            with self.encounters.lock:
                context = self.encounters.context
                for candidate_id,forecast in payload['forecasts']:
                    context.forecasts[(result['snapshot'],candidate_id)] = forecast
                while len(context.forecasts)>32:
                    context.forecasts.popitem(last=False)
                self.reasoner.assessed[(result['snapshot'],key[0])] = result
                while len(self.reasoner.assessed)>96:
                    self.reasoner.assessed.popitem(last=False)
            with self.lock:
                self.jobs.pop(key,None)
                if self.closed:
                    return
                self.ready[key] = {'status':'ready','result':result}
                self.completed += 1
                duration = max(.01,payload.get('worker_seconds',result.get('seconds',self.cost)))
                if payload.get('worker_pid'):self.worker_pids.add(payload['worker_pid'])
                self.cost = .8*self.cost+.2*duration
                self.compute_seconds += duration
                while len(self.ready)>128:
                    self.ready.popitem(last=False)
        except Exception as exc:
            with self.lock:
                self.jobs.pop(key,None)
                if future.cancelled():
                    self.discarded += 1
                elif not self.closed:
                    self.failed += 1
                    self.ready[key] = {'status':'failed','error':str(exc)}

    def _submit(self, owner, at):
        key = self._key(owner,at)
        if key in self.jobs or key in self.ready or self.closed:
            return
        try:
            future = self.executor.submit(_assess_job,owner,self.encounters.context.times[key[1]])
        except Exception as exc:
            self.ready[key]={'status':'failed','error':str(exc)};self.failed+=1;return
        self.jobs[key] = future
        self.peak_pending = max(self.peak_pending,len(self.jobs))
        future.add_done_callback(lambda f,k=key:self._complete(k,f))

    def poll(self, owner, at, rate=4., exact=False, prefer_current=False):
        context = self.encounters.context
        if owner is not None and owner not in context.indices:
            raise ValueError('Unknown car')
        current_key = self._key(owner,at)
        if not math.isfinite(rate) or not 0 < rate <= 16:
            raise ValueError('Invalid playback speed')
        with self.lock:
            if self.closed:
                raise ValueError('Forecast workers have stopped')
            seek = self.anchor is not None and (at < self.anchor-.5 or at-self.anchor>8.)
            changed = owner != self.active_owner
            self.active_owner, self.anchor = owner,at
            # Superseded queued work never creates an accumulating backlog.
            for key,f in list(self.jobs.items()):
                t=context.times[key[1]]
                if key[0]!=owner or t < at-2. or seek:
                    if f.cancel():self.jobs.pop(key,None)
            if exact:
                for key,f in list(self.jobs.items()):
                    if key!=current_key and len(self.jobs)>=self.workers+2 and f.cancel():self.jobs.pop(key,None)
                if len(self.jobs)<self.workers+2:self._submit(owner,at)
            else:
                cadence = max(.5,math.ceil(rate*self.cost/self.workers*1.25*2)/2)
                start = math.floor(at/cadence)*cadence
                horizon = min(context.times[-1],at+max(8.,rate*max(3.,self.cost*2+.5)))
                # Current snapshot has first claim on capacity; then buffer ahead.
                if prefer_current or not any(k[0]==owner and 0 <= at-context.times[k[1]] <= cadence for k in self.ready):
                    if len(self.jobs)<self.workers+2:self._submit(owner,at)
                t=start
                while t<=horizon and len(self.jobs)<self.workers+2:
                    if t>=at-.001:self._submit(owner,t)
                    t+=cadence
            entry = self.ready.get(current_key) if exact else None
            if not exact:
                past=[(context.times[k[1]],v) for k,v in self.ready.items()
                    if k[0]==owner and context.times[k[1]] <= at and v['status']=='ready']
                if past:entry=max(past,key=lambda p:p[0])[1]
            stats={'workers':self.workers,'pending':len(self.jobs),'ready':len(self.ready),
                'completed':self.completed,'discarded':self.discarded,'failed':self.failed,
                'mean_job_seconds':round(self.cost,4),'peak_pending':self.peak_pending,'worker_processes_used':len(self.worker_pids)}
            if not entry and current_key in self.ready:entry=self.ready[current_key]
            if not entry:
                return {'status':'pending','requested_time':context.times[current_key[1]],'stats':stats}
            value={**entry,'stats':stats}
            if entry['status']=='ready':
                value['age_seconds']=max(0.,at-entry['result']['time'])
                # Cheap current geometry screen removes already-passed proposals.
                if owner:
                    with self.encounters.lock:
                        eligible={p['id'] for p in context.candidates(owner,at)['candidates']}
                    value['eligible_now']=[p['id'] for p in entry['result']['candidates'] if p['id'] in eligible]
            return value

    def close(self):
        with self.lock:
            self.closed=True
            for future in list(self.jobs.values()):future.cancel()
        self.executor.shutdown(wait=False,cancel_futures=True)

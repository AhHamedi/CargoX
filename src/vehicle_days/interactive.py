"""An interactive demonstration using the existing physical driving engine.

Reality uses Simulation and Driver; speculative branches retain the explicitly
approximate public lane projection. Only frame zero is imported from the old
recording. Later poses, arrivals and agreements are generated here. Forecast
jobs receive one observed state, never later frames or recorded arrivals.
"""
from bisect import bisect_right
from collections import OrderedDict
from copy import copy, deepcopy
from dataclasses import asdict
import math
import os
from pathlib import Path
import secrets
import threading
import time

from .encounters import (FORECAST_STEP, HORIZON, first_crossing, physical_frame,
                         shared_hold_obstruction, write_record)
from .layout_traffic import Driver, SPEED
from .reasoning import RollingContext, TrafficReasoner
from .encounters import EncounterService

STEP = FORECAST_STEP
ESTIMATE_MAX_AGE = 10.


class InteractiveContext(RollingContext):
    def instruction(self, index, candidate, public=False):
        instruction = super().instruction(index, candidate, public)
        # A forecast may remain conditional, but the accepted live promise has
        # no twenty-second escape clause. Model that same rule for its horizon.
        instruction.expires = 1e9
        return instruction


def snapshot_context(payload):
    """Adapt one evolving state to the existing isolated public-intent reasoner."""
    context = InteractiveContext.__new__(InteractiveContext)
    context.geometry = payload['geometry']
    context.routes, context.meta, context.zones = payload['routes'], payload['meta'], payload['zones']
    context.owners = payload['owners']
    context.indices = {o:i for i,o in enumerate(context.owners)}
    context.source = payload['source']
    context.runtime = 'interactive-lane-v1'
    context.record = {'frames':[payload['frame']]}
    context.times = [payload['frame']['t']]
    context.progress = [payload['progress']]
    context._public, context._baseline_forecasts = {}, {}
    context.forecasts = OrderedDict()
    context.commitments = deepcopy(payload.get('commitments', []))
    context._template, context.baseline_audit = None, None
    return context


def effect_estimate(baseline, alternative, owner, target):
    ordinary = first_crossing(baseline['progress'], owner, target) if target is not None else None
    changed = first_crossing(alternative['progress'], owner, target) if target is not None else None
    if ordinary is not None and changed is not None:
        return {'seconds_saved':round(ordinary-changed, 2), 'confidence':'conditional',
                'method':'Paired crossing-time forecast', 'scope':'at this crossing'}
    # A finite, conservative approximation when a crossing lies beyond the
    # horizon. Compare positions at the same time rather than invent arrivals.
    end = min(baseline['progress'][-1][0], alternative['progress'][-1][0])
    def at(samples):
        i = max(0, bisect_right([p[0] for p in samples], end)-1)
        return samples[i][1][owner]
    difference = at(alternative['progress'])-at(baseline['progress'])
    return {'seconds_saved':round(difference/SPEED, 2), 'confidence':'rough',
            'method':'End-of-horizon progress difference divided by ordinary road speed',
            'scope':'time equivalent over the forecast horizon', 'horizon_seconds':round(end-baseline['progress'][0][0], 2)}


def assess_snapshot(payload, owner):
    """Pure process-pool job. No shared session or viewer state is modified."""
    began = time.perf_counter()
    context = snapshot_context(payload)
    reasoner = TrafficReasoner(EncounterService(context=context))
    assessment = reasoner.assess(owner, context.times[0])
    candidates, futures = [], {}
    world = context.public_world(0)
    baseline = context._baseline_forecasts[0]
    for candidate in assessment['candidates']:
        # Internal cache is immutable for this pure job. Public forecast()
        # remains detached; avoid copying both full eighty-car worlds twice.
        forecast = context.forecasts.get((0,candidate['id']))
        if forecast is None:
            forecast = context.forecast(0,candidate)
        branch = forecast['_branches']['concession']
        # The progress trace is already retained in the shared public projection;
        # recover alternative route progress from its own public frame poses.
        actors = list(dict.fromkeys([owner,candidate['seller'],candidate['buyer'],
                                    *candidate.get('beneficiaries',[])]))
        progress, previous = [], {o:5. for o in actors}
        for frame in branch['frames']:
            for o in actors:
                row=frame['cars'][context.indices[o]]
                previous[o] = world['routes'][o].project(row[0], row[1], previous[o])[0]
            progress.append((frame['t'], dict(previous)))
        alternative = {'progress':progress}
        zone = next(z for z in world['zones'] if z['id']==candidate['zone_id'])
        effects = {o:effect_estimate(baseline, alternative, o,
            zone['ranges'][o][1] if o in zone['ranges'] else None) for o in actors}
        for o in actors:
            if o not in zone['ranges']:
                effects[o]['scope']='queue-release time equivalent over the forecast horizon'
        value = {k:deepcopy(candidate[k]) for k in ('id','seller','buyer','point','kind','zone_id')}
        value.update(action_type=candidate.get('action_type','priority_exchange'),
            changes_order=candidate.get('changes_order',True),
            donor_motion_changes=candidate.get('donor_motion_changes',True),initiator=candidate.get('initiator',owner),
            beneficiaries=list(candidate.get('beneficiaries',[candidate['buyer']])),
            participants=list(candidate.get('participants',[candidate['seller'],candidate['buyer']])),
            causal_chain=list(candidate.get('causal_chain',[candidate['buyer'],candidate['seller']])))
        value.update(title=('Yield to '+candidate['buyer'] if candidate['seller']==owner else 'Ask '+candidate['seller']+' to yield'),
                     observation_time=context.times[0], effects=effects,
                     delivery_status=candidate.get('delivery_status','predicted'),
                     remaining_constraints=candidate['remaining_constraints'])
        candidates.append(value)
        futures[candidate['id']] = {'status':'ready', 'time':context.times[0], 'owners':actors,
            'frames':[{'t':f['t'], 'cars':[f['cars'][context.indices[o]] for o in actors]} for f in branch['frames']],
            'candidate':value, 'effects':effects, 'horizon_seconds':HORIZON,
            'uncertainty':'Conditional approximation. Later decisions and changing traffic can interrupt this future.'}
    return {'time':context.times[0], 'candidates':candidates, 'futures':futures,
        'normal_order':assessment['normal_order'], 'rejected':assessment['rejected'],
        'request_intents':assessment.get('requests',[]),
        'cars_assessed':len(context.owners), 'method':assessment['method'],
        'seconds':round(time.perf_counter()-began, 4), 'pid':os.getpid()}


class InteractiveDemo:
    def __init__(self, context, executor, workers=16, directory=None):
        self.context, self.executor, self.workers = context, executor, workers
        self.directory = Path(directory) if directory is not None else None
        self.lock = threading.RLock()
        self.jobs, self.ready = {}, OrderedDict()
        self.completed = self.failed = self.discarded = self.peak_pending = 0
        self.worker_pids = set()
        self.forecast_seconds = []
        self.closed = False
        self.session = secrets.token_hex(6)
        self.revision = 0
        self.selected = None
        self._new()

    def _new(self):
        self._save('reset')
        self.revision += 1
        for future in list(self.jobs.values()):
            future.cancel()
        self.jobs.clear(); self.ready.clear()
        c = self.context
        self.world, self.driver = c.restored_world(0)
        self.world.horizon = 600.
        self.frames, self.progress, self.contracts, self.events, self.native_states = [], [], [], [], []
        self.instructions = []
        self.requests = {}
        self.request_serial = 0
        self.last_request_scan = -1.
        self.arrivals, self.ended, self.failure = {}, False, None
        self.latest_observed = 0.
        self.selected = None
        self._capture()

    def _save(self, reason):
        if self.directory is None or not getattr(self, 'frames', None):
            return None
        meaningful = self.selected is not None or self.latest_observed>0 or self.failure or any(
            e['kind'] in ('accepted','execution_fault','physical_fault') for e in self.events)
        if not meaningful:
            return None
        value = {'format':'interactive-driving-v2', 'source':self.source,
            'revision':self.revision, 'reference_source':self.context.source,
            'model':'Kinematic bicycle physics', 'forecast_model':'Approximate lane projection',
            'model_calls':0, 'owners':list(self.context.owners), 'routes':self.context.meta,
            'frames':self.frames, 'events':self.events, 'arrivals':self.arrivals,
            'failure':self.failure, 'ended':self.ended, 'selected':self.selected,
            'observed_until':self.latest_observed, 'buffered_until':self.frames[-1]['t'],
            'observation_scope':'Frames and events after observed_until are prepared continuation, not yet observed driving.',
            'save_reason':reason, 'agreements':[self._contract_dict(c) for c in self.instructions],
            'requests':deepcopy(list(self.requests.values()))}
        path = self.directory/f'{self.session}-{self.revision:04d}.json.gz'
        write_record(path, value)
        return path

    @property
    def source(self):
        return f'interactive:{self.session}:{self.revision}'

    def _check_revision(self, revision):
        if type(revision) is not int or revision != self.revision:
            raise ValueError('This decision belongs to an older world. Use the current options.')

    @property
    def instruction(self):
        """The first live promise, for the retained single-passage inspector."""
        return next((c for c in self.instructions if c.status=='pending'),
                    self.instructions[-1] if self.instructions else None)

    def _contract_dict(self, contract):
        zone = next(z for z in self.context.zones if z['id']==contract.zone_id)
        return {**asdict(contract), 'id':getattr(contract,'id',
            f'{contract.zone_id}:{contract.seller}:{contract.buyer}@{contract.start:.4f}'),
            'point':[zone['x'],zone['y']],
            'participants':list(getattr(contract,'participants',[contract.seller,contract.buyer])),
            'action_type':getattr(contract,'action_type','priority_exchange')}

    def _active(self, index):
        return [self._contract_dict(c) for c in self.contracts[index] if c.status=='pending']

    def _refresh_controls(self):
        before = {id(c):c.status for c in self.instructions}
        self.actions, self.controls = self.driver.actions(self.world, self.instructions)
        for contract in self.instructions:
            obstruction=None
            if contract.status=='pending':
                a,b=contract.seller,contract.buyer
                car=self.world.cars[a]
                obstruction=shared_hold_obstruction(self.context.routes[a],self.context.routes[b],
                    self.driver.progress[a],self.driver.progress[b],contract.seller_entry,contract.buyer_exit,
                    (car['x'],car['y'],car['heading_rad']))
                if obstruction:
                    contract.status='unfulfillable_passage'
            if before[id(contract)]=='pending' and contract.status!='pending':
                status = contract.status
                self.events.append({'time':self.world.time,
                    'kind':'delivered' if status=='delivered' else 'execution_fault',
                    **self._contract_dict(contract),
                    **({'detail':obstruction} if obstruction else {})})
                if status != 'delivered':
                    self.ended=True
                    self.failure=(f'{contract.seller} cannot yield from ahead of {contract.buyer} on their shared lane.'
                                  if obstruction else status)
        self.arrivals = {o:c['arrival'] for o,c in self.world.cars.items() if c.get('arrival') is not None}
        if self.world.termination:
            if not self.ended and self.world.termination['kind']!='completed':
                self.events.append({'time':self.world.time,'kind':'physical_fault', 'detail':deepcopy(self.world.termination)})
            self.ended = True
            if self.world.termination['kind']!='completed':
                self.failure = self.world.termination['kind']

    def _capture(self, refresh=True):
        if refresh:
            self._refresh_controls()
        self.frames.append(physical_frame(self.world, self.controls, self.context.owners))
        self.progress.append(dict(self.driver.progress))
        self.contracts.append(deepcopy(self.instructions))
        self.native_states.append(deepcopy(self.world.cars))
        if self.ended and self.failure:
            self._save('fault')

    def _advance(self, to, yield_frames=False):
        revision = self.revision
        while self.world.time+1e-6 < to and not self.ended and not self.closed:
            # Exact .1s controller ticks, .02s internal physics; .2s display samples.
            for _ in range(2):
                if self.ended:
                    break
                self.world.step(self.actions)
                self._refresh_controls()
            self._capture(refresh=False)
            if yield_frames:
                # A long buffer fill must not starve current-state reads or a
                # decision. Publish complete .2s samples, then yield the lock.
                self.lock.release()
                try:
                    time.sleep(0)
                finally:
                    self.lock.acquire()
                if revision != self.revision or self.closed:
                    return False
        return True

    def _index(self, at, ceil=False):
        if type(at) not in (int,float) or not math.isfinite(at) or not 0 <= at <= self.world.time+.001:
            raise ValueError('Choose a time already present in the live world')
        if ceil:
            return min(len(self.frames)-1, max(0, bisect_right([f['t'] for f in self.frames], at-1e-6)))
        return max(0, bisect_right([f['t'] for f in self.frames], at+1e-6)-1)

    def _payload(self, index):
        c = self.context
        return {'source':self.source, 'geometry':c.geometry, 'routes':c.routes, 'meta':c.meta,
                'zones':c.zones, 'owners':c.owners, 'frame':deepcopy(self.frames[index]),
                'progress':dict(self.progress[index]), 'commitments':deepcopy(self.contracts[index])}

    def _packet(self, after=-1., start=None):
        frames = [f for f in self.frames if f['t'] > after+1e-6] if start is None else self.frames[start:]
        return {'revision':self.revision, 'source':self.source, 'frames':deepcopy(frames),
                'time':self.world.time, 'arrivals':dict(self.arrivals), 'events':deepcopy(self.events),
                'ended':self.ended, 'failure':self.failure,
                'active_contracts':[self._contract_dict(c) for c in self.instructions if c.status=='pending'],
                'active_contract':self._contract_dict(self.instruction) if self.instruction else None,
                'model':'Kinematic bicycle physics', 'forecast_model':'Approximate lane projection', 'model_calls':0}

    def _complete(self, key, future):
        try:
            value = future.result()
            with self.lock:
                self.jobs.pop(key, None)
                if key[0]!=self.revision or self.closed:
                    self.discarded += 1
                    return
                self.ready[key] = value
                self.completed += 1
                self.worker_pids.add(value['pid'])
                if type(value.get('seconds')) in (int,float) and value['seconds']>0:
                    self.forecast_seconds=(self.forecast_seconds+[value['seconds']])[-32:]
                while len(self.ready)>96:
                    self.ready.popitem(last=False)
        except Exception as exc:
            with self.lock:
                self.jobs.pop(key, None)
                if future.cancelled() or key[0]!=self.revision or self.closed:
                    self.discarded += 1
                else:
                    self.failed += 1
                    self.ready[key] = {'error':str(exc)}
                    while len(self.ready)>96:
                        self.ready.popitem(last=False)

    def _submit(self, owner, index):
        key = (self.revision, owner, index)
        if key in self.jobs or key in self.ready or len(self.jobs)>=self.workers:
            return
        try:
            future = self.executor.submit(assess_snapshot, self._payload(index), owner)
        except Exception as exc:
            self.ready[key] = {'error':str(exc)}
            self.failed += 1
            while len(self.ready)>96:
                self.ready.popitem(last=False)
            return
        self.jobs[key] = future
        self.peak_pending = max(self.peak_pending, len(self.jobs))
        future.add_done_callback(lambda f,k=key:self._complete(k,f))

    def _stats(self):
        return {'workers':self.workers, 'pending':len(self.jobs), 'ready':len(self.ready),
                'completed':self.completed, 'failed':self.failed, 'discarded':self.discarded,
                'peak_pending':self.peak_pending, 'worker_processes_used':len(self.worker_pids),
                'forecast_mean_job_seconds':round(sum(self.forecast_seconds)/len(self.forecast_seconds),4)
                    if self.forecast_seconds else 0.}

    def _eligible(self, context, owner, index):
        now=self.frames[index]['t']
        result=context.candidates(owner,now)['candidates']
        current,chain=owner,[owner]
        seen=set()
        while current not in seen:
            seen.add(current)
            row=self.frames[index]['cars'][self.context.indices[current]]
            if row[5]!='following' or row[6] not in self.context.indices:
                break
            if row[6] in seen:
                current=owner;break
            current=row[6];chain.append(current)
        if current!=owner:
            ids={a['id'] for a in result}
            for candidate in context.candidates(current,now)['candidates']:
                if candidate['buyer']==current and candidate['id'] not in ids:
                    candidate=deepcopy(candidate)
                    candidate.update(initiator=owner,beneficiaries=[owner,current],
                        causal_chain=chain+[candidate['seller']],
                        participants=list(dict.fromkeys([owner,candidate['seller'],current])))
                    result.append(candidate);ids.add(candidate['id'])
        return result

    def _conflict_reason(self, candidate, index, context=None):
        context=context if context is not None else snapshot_context(self._payload(index))
        if context.hold_obstruction(0,candidate):
            return f"{candidate['buyer']} cannot pass {candidate['seller']} on their shared approach. That car must clear the lane first."
        actors = set(candidate.get('participants',[candidate['seller'],candidate['buyer']]))
        active = [c for c in self.contracts[index] if c.status=='pending']
        for contract in active:
            if actors.intersection(getattr(contract,'participants',[contract.seller,contract.buyer])):
                return 'A participant is already carrying out a binding agreement.'
            if self.context.reservation_conflict(candidate['seller'],candidate['buyer'],candidate['zone_id'],[contract]):
                return 'These agreements reserve overlapping road space.'
        # A promise adds seller -> buyer to the wait graph. Its ordinary
        # buyer-yields-seller edge is replaceable; longitudinal following is not.
        rows = dict(zip(self.context.owners,self.frames[index]['cars']))
        graph = {o:set() for o in rows}
        for owner,row in rows.items():
            if row[4]=='moving' and row[5] in ('following','yielding','roundabout','parked_vehicle') and row[6] in rows:
                graph[owner].add(row[6])
        buyer,seller = candidate['buyer'],candidate['seller']
        if rows[buyer][5] in ('yielding','roundabout'):
            graph[buyer].discard(seller)
        for contract in active:
            graph[contract.seller].add(contract.buyer)
        seen,queue = set(),[buyer]
        while queue:
            node = queue.pop()
            if node==seller:
                return 'This promise would create a circular wait.'
            if node not in seen:
                seen.add(node);queue.extend(graph[node]-seen)
        return None

    def _discover_requests(self, index):
        """Rule-based NPC intents from observed delay causes, never future poses.

        This is one inexpensive shared cause scan, not eighty alternative
        rollouts. It runs only after the viewer has observed this snapshot.
        """
        now = self.frames[index]['t']
        if now-self.last_request_scan<.8:
            return
        self.last_request_scan = now
        context = snapshot_context(self._payload(index))
        rows = dict(zip(context.owners,self.frames[index]['cars']))
        seen_keys, head_candidates = set(),{}
        for requester,row in rows.items():
            if row[4]!='moving':
                continue
            chain,at = [requester],requester
            visited = set()
            while at not in visited:
                visited.add(at)
                current=rows[at]
                if current[5]=='following' and current[6] in rows:
                    at=current[6];chain.append(at);continue
                if current[5] not in ('yielding','roundabout') or current[6] not in rows:
                    break
                recipient=current[6]
                if at not in head_candidates:
                    head_candidates[at]=context.candidates(at,now)['candidates']
                for candidate in head_candidates[at]:
                    if candidate['buyer']!=at or candidate['seller']!=recipient:
                        continue
                    if self._conflict_reason(candidate,index,context):
                        continue
                    key=f"{candidate['id']}|{requester}"
                    seen_keys.add(key)
                    old=self.requests.get(key)
                    if old is not None and old['status'] in ('open','declined','unavailable'):
                        old['last_seen']=now
                        continue
                    same_decline=any(r['candidate_id']==candidate['id'] and r['recipient']==recipient and r['status']=='declined' for r in self.requests.values())
                    self.request_serial+=1
                    request={'request_id':f'request-{self.request_serial}', 'id':candidate['id'],
                        'candidate_id':candidate['id'],'requester':requester,'recipient':recipient,
                        'buyer':at,'seller':recipient,'created_at':now,'last_seen':now,
                        'status':'declined' if same_decline else 'open','origin':'heuristic_delay_request',
                        'causal_chain':chain+[recipient],'beneficiaries':[requester],
                        'point':list(candidate['point']),'kind':candidate['kind'],
                        'explanation':f"{requester} is delayed by "+' → '.join(chain[1:]+[recipient])}
                    self.requests[key]=request
                    self.events.append({**deepcopy(request),'time':now,'kind':'request_created','passage_kind':request['kind']})
                break
        for key,request in self.requests.items():
            if request['status'] not in ('open','declined','unavailable') or key in seen_keys:
                continue
            buyer=request['buyer']
            if buyer not in head_candidates:
                head_candidates[buyer]=context.candidates(buyer,now)['candidates']
            still_possible=rows[request['requester']][4]=='moving' and any(
                c['id']==request['candidate_id'] for c in head_candidates[buyer])
            if not still_possible:
                request['status']='withdrawn';request['withdrawn_at']=now
                self.events.append({'time':now,'kind':'request_withdrawn',
                                    'request_id':request['request_id'],'requester':request['requester'],
                                    'recipient':request['recipient']})

    def _publish_predicted_requests(self, result, index):
        now=self.frames[index]['t']
        if result['time']>now+.001:
            return
        context=snapshot_context(self._payload(index))
        candidates={}
        for descriptor in result.get('request_intents',[]):
            requester=descriptor['requester'];recipient=descriptor['recipient']
            buyer=descriptor.get('buyer',requester)
            if self.frames[index]['cars'][context.indices[requester]][4]!='moving':
                continue
            if buyer not in candidates:
                candidates[buyer]={c['id']:c for c in context.candidates(buyer,now)['candidates']}
            candidate_id=descriptor.get('candidate',descriptor.get('candidate_id',descriptor['id']))
            candidate=candidates[buyer].get(candidate_id)
            if not candidate or self._conflict_reason(candidate,index,context):
                continue
            key=f'{candidate_id}|{requester}'
            old=self.requests.get(key)
            if old is not None and old['status'] in ('open','declined','accepted','unavailable'):
                old['last_seen']=now
                continue
            same_decline=any(r['candidate_id']==candidate_id and r['recipient']==recipient and
                             r['status']=='declined' for r in self.requests.values())
            self.request_serial+=1
            request={'request_id':f'request-{self.request_serial}', 'id':candidate_id,
                'candidate_id':candidate_id,'requester':requester,'recipient':recipient,
                'buyer':buyer,'seller':recipient,'created_at':now,'last_seen':now,
                'status':'declined' if same_decline else 'open','origin':'heuristic_delay_request',
                'causal_chain':list(descriptor.get('causal_chain',[requester,recipient])),
                'beneficiaries':list(descriptor.get('beneficiaries',[requester])),
                'point':list(candidate['point']),'kind':candidate['kind'],
                'forecast_observation_time':result['time'],
                'explanation':'Predicted avoidable delay; request generated by the delayed car.'}
            self.requests[key]=request
            self.events.append({**deepcopy(request),'time':now,'kind':'request_created','passage_kind':request['kind']})

    def _resolve_request_assessment(self, owner, result, index):
        """A checked rejection is a result, rather than endless calculation.

        Keep its stable encounter identity so the shared cause scan cannot
        recreate the request on every poll. A later checked, feasible exchange
        can reopen it; an explicit human refusal remains a refusal.
        """
        now=self.frames[index]['t']
        rejected={c['id']:c for c in result.get('rejected',[])}
        available={c['id'] for c in result['candidates']}
        for request in self.requests.values():
            if request['recipient']!=owner or request['created_at']>now+.001:
                continue
            candidate=request['candidate_id']
            if request['status']=='open' and candidate in rejected:
                request.update(status='unavailable',assessed_at=result['time'],
                    unavailable_at=now,rejection=deepcopy(rejected[candidate]))
                self.events.append({'time':now,'kind':'request_withdrawn',
                    'request_id':request['request_id'],'requester':request['requester'],
                    'recipient':owner,'reason':deepcopy(rejected[candidate])})
            elif request['status']=='unavailable' and candidate in available and result['time']>request['assessed_at']+.001:
                request.update(status='open',last_seen=now)
                self.events.append({'time':now,'kind':'request_reopened',
                    'request_id':request['request_id'],'requester':request['requester'],
                    'recipient':owner})

    def _inbox(self, owner, at, candidates):
        by_id={c['id']:c for c in candidates}
        grouped={}
        for request in self.requests.values():
            if request['recipient']==owner and request['status']=='open' and request['created_at']<=at+.001:
                grouped.setdefault(request['candidate_id'],[]).append(request)
        inbox=[]
        for candidate_id, requests in grouped.items():
            # Prefer the controlled queue head as the visible sender when its
            # followers ask for the same exchange. Individual causes remain
            # recorded; one physical action produces one actionable card.
            lead=next((r for r in requests if r['requester']==r['buyer']),requests[0])
            candidate=by_id.get(candidate_id)
            value={**deepcopy(lead),**deepcopy(candidate or {}),
                'assessment_status':'ready' if candidate else 'checking','actionable':candidate is not None,
                'initiator':lead['requester'],'requesters':[r['requester'] for r in requests],
                'request_ids':[r['request_id'] for r in requests], 'request_id':lead['request_id']}
            inbox.append(value)
        return inbox

    def options(self, owner, at, pace=1.):
        if owner not in self.context.indices:
            raise ValueError('Unknown car')
        if type(pace) not in (int,float) or not math.isfinite(pace) or not 1<=pace<=4:
            raise ValueError('Choose a pace between 1× and 4×')
        index = self._index(at)
        self.latest_observed = max(self.latest_observed, at)
        if owner != self.selected:
            self.selected = owner
            for key, future in list(self.jobs.items()):
                if key[1]!=owner and future.cancel():
                    self.jobs.pop(key, None)
        self._discover_requests(index)
        active=self._active(index)
        relevant=next((c for c in active if owner in c['participants']),None)
        common={'revision':self.revision,'active_contracts':active,
                'active_contract':relevant,'stats':self._stats()}
        if self.frames[index]['cars'][self.context.indices[owner]][4] != 'moving':
            return {**common,'status':'ready','time':self.frames[index]['t'],
                    'candidates':[],'normal_order':[],'incoming_requests':[]}
        self._submit(owner, index)
        cost=sum(self.forecast_seconds)/len(self.forecast_seconds) if self.forecast_seconds else 3.
        cadence=max(.4,cost*pace/max(1,self.workers)*1.3)
        stride=max(2,math.ceil(cadence/STEP))
        for i in range(index+stride, min(len(self.frames),index+math.ceil(16./STEP)+1), stride):
            self._submit(owner, i)
        eligible_context=snapshot_context(self._payload(index))
        eligible = self._eligible(eligible_context,owner,index)
        eligible_ids = {a['id'] for a in eligible}
        eligible_zones = {a['zone_id'] for a in eligible}
        prior = [(k,v) for k,v in self.ready.items() if k[0]==self.revision and k[1]==owner
                 and k[2]<=index and self.frames[index]['t']-v.get('time',-1e9)<=ESTIMATE_MAX_AGE+.001 and 'error' not in v]
        if not prior:
            error = self.ready.get((self.revision,owner,index),{}).get('error')
            return {**common,'status':'failed' if error else 'pending','error':error,
                    'time':self.frames[index]['t'],'incoming_requests':self._inbox(owner,at,[]),'stats':self._stats()}
        key, result = max(prior,key=lambda p:p[0][2])
        self._publish_predicted_requests(result,index)
        self._resolve_request_assessment(owner,result,index)
        candidates,rejected=[],[]
        for candidate in result['candidates']:
            if candidate['id'] not in eligible_ids:
                continue
            if any(r['status']=='declined' and r['recipient']==owner and r['candidate_id']==candidate['id'] for r in self.requests.values()):
                continue
            reason=self._conflict_reason(candidate,index,eligible_context)
            if reason:
                rejected.append({'id':candidate['id'],'reason':reason})
            else:
                candidates.append(deepcopy(candidate))
        return {**common,'status':'ready','time':result['time'],'candidates':candidates,
                'estimate_age_seconds':round(self.frames[index]['t']-result['time'],3),
                'refreshing':key[2]<index,
                'incoming_requests':self._inbox(owner,at,candidates),'rejection_reasons':rejected,
                'normal_order':[deepcopy(o) for o in result['normal_order'] if o['zone_id'] in eligible_zones],
                'stats':self._stats(),
                'scope':'Conditional delay alternatives, including independently initiated requests.'}

    def decline(self, owner, at, request_id):
        self._index(at)
        request=next((r for r in self.requests.values() if r['request_id']==request_id),None)
        if request is None or request['recipient']!=owner or request['created_at']>at+.001:
            raise ValueError('That request is no longer open for this car.')
        if request['status']=='declined':
            return {'status':'declined','revision':self.revision,'request_id':request_id}
        if request['status']!='open':
            raise ValueError('That request is no longer open for this car.')
        at=max(at,self.latest_observed)
        for other in self.requests.values():
            if other['candidate_id']==request['candidate_id'] and other['recipient']==owner and other['status']=='open':
                other['status']='declined';other['declined_at']=at
                self.events.append({'time':at,'kind':'request_declined','request_id':other['request_id'],
                    'requester':other['requester'],'recipient':owner})
        return {'status':'declined','revision':self.revision,'request_id':request_id}

    def future(self, owner, at, candidate):
        if owner not in self.context.indices or not isinstance(candidate,str):
            raise ValueError('Choose a current car and option')
        index = self._index(at)
        matches = [(k,v) for k,v in self.ready.items() if k[0]==self.revision and k[1]==owner and
                   k[2]<=index and self.frames[index]['t']-v.get('time',-1e9)<=ESTIMATE_MAX_AGE+.001 and candidate in v.get('futures',{})]
        if not matches:
            self._submit(owner,index)
            return {'status':'pending', 'revision':self.revision}
        key, value = max(matches,key=lambda p:p[0][2])
        result = deepcopy(value['futures'][candidate])
        result['revision'] = self.revision
        return result

    def apply(self, owner, at, candidate, pace=1.):
        if type(pace) not in (int,float) or not math.isfinite(pace) or not 1<=pace<=4:
            raise ValueError('Choose a pace between 1× and 4×')
        if owner not in self.context.indices:
            raise ValueError('Unknown car')
        if isinstance(candidate,str) and candidate.startswith('request-'):
            request=next((r for r in self.requests.values() if r['request_id']==candidate),None)
            if request is None or request['recipient']!=owner or request['status']!='open' or request['created_at']>at+.001:
                raise ValueError('That request is no longer available to this car.')
            candidate=request['candidate_id']
        # A decision starts at the next control tick; it never rewinds displayed
        # time or replaces reality with the old speculative trajectory.
        index = self._index(at, ceil=True)
        tick = self.frames[index]['t']
        if self.failure and tick>=self.world.time-1e-6:
            raise ValueError('The drive stopped after an execution fault. Start a new drive.')
        if tick < self.latest_observed-.001 or any(e['kind']=='accepted' and e['time']>tick for e in self.events):
            raise ValueError('This is a past observation. Return to the current drive before making a decision.')
        recent = [(k,v) for k,v in self.ready.items() if k[0]==self.revision and k[1]==owner and
                  k[2]<=index and self.frames[index]['t']-v.get('time',-1e9)<=ESTIMATE_MAX_AGE+.001
                  and any(a['id']==candidate for a in v.get('candidates',[]))]
        if not recent:
            raise ValueError('That move is out of date. Use a current option.')
        newest = [(k,v) for k,v in self.ready.items() if k[0]==self.revision and k[1]==owner and
                  k[2]<=index and 'error' not in v]
        if newest:
            _, latest = max(newest,key=lambda pair:pair[0][2])
            if any(a['id']==candidate for a in latest.get('rejected',[])):
                raise ValueError('A newer forecast ruled out that move. Use a current option.')
        _,recent_result=max(recent,key=lambda pair:pair[0][2])
        cached=next(a for a in recent_result['candidates'] if a['id']==candidate)
        context = snapshot_context(self._payload(index))
        available = self._eligible(context,owner,index)
        selected = next((a for a in available if a['id']==candidate),None)
        if selected is None and owner in cached.get('beneficiaries',[]):
            # A queue member may ask for its head's passage; it cannot invent
            # that dependency after the queue has separated.
            at_owner,seen=owner,set()
            while at_owner not in seen and at_owner!=cached['buyer']:
                seen.add(at_owner)
                row=self.frames[index]['cars'][self.context.indices[at_owner]]
                if row[5]!='following' or row[6] not in self.context.indices:
                    break
                at_owner=row[6]
            if at_owner==cached['buyer']:
                selected=next((a for a in context.candidates(at_owner,tick)['candidates'] if a['id']==candidate),None)
                if selected is not None:
                    for name in ('participants','beneficiaries','causal_chain','initiator'):
                        if name in cached:selected[name]=deepcopy(cached[name])
        if selected is None:
            raise ValueError('Those cars can no longer exchange this passage. Driving continues.')
        if any(self.frames[index]['cars'][self.context.indices[o]][4]!='moving' for o in (selected['seller'],selected['buyer'])):
            raise ValueError('That passage is no longer active.')
        for name in ('action_type','changes_order','delivery_status','donor_motion_changes'):
            if name in cached:selected[name]=deepcopy(cached[name])
        conflict=self._conflict_reason(selected,index,context)
        if conflict:
            raise ValueError(conflict)
        retained=deepcopy(self.contracts[index])
        # Restore the current control tick, discard only unobserved buffered
        # continuation, then generate reality from the installed contract.
        self.world = copy(self.world)
        self.world.cars = deepcopy(self.native_states[index])
        self.world._time, self.world.horizon = tick, 600.
        self.world.termination, self.world._step_events = None, []
        self.world.events = [e for e in self.world.events if e['time']<=tick]
        self.world._speeding = set()
        self.driver = Driver(self.context.routes,self.context.meta,self.context.zones,self.context.geometry['roundabout'])
        self.driver.progress = dict(self.progress[index])
        self.frames, self.progress, self.contracts, self.native_states = (self.frames[:index],self.progress[:index],
            self.contracts[:index],self.native_states[:index])
        self.events = [e for e in self.events if e['time']<=tick]
        self.arrivals = {o:t for o,t in self.arrivals.items() if t<=tick}
        self.ended, self.failure = False, None
        self.latest_observed = max(self.latest_observed, at)
        self.instructions = retained
        instruction = context.instruction(0,selected)
        # Forecast timeout is not permission to break a live accepted promise.
        # The live execution holds it until delivery or an explicit order fault.
        instruction.expires = 1e9
        instruction.id = f'{candidate}@{tick:.4f}'
        instruction.participants = list(selected.get('participants',[selected['seller'],selected['buyer']]))
        instruction.action_type=selected.get('action_type','priority_exchange')
        self.instructions.append(instruction)
        for request in self.requests.values():
            if request['status']=='open' and request['candidate_id']==candidate:
                request['status']='accepted';request['accepted_at']=tick
        self.events.append({'time':tick, 'kind':'accepted', **self._contract_dict(instruction)})
        self.revision += 1
        for f in list(self.jobs.values()):
            f.cancel()
        self.jobs.clear();self.ready.clear()
        self._capture()
        self._advance(tick+max(2.,pace))
        packet = self._packet(start=index)
        packet.update(at=tick, accepted=selected)
        return packet

    def command(self, payload):
        with self.lock:
            if self.closed:
                raise ValueError('The demonstration has stopped')
            action = payload.get('action')
            if action=='new':
                self._new()
                return {**self._packet(), 'owners':list(self.context.owners), 'routes':deepcopy(self.context.meta)}
            requested_revision = payload.get('revision')
            if action in ('advance','options','future') and type(requested_revision) is int and requested_revision!=self.revision:
                return {'status':'superseded', 'revision':self.revision, 'source':self.source, 'frames':[]}
            self._check_revision(requested_revision)
            if action=='advance':
                to, after = payload.get('to'), payload.get('after',-1.)
                if type(to) not in (int,float) or not math.isfinite(to) or not 0<=to<=600.:
                    raise ValueError('Invalid advance time')
                if type(after) not in (int,float) or not math.isfinite(after):
                    raise ValueError('Invalid frame cursor')
                if not self._advance(min(to,self.world.time+8.), yield_frames=True):
                    return {'status':'superseded', 'revision':self.revision, 'source':self.source, 'frames':[]}
                return self._packet(after)
            if action=='options':
                return self.options(payload.get('car'),payload.get('at'),payload.get('pace',1.))
            if action=='future':
                return self.future(payload.get('car'),payload.get('at'),payload.get('candidate'))
            if action=='decline':
                return self.decline(payload.get('car'),payload.get('at'),payload.get('request_id'))
            if action=='apply':
                return self.apply(payload.get('car'),payload.get('at'),payload.get('candidate',payload.get('request_id')),payload.get('pace',1.))
            raise ValueError('Unknown demonstration action')

    def close(self):
        with self.lock:
            try:
                self._save('close')
            finally:
                self.closed = True
                for future in list(self.jobs.values()):
                    future.cancel()

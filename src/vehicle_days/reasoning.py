"""Rolling public-traffic reasoning. No learned model or private utilities.

One coupled ordinary forecast supplies passage order and blocking dependencies for
all cars. Alternatives can exchange priority, coordinate an earlier gap, or
release an observed queue. Incomplete forecasts stay conditional; a known order
fault or a persistent blocking cycle is rejected. No welfare reward is used.
"""
from collections import OrderedDict
from copy import deepcopy
import math
import time

from .encounters import Context, commitment_wait_cycle, first_crossing, following_chain, project


class RollingContext(Context):
    """Memoise unchanged forecast calculations without changing their semantics."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forecasts = OrderedDict()

    def forecast_view(self, index, candidate):
        """Internal read-only view; callers must not expose or mutate traces."""
        key = (index, candidate['id'])
        if key not in self.forecasts:
            self.forecasts[key] = super().forecast(index, candidate)
            while len(self.forecasts) > 24:
                self.forecasts.popitem(last=False)
        return self.forecasts[key]

    def forecast(self, index, candidate):
        return deepcopy(self.forecast_view(index,candidate))


def passage_evidence(context, frames, candidate, index):
    zone = next(z for z in context.zones if z['id'] == candidate['zone_id'])
    owners = [candidate['seller'], candidate['buyer']]
    near = {o:context.progress[index][o] for o in owners}
    samples = []
    for frame in frames:
        for o in owners:
            row = frame['cars'][context.indices[o]]
            near[o] = context.routes[o].project(row[0], row[1], near[o])[0]
        samples.append((frame['t'], dict(near)))
    timing = {o:{'enters':first_crossing(samples,o,zone['ranges'][o][0]),
                 'clears':first_crossing(samples,o,zone['ranges'][o][1])} for o in owners}
    buyer_clear = timing[candidate['buyer']]['clears']
    seller_entry = timing[candidate['seller']]['enters']
    return {'timing':timing,'timing_method':'Crossing times interpolated from replay samples; delivery is separately observed at controller ticks.','order_observed':buyer_clear is not None and
        (seller_entry is None or buyer_clear <= seller_entry+.01)}


def blockers(context, frames, candidate, until=None):
    found = {}
    for f in frames:
        if until is not None and f['t'] > until:
            break
        for o in (candidate['seller'], candidate['buyer']):
            row = f['cars'][context.indices[o]]
            if row[5] not in ('yielding','roundabout','following','parked_vehicle') or not row[6]:
                continue
            if row[6] in (candidate['seller'], candidate['buyer']):
                continue
            key = (o,row[5],row[6])
            if key not in found:
                found[key] = {'car':o,'reason':row[5],'other':row[6],'first_seen':f['t'],'last_seen':f['t']}
            found[key]['last_seen'] = f['t']
    return list(found.values())


def persistent_wait_cycle(frames, owners, affected, promise_edges=None):
    """Find a stationary, unchanged blocking cycle in the forecast tail.

    A forecast ending before passage is not itself a fault. A cycle of cars
    continually waiting on one another is a known reason not to bind this deal.
    """
    if not frames or frames[-1]['t']-frames[0]['t']<2.:
        return []
    tail=[f for f in frames if f['t']>=frames[-1]['t']-2.]
    stable=None
    for frame in tail:
        edges={(owner,row[6]) for owner,row in zip(owners,frame['cars'])
               if row[4]=='moving' and row[3]<.15 and
               row[5] in ('following','yielding','roundabout','parked_vehicle') and row[6] in owners}
        # The displayed reason is only the strongest current speed constraint.
        # A binding hold still exists when a different car masks that reason.
        edges.update((seller,buyer) for seller,buyer in (promise_edges or {}).get(frame['t'],())
                     if frame['cars'][owners.index(seller)][3]<.15)
        stable=edges if stable is None else stable & edges
    graph={}
    for owner,peer in stable or ():
        graph.setdefault(owner,set()).add(peer)
    done,path,visiting=set(),[],{}
    def visit(owner):
        if owner in visiting:
            return path[visiting[owner]:]+[owner]
        if owner in done:
            return []
        visiting[owner]=len(path);path.append(owner)
        for peer in sorted(graph.get(owner,())):
            cycle=visit(peer)
            if cycle:
                return cycle
        path.pop();visiting.pop(owner);done.add(owner)
        return []
    for root in affected:
        cycle=visit(root)
        if cycle:
            return cycle
    return []


def held_promise_edges(context, index, frames, candidate):
    """Hard dependencies at reached hold boundaries, using public paths only.

    Merely following a donor before an earlier fork is not sufficient: the
    donor may still advance beyond that fork and leave the buyer room to turn.
    The promise becomes a stationary wait edge here only near its own boundary.
    """
    world=context.public_world(index)
    promises=[*context.public_commitments(index),context.instruction(index,candidate,True)]
    owners={o for p in promises for o in (p.seller,p.buyer)}
    previous={o:world['routes'][o].project(*frames[0]['cars'][context.indices[o]][:2],5.)[0]
              for o in owners}
    result={}
    for frame in frames:
        for owner in owners:
            row=frame['cars'][context.indices[owner]]
            previous[owner]=world['routes'][owner].project(*row[:2],previous[owner])[0]
        edges=set()
        for promise in promises:
            seller=frame['cars'][context.indices[promise.seller]]
            if not promise.start<=frame['t']<=promise.expires:
                continue
            if previous[promise.buyer]>promise.buyer_exit:
                continue
            distance=promise.seller_entry-previous[promise.seller]
            if seller[4]=='moving' and seller[3]<.15 and -.2<=distance<=5.5:
                edges.add((promise.seller,promise.buyer))
        result[frame['t']]=edges
    return result


def terminal_parked_queue(frames, owners, buyer):
    """A motionless queue ending at a parked body cannot sell clearance."""
    if not frames or frames[-1]['t']-frames[0]['t']<2.:
        return []
    tail=[f for f in frames if f['t']>=frames[-1]['t']-2.]
    rows=dict(zip(owners,tail[-1]['cars']))
    chain=[buyer]
    while chain[-1] in rows:
        row=rows[chain[-1]]
        if row[3]>=.15 or row[6] not in rows or row[6] in chain:
            return []
        if row[5]=='parked_vehicle' and rows[row[6]][4]=='arrived':
            chain.append(row[6]);break
        if row[5]!='following':
            return []
        chain.append(row[6])
    else:
        return []
    if len(chain)<2 or rows[chain[-1]][4]!='arrived':
        return []
    for frame in tail:
        current=dict(zip(owners,frame['cars']))
        if any(current[o][3]>=.15 or current[o][6]!=p for o,p in zip(chain,chain[1:])):
            return []
    return chain


def _waits_for(frames, indices, buyer, seller):
    return any(f['cars'][indices[buyer]][5] in ('yielding','roundabout') and
               f['cars'][indices[buyer]][6]==seller for f in frames)


def _early_gap_changes(baseline, alternative, indices, buyer, seller):
    """A same-order offer needs an actual counterparty and motion change."""
    end=min(baseline[-1]['t'],alternative[-1]['t'])
    ordinary={round(f['t'],5):f['cars'] for f in baseline if f['t']<=end+.001}
    donor_changed=buyer_changed=False
    for frame in alternative:
        before=ordinary.get(round(frame['t'],5))
        if before is None:
            continue
        s,b=indices[seller],indices[buyer]
        donor_changed |= abs(before[s][8]-frame['cars'][s][8])>.05
        buyer_changed |= (abs(before[b][3]-frame['cars'][b][3])>.05 or
                          math.dist(before[b][:2],frame['cars'][b][:2])>.05)
    return {'buyer_motion_changes':buyer_changed,'donor_motion_changes':donor_changed}


class TrafficReasoner:
    def __init__(self, encounters):
        self.encounters = encounters
        self.joint = OrderedDict()
        self.assessed = OrderedDict()

    def _joint(self, index):
        c = self.encounters.context
        if index in self.joint:
            return self.joint[index]
        world = c.public_world(index)
        if index not in c._baseline_forecasts:
            c._baseline_forecasts[index] = project(world,c.geometry['roundabout'],c.times[index],
                                                     existing=c.public_commitments(index))
        baseline = c._baseline_forecasts[index]
        agents = {o:{'candidates':[],'normal_order':[],'unresolved':[]} for o in c.owners}
        observed=dict(zip(c.owners,baseline['frames'][0]['cars']))
        delay_pairs=set()
        for frame in baseline['frames']:
            rows=dict(zip(c.owners,frame['cars']))
            for owner,row in rows.items():
                if row[5] in ('yielding','roundabout') and row[6] in rows:
                    delay_pairs.add((owner,row[6]))
                elif row[5]=='following':
                    queue=following_chain(rows,owner)
                    head=rows[queue[-1]]
                    if head[5] in ('yielding','roundabout') and head[6] in rows:
                        delay_pairs.add((owner,head[6]))
        pairs = {}
        for owner in c.owners:
            for proposal in c.candidates(owner,c.times[index])['candidates']:
                pairs.setdefault(proposal['zone_id'],{})[proposal['seller']] = proposal
        direct=[]
        for zid, directions in pairs.items():
            candidate = next(iter(directions.values()))
            owners = [candidate['seller'],candidate['buyer']]
            zone = next(z for z in world['zones'] if z['id']==zid)
            entry = {o:first_crossing(baseline['progress'],o,zone['ranges'][o][0]) for o in owners}
            known = sorted((t,o) for o,t in entry.items() if t is not None)
            first=known[0][1] if known and not (len(known)==2 and abs(known[0][0]-known[1][0])<.01) else None
            order_basis='predicted_entry'
            if first is None:
                waits={o:_waits_for(baseline['frames'][:1],c.indices,o,next(p for p in owners if p!=o)) for o in owners}
                if sum(waits.values())==1:
                    first=next(o for o in owners if not waits[o])
                    order_basis='current_blocker'
                else:
                    waits={o:_waits_for(baseline['frames'],c.indices,o,next(p for p in owners if p!=o)) for o in owners}
                    if sum(waits.values())==1:
                        first=next(o for o in owners if not waits[o])
                        order_basis='forecast_blocker'
            if first is None:
                for o in owners:
                    agents[o]['unresolved'].append({'zone_id':zid,'reason':'ordinary_order_unresolved'})
                continue
            second = next(o for o in owners if o!=first)
            for o in owners:
                agents[o]['normal_order'].append({'zone_id':zid,'first':first,'second':second})
            proposals=[]
            if first in directions and (second,first) in delay_pairs:
                proposals.append((directions[first],'priority_exchange',True))
            # Ordinary final order can stay unchanged while the priority holder
            # leaves an earlier gap and prevents another car's unnecessary stop.
            if second in directions:
                clear=first_crossing(baseline['progress'],first,zone['ranges'][first][1])
                before_clear=[f for f in baseline['frames'] if clear is None or f['t']<=clear+.001]
                if _waits_for(before_clear,c.indices,first,second):
                    proposals.append((directions[second],'early_gap',False))
            for raw,kind,changes_order in proposals:
                proposal=deepcopy(raw)
                proposal.update(normal_order=[first,second],ordinary_entry_times=entry,
                    ordinary_order_basis=order_basis,
                    assessment='requires_conditional_alternative',action_type=kind,changes_order=changes_order,
                    initiator=proposal['buyer'],beneficiaries=[proposal['buyer']],
                    causal_chain=[proposal['buyer'],proposal['seller']],
                    participants=[proposal['seller'],proposal['buyer']])
                direct.append(proposal)
                for o in owners:
                    agents[o]['candidates'].append(deepcopy(proposal))
        # A queued car can ask the remote priority holder to release its head.
        # This is a B/C passage instruction with A as initiator/beneficiary,
        # never an invented A/B overtake on a single lane.
        for owner in c.owners:
            chain=following_chain(observed,owner)
            if len(chain)<2:
                continue
            for proposal in direct:
                if proposal['buyer'] not in chain[1:] or owner in proposal['participants']:
                    continue
                stop=chain.index(proposal['buyer'])
                indirect=deepcopy(proposal)
                indirect.update(initiator=owner,beneficiaries=[owner,proposal['buyer']],
                    causal_chain=chain[:stop+1]+[proposal['seller']],
                    participants=list(dict.fromkeys([owner,proposal['seller'],proposal['buyer']])),
                    request_reason='release_my_queue')
                if not any(p['id']==indirect['id'] for p in agents[owner]['candidates']):
                    agents[owner]['candidates'].append(indirect)
        requests=[]
        for owner,agent in agents.items():
            for proposal in agent['candidates']:
                if proposal['initiator']!=owner:
                    continue
                requests.append({'id':proposal['id']+'@'+owner,'candidate':proposal['id'],
                    'requester':owner,'recipient':proposal['seller'],'seller':proposal['seller'],
                    'buyer':proposal['buyer'],'beneficiaries':proposal['beneficiaries'],
                    'causal_chain':proposal['causal_chain'],'point':proposal['point'],
                    'action_type':proposal['action_type'],'changes_order':proposal['changes_order'],
                    'reason':proposal.get('request_reason','avoid_expected_yield'),
                    'status':'awaiting_local_forecast','observation_time':c.times[index]})
        value = {'agents':agents,'baseline':baseline,'requests':requests}
        self.joint[index] = value
        while len(self.joint)>6:
            self.joint.popitem(last=False)
        return value

    def assess(self, owner, at):
        started = time.perf_counter()
        with self.encounters.lock:
            c = self.encounters.context
            if owner is not None and owner not in c.indices:
                raise ValueError('Unknown car')
            index = c.frame_index(at)
            key = (index,owner)
            if key in self.assessed:
                result = deepcopy(self.assessed[key])
                result['cache_hit'] = True
                result['request_seconds'] = round(time.perf_counter()-started,4)
                return result
            joint = self._joint(index)
            actions, rejected = [], []
            for proposal in (joint['agents'][owner]['candidates'] if owner is not None else []):
                current=dict(zip(c.owners,c.record['frames'][index]['cars']))
                known_cycle=commitment_wait_cycle(current,proposal['seller'],proposal['buyer'],
                                                   getattr(c,'commitments',[]))
                if known_cycle:
                    rejected.append({'id':proposal['id'],'seller':proposal['seller'],'buyer':proposal['buyer'],
                        'reason':'current_blocking_cycle','blocking_cycle':known_cycle,
                        'explanation':'This promise would close a current queue/blocking cycle.'})
                    continue
                obstruction=c.hold_obstruction(index,proposal)
                if obstruction:
                    rejected.append({'id':proposal['id'],'seller':proposal['seller'],'buyer':proposal['buyer'],
                        'reason':'held_path_obstruction','obstruction':obstruction,
                        'explanation':'The promised hold would obstruct the buyer’s only announced passage.'})
                    continue
                forecast = (c.forecast_view(index,proposal) if hasattr(c,'forecast_view') else c.forecast(index,proposal))
                branch = forecast['_branches']['concession']
                evidence = passage_evidence(c,branch['frames'],proposal,index)
                tail=branch['frames'][-1]
                delivered=forecast['concession_status']=='delivered' and evidence['order_observed']
                promises={} if delivered else held_promise_edges(c,index,branch['frames'],proposal)
                queue=following_chain(dict(zip(c.owners,tail['cars'])),proposal['buyer'])
                tail_obstruction=(c.hold_obstruction(index,proposal,frame=tail)
                    if not delivered and proposal['seller'] in queue[1:] else None)
                queued=not delivered and proposal['seller'] in queue[1:] and bool(tail_obstruction or (
                    proposal['seller'],proposal['buyer']) in promises.get(tail['t'],()))
                until=evidence['timing'][proposal['buyer']]['clears'] if delivered else None
                before_clear=[f for f in branch['frames'] if until is None or f['t']<=until+.001]
                cycle=[] if delivered else persistent_wait_cycle(before_clear,c.owners,
                    [proposal['buyer'],proposal['seller']],promises)
                parked=[] if delivered else terminal_parked_queue(before_clear,c.owners,proposal['buyer'])
                status=forecast['concession_status']
                invalid=not forecast.get('existing_commitments_preserved',True) or status not in ('pending','delivered') or queued or cycle or parked or (
                    status=='delivered' and not evidence['order_observed'])
                if invalid:
                    rejected.append({'id':proposal['id'],'seller':proposal['seller'],'buyer':proposal['buyer'],
                        'reason':('buyer_queued_behind_seller' if queued else 'blocking_cycle' if cycle else 'parked_obstruction' if parked else
                                  'existing_commitment_conflict' if not forecast.get('existing_commitments_preserved',True) else
                                  status if status!='delivered' else 'order_not_observed'),
                        'explanation':('The buyer queues behind the car it asks to wait; this manoeuvre cannot reverse their lane order.' if queued else
                                       'The forecast has an unchanged stationary blocking cycle.' if cycle else
                                       'The queue ends at a parked body and cannot clear the promised passage.' if parked else
                                       'The forecast violates the required passage order or an explicit deadline.'),
                        'buyer_queued_behind_seller':queued,'blocking_queue':queue if queued else [],
                        'blocking_cycle':cycle,'parked_queue':parked})
                    continue
                gap_motion=None
                if proposal['action_type']=='early_gap':
                    gap_motion=_early_gap_changes(joint['baseline']['frames'],branch['frames'],c.indices,
                                                  proposal['buyer'],proposal['seller'])
                    if not gap_motion['buyer_motion_changes']:
                        rejected.append({'id':proposal['id'],'seller':proposal['seller'],'buyer':proposal['buyer'],
                            'reason':'redundant_default_gap','explanation':'Normal driving already gives the same gap.'})
                        continue
                action = deepcopy(proposal)
                action.update(assessment='predicted_deliverable_exchange' if status=='delivered' else 'conditional_delay_exchange',
                    delivery_status='predicted' if status=='delivered' else 'conditional',
                    predicted_effects=deepcopy(forecast['outcomes']),delivery_evidence=evidence,
                    remaining_constraints=blockers(c,branch['frames'],proposal,
                        evidence['timing'][proposal['buyer']]['clears']))
                if gap_motion is not None:
                    action.update(gap_motion)
                actions.append(action)
            result = {'source':c.source,'snapshot':index,'time':c.times[index],
                'horizon_seconds':25.,'candidates':actions,
                'normal_order':joint['agents'][owner]['normal_order'] if owner is not None else [],
                'rejected':rejected,'unresolved':joint['agents'][owner]['unresolved'] if owner is not None else [],
                'all_agents':deepcopy(joint['agents']),'requests':deepcopy(joint['requests']),'cars_assessed':len(c.owners),
                'method':'Shared public-state forecast, causal queue graph, safe passage exchanges and early-gap comparisons.',
                'scope':'Detailed alternatives are evaluated for the inspected car. Other cars can initiate provisional requests from their own predicted delay. Incomplete passage remains conditional; no welfare/profit filter or learned-model claim.',
                'seconds':round(time.perf_counter()-started,4),'cache_hit':False}
            self.assessed[key] = deepcopy(result)
            while len(self.assessed)>16:
                self.assessed.popitem(last=False)
            return result

    def require_action(self, owner, at, candidate_id, snapshot):
        result = self.assess(owner,at)
        if snapshot != result['snapshot']:
            raise ValueError('Traffic has changed; recalculate the current options')
        action = next((a for a in result['candidates'] if a['id']==candidate_id),None)
        if action is None:
            raise ValueError('This move is normal priority, unavailable, or not physically feasible now')
        return action

    def execution_evidence(self, result):
        with self.encounters.lock:
            job = self.encounters.jobs[result['id']]['result']
            field = 'branches' if result['status'] in ('complete','failed') and 'branches' in job else 'forecast_branches'
            branch = job.get(field,{}).get('concession')
            if not branch:
                return None
            c = self.encounters.context
            evidence = passage_evidence(c,branch['frames'],result['candidate'],result['snapshot'])
            evidence.update(kind='physical' if field=='branches' else 'forecast',
                commitment_status=branch.get('concession_status'),delivery_observed_at=branch.get('delivered_at'),
                other_constraints=blockers(c,branch['frames'],result['candidate'],
                    evidence['timing'][result['candidate']['buyer']]['clears']))
            return evidence

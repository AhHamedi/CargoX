"""Public-intent forecasts and isolated, evaluator-only physical comparisons.

Forecasts never read future recording frames or private peer goals. The viewer's
physical validation explicitly uses the full scripted world; it is not an agent
observation or a claim of real-time execution.
"""
from __future__ import annotations

from bisect import bisect_right
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import lru_cache
import gzip
import hashlib
import json
import math
from pathlib import Path
import threading
import time

from .driving import Simulation, VehicleConfig
from .driving_roads import nearest_on_path
from .layout_traffic import (Driver, MAP, OUTPUT, ROOT, Route, SPEED, make_scenario,
                            passage_overlap)

HORIZON = 25.
FORECAST_STEP = .2
MAX_WAIT = 20.
VERSION = 'passage-inspector-v1'


@lru_cache(maxsize=4096)
def can_hold_before_entry(distance, speed, acceleration=0.):
    """Can the existing longitudinal controller keep this seller outside?

    Use one display/control-dispatch interval of continuing observed motion,
    then the actual stop-boundary/cruise loop. This is a feasibility screen,
    not permission to override physical collision or lateral constraints.
    In particular a stationary car need not move backwards to a nominal line.
    """
    if not all(math.isfinite(v) for v in (distance,speed,acceleration)) or distance<=.2:
        return False
    speed=max(0.,speed)
    acceleration=max(0.,min(2.,acceleration))
    remaining=distance-speed*FORECAST_STEP-.5*acceleration*FORECAST_STEP**2
    speed+=acceleration*FORECAST_STEP
    # Once the stop buffer is reached, even an abrupt zero-speed target takes
    # at most this servo distance. Far approaches need no iterative work.
    servo=lambda v:(v/1.5 if v<=4. else (v*v-16.)/12.+4./1.5)
    if remaining>=5.+servo(max(speed,SPEED))+.1*max(speed,SPEED)+.2:
        return True
    for _ in range(200):
        if remaining<=.2:
            return False
        target=min(SPEED,math.sqrt(2.*max(0.,remaining-5.)))
        if target==0. and speed<.01:
            return True
        control=max(-6.,min(2.,1.5*(target-speed)))
        following=max(0.,speed+control*.1)
        remaining-=(speed+following)*.05
        speed=following
    return False


def following_chain(cars, owner):
    """Public controller causes, restricted to a currently observed queue."""
    chain,seen=[owner],{owner}
    while chain[-1] in cars:
        row=cars[chain[-1]]
        if row[5]!='following' or row[6] not in cars or row[6] in seen:
            break
        chain.append(row[6]);seen.add(row[6])
    return chain


def shared_hold_obstruction(seller_route, buyer_route, seller_s, buyer_s,
                            seller_entry, buyer_exit, seller_pose=None):
    """Recognise an order reversal that the single-lane controller cannot drive.

    A future holding point alone is insufficient: on different approaches the
    buyer may get there first. Require an already-established lead and a
    continuous co-directed shared approach all the way to the holding area.
    A fork, bypass, opposite lane or separate approach leaves this unproven.
    All geometry is bounded by the announced passage, not a private route tail.
    """
    start=max(0.,buyer_s)
    end=min(buyer_exit,buyer_route.length)
    hold_end=min(seller_entry-.2,seller_route.length)
    if end<=start+.2 or seller_s>=hold_end:
        return None
    points=[list(buyer_route.at(start)[:2])]
    points.extend(list(p) for p,s in zip(buyer_route.points,buyer_route.ss) if start<s<end)
    points.append(list(buyer_route.at(end)[:2]))
    # Arc-length roundoff can clip exactly onto an existing world-space vertex.
    points=[p for i,p in enumerate(points) if not i or math.dist(p,points[i-1])>1e-7]
    if len(points)<2:
        return None
    def match(pose):
        distance_s,_,distance=nearest_on_path(points,pose[0],pose[1])
        s=start+distance_s
        alignment=math.cos(pose[2]-buyer_route.at(s)[2])
        if distance>1.2 or alignment<.85 or s>=end-.2:
            return None
        return s,distance,alignment
    current=match(seller_pose or seller_route.at(seller_s))
    # Body centres must already be ordered, rather than merely close later.
    if current is None or current[0]<start+4.3:
        return None
    held=match(seller_route.at(hold_end))
    if held is None or held[0]<current[0]:
        return None
    # Include every bend vertex, with bounded spacing along straight sections.
    # The narrow corridor leaves margin relative to the driver's 2.6 m test.
    count=max(1,math.ceil((hold_end-seller_s)/.5))
    samples={seller_s+(hold_end-seller_s)*i/count for i in range(count+1)}
    samples.update(s for s in seller_route.ss if seller_s<s<hold_end)
    previous=current[0]
    for s in sorted(samples):
        value=match(seller_route.at(s))
        if value is None or value[0]<previous-.05:
            return None
        previous=value[0]
    nominal=seller_route.at(max(seller_s,seller_entry-5.))
    at_hold=match(nominal)
    return {'reason':'shared_lane_order',
            'holding_point':list(nominal[:2]),
            'buyer_position':round(start,3),
            'buyer_at_hold':round(at_hold[0] if at_hold else held[0],3),
            'buyer_clearance':round(end,3),
            'explanation':'The yielder already leads on the shared approach; holding it blocks the promised clearance.'}


def commitment_wait_cycle(cars, seller, buyer, commitments=()):
    """Known current wait dependencies that a new binding edge would close."""
    graph={owner:set() for owner in cars}
    for owner,row in cars.items():
        if row[4]=='moving' and row[5] in ('following','yielding','roundabout','parked_vehicle') and row[6] in cars:
            graph[owner].add(row[6])
    if cars[buyer][5] in ('yielding','roundabout'):
        graph[buyer].discard(seller)
    for contract in commitments:
        value=contract if isinstance(contract,dict) else vars(contract)
        if value.get('status','pending')=='pending':
            graph[value['seller']].add(value['buyer'])
    queue=[(buyer,[buyer])];seen=set()
    for at,path in queue:
        if at==seller:
            return [seller]+path
        if at not in seen:
            seen.add(at)
            queue.extend((peer,path+[peer]) for peer in graph[at] if peer not in seen)
    return []


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(gzip.compress(json.dumps(value, separators=(',', ':'), allow_nan=False).encode(), mtime=0))
    temporary.replace(path)


@dataclass
class Concession:
    seller: str
    buyer: str
    zone_id: int
    seller_entry: float
    buyer_exit: float
    start: float
    expires: float
    status: str = 'pending'
    delivered_at: float | None = None

    def observe(self, progress, now):
        if self.status != 'pending':
            return
        if progress[self.buyer] > self.buyer_exit:
            self.status, self.delivered_at = 'delivered', now
        elif progress[self.seller] > self.seller_entry:
            self.status = 'violated_order'
        elif now > self.expires:
            self.status = 'delivery_timeout'


class Context:
    def __init__(self, recording_path=OUTPUT, map_path=MAP):
        raw = Path(recording_path).read_bytes()
        self.record = json.loads(gzip.decompress(raw))
        self.geometry = json.loads(Path(map_path).read_text())
        if self.record['geometry_sha256'] != hashlib.sha256(Path(map_path).read_bytes()).hexdigest():
            raise ValueError('The recording and road geometry do not match')
        if self.record.get('negotiations') != 0:
            raise ValueError('This inspector currently accepts the model-free recording with no existing contracts')
        self.source = hashlib.sha256(raw).hexdigest()
        self.runtime = digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), ROOT/'src/vehicle_days/layout_traffic.py', ROOT/'src/vehicle_days/driving.py']})
        self.routes = {o: Route(v['points']) for o, v in self.record['routes'].items()}
        self.meta = self.record['routes']
        self.zones = self.record['conflicts']
        self.times = [f['t'] for f in self.record['frames']]
        self.owners = self.record['owners']
        self.indices = {o:i for i,o in enumerate(self.owners)}
        # This derives branch state from observed history, never future poses.
        progress = {o:5. for o in self.owners}
        self.progress = []
        for frame in self.record['frames']:
            for o,c in zip(self.owners, frame['cars']):
                progress[o] = self.routes[o].project(c[0],c[1],progress[o])[0]
            self.progress.append(dict(progress))
        self._public, self._baseline_forecasts = {}, {}
        self._template = None
        self.baseline_audit = None

    def frame_index(self, at):
        if type(at) not in (int,float) or not math.isfinite(at) or not 0 <= at <= self.times[-1]:
            raise ValueError('Choose a time inside the recorded day')
        return max(0, bisect_right(self.times, at)-1)

    def candidates(self, owner, at):
        if owner not in self.indices:
            raise ValueError('Unknown car')
        index = self.frame_index(at)
        frame, progress = self.record['frames'][index], self.progress[index]
        announced = self.public_world(index)
        public_zones = {z['id']:z for z in announced['zones']}
        cars = dict(zip(self.owners, frame['cars']))
        found = []
        pending=[c for c in getattr(self,'commitments',[]) if
                 (c.get('status','pending') if isinstance(c,dict) else c.status)=='pending']
        if cars[owner][4] != 'moving':
            return self.envelope(index, [])
        for z in self.zones:
            if owner not in (z['a'],z['b']):
                continue
            peer = z['b'] if z['a'] == owner else z['a']
            if z['id'] not in public_zones or any(z['ranges'][o][1] >
                    announced['offsets'][o]+announced['routes'][o].length for o in (owner,peer)):
                continue
            a,b = cars[owner],cars[peer]
            if b[4] != 'moving':
                continue
            ad,bd = z['ranges'][owner][0]-progress[owner],z['ranges'][peer][0]-progress[peer]
            # An occupied passage cannot be sold retrospectively. A stopped
            # requester at the ordinary five-metre yield line remains eligible.
            if min(ad,bd) <= .2 or max(ad,bd) > HORIZON*SPEED:
                continue
            causing_delay=any(cars[o][5] in ('yielding','roundabout') and cars[o][6]==p
                              for o,p in ((owner,peer),(peer,owner)))
            if not causing_delay and not passage_overlap(z['ranges'][owner],z['ranges'][peer],progress[owner],progress[peer],a[3],b[3]):
                continue
            # Lane following cannot sell an overtake on this one-lane-per-way map.
            dx,dy = b[0]-a[0],b[1]-a[1]
            lateral = abs(-math.sin(a[2])*dx+math.cos(a[2])*dy)
            queued=peer in following_chain(cars,owner)[1:] or owner in following_chain(cars,peer)[1:]
            if queued and lateral < 2.6 and math.cos(a[2]-b[2]) > .85:
                continue
            r = self.geometry['roundabout']
            kind = 'roundabout entry' if math.hypot(z['x']-r['x'],z['y']-r['y']) < r['radius']+13 else 'junction passage'
            for seller,buyer in ((owner,peer),(peer,owner)):
                if self.reservation_conflict(seller,buyer,z['id'],pending):
                    continue
                announced_stop=announced['inputs']['vehicles'][self.indices[buyer]]['announced_stop_s']
                exit_s=z['ranges'][buyer][1]-announced['offsets'][buyer]
                if announced_stop is not None and announced_stop<=exit_s+1.5:
                    continue  # The announced parking stop cannot deliver this clearance.
                seller_distance=z['ranges'][seller][0]-progress[seller]
                if not can_hold_before_entry(seller_distance,cars[seller][3],cars[seller][8]):
                    continue
                found.append({'id':f"{z['id']}:{seller}:{buyer}", 'zone_id':z['id'],
                    'seller':seller,'buyer':buyer,'kind':kind,'point':[z['x'],z['y']],
                    'label':f'{seller} gives {buyer} the next passage', 'max_wait_seconds':MAX_WAIT,
                    'seconds_to_passage':round(min(ad,bd)/SPEED,2),
                    'distance_between_cars_m':round(math.dist(a[:2],b[:2]),2),
                    'hold_feasibility':'Observed motion plus existing longitudinal stop controller',
                    'seller_distance_to_entry_m':round(seller_distance,3),
                    'seller_speed_mps':cars[seller][3]})
        found.sort(key=lambda c:(c['seconds_to_passage'],c['zone_id'],c['seller']))
        return self.envelope(index, found)

    def resource_bounds(self, zone_id):
        cache=getattr(self,'_resource_bounds_cache',None)
        if cache is None:
            self._resource_bounds_cache=cache={}
        if zone_id not in cache:
            zone=next(z for z in self.zones if z['id']==zone_id)
            points=[]
            for owner in (zone['a'],zone['b']):
                route=self.routes[owner]
                start,end=zone['ranges'][owner]
                lo,hi=max(0.,start-5.),min(route.length,end+3.)
                points.extend(route.at(min(hi,lo+i))[:2] for i in range(math.ceil(hi-lo)+1))
            cache[zone_id]=(min(p[0] for p in points)-3.,min(p[1] for p in points)-3.,
                            max(p[0] for p in points)+3.,max(p[1] for p in points)+3.)
        return cache[zone_id]

    def reservation_conflict(self, seller, buyer, zone_id, pending=None):
        pending=pending if pending is not None else getattr(self,'commitments',[])
        actors={seller,buyer}
        footprint=None
        for contract in pending:
            value=contract if isinstance(contract,dict) else vars(contract)
            if value.get('status','pending')!='pending':
                continue
            if actors.intersection(value.get('participants',[value['seller'],value['buyer']])):
                return True
            footprint=self.resource_bounds(zone_id) if footprint is None else footprint
            other=self.resource_bounds(value['zone_id'])
            if not (footprint[2]<other[0] or other[2]<footprint[0] or
                    footprint[3]<other[1] or other[3]<footprint[1]):
                return True
        return False

    def envelope(self, index, candidates):
        return {'source':self.source,'snapshot':index,'time':self.times[index],
            'horizon_seconds':HORIZON,'candidates':candidates,
            'scope':'Candidate passage demand. No distance-radius gate, price or social-welfare filter.'}

    def hold_obstruction(self, index, candidate, frame=None):
        world=self.public_world(index)
        zone=next((z for z in world['zones'] if z['id']==candidate['zone_id']),None)
        if zone is None:
            return None
        a,b=candidate['seller'],candidate['buyer']
        key=(candidate['zone_id'],a,b)
        cache=world.setdefault('_hold_checks',{})
        if frame is None and key in cache:
            return cache[key]
        rows=dict(zip(self.owners,(frame or self.record['frames'][index])['cars']))
        if frame is None:
            progress={o:self.progress[index][o]-world['offsets'][o] for o in (a,b)}
        else:
            progress={o:nearest_on_path(world['routes'][o].points,*rows[o][:2])[0] for o in (a,b)}
        value=shared_hold_obstruction(world['routes'][a],world['routes'][b],progress[a],progress[b],
            zone['ranges'][a][0],zone['ranges'][b][1],rows[a][:3])
        if value:
            value={**value,'seller':a,'buyer':b}
        if frame is None:
            cache[key]=value
        return value

    def resolve(self, owner, at, candidate_id):
        listing = self.candidates(owner, at)
        candidate = next((c for c in listing['candidates'] if c['id'] == candidate_id), None)
        if candidate is None:
            raise ValueError('This passage is no longer available at the selected snapshot')
        return listing['snapshot'], candidate

    def public_world(self, index):
        if index in self._public:
            return self._public[index]
        frame = self.record['frames'][index]
        routes,meta,cars,offsets,inputs = {},{},{},{},[]
        for o,c in zip(self.owners,frame['cars']):
            s,original = self.progress[index][o],self.routes[o]
            if c[4] == 'arrived':
                h = c[2]
                points = [[c[0]-3*math.cos(h),c[1]-3*math.sin(h)],
                          [c[0]+3*math.cos(h),c[1]+3*math.sin(h)]]
                offset,stop = s-3.,3.
            else:
                offset,end = max(0.,s-5.),min(original.length,s+SPEED*HORIZON+12.)
                points = [list(original.at(offset)[:2])]
                points += [list(p) for p,ps in zip(original.points,original.ss) if offset < ps < end]
                points += [list(original.at(end)[:2])]
                stop = self.meta[o]['goal_s']-offset if self.meta[o]['goal_s'] <= end else None
            routes[o] = Route(points)
            meta[o] = {'goal_s':stop if stop is not None else routes[o].length+1000.}
            offsets[o] = offset
            cars[o] = dict(id=o,x=c[0],y=c[1],heading_rad=c[2],speed_mps=c[3],status=c[4],
                           steering_angle_rad=c[9],acceleration_mps2=c[8])
            inputs.append({'id':o,'pose':[c[0],c[1],c[2]],'speed_mps':c[3],
                'status':c[4],'announced_path':points,'announced_stop_s':stop})
        zones = []
        for z in self.zones:
            if any(cars[o]['status'] != 'moving' for o in (z['a'],z['b'])):
                continue
            ranges = {o:[v-offsets[o] for v in z['ranges'][o]] for o in (z['a'],z['b'])}
            if any(bounds[0] > routes[o].length or bounds[1] < 0 or bounds[1] > routes[o].length
                   for o,bounds in ranges.items()):
                continue
            zones.append({**z,'ranges':ranges})
        result = {'routes':routes,'meta':meta,'cars':cars,'zones':zones,'offsets':offsets,
            'inputs':{'time':frame['t'],'horizon_seconds':HORIZON,'vehicles':inputs,
                'assumptions':['Public poses and explicitly announced near-term paths/stops.',
                    'Announcements remain unchanged during this forecast.',
                    'No private deadlines, wallets, memories or unannounced route tails.',
                    'Destination effects beyond announcements remain unresolved.']}}
        self._public[index] = result
        if len(self._public) > 12:
            expired = next(iter(self._public))
            self._public.pop(expired)
            self._baseline_forecasts.pop(expired,None)
        return result

    def instruction(self, index, candidate, public=False):
        z = next(z for z in self.zones if z['id'] == candidate['zone_id'])
        a,b = candidate['seller'],candidate['buyer']
        offsets = self.public_world(index)['offsets'] if public else {a:0.,b:0.}
        return Concession(a,b,z['id'],z['ranges'][a][0]-offsets[a],
            z['ranges'][b][1]-offsets[b],self.times[index],self.times[index]+MAX_WAIT)

    def public_commitments(self, index):
        """Copy accepted absolute-route promises into this public snapshot."""
        offsets=self.public_world(index)['offsets']
        result=[]
        for original in getattr(self,'commitments',[]):
            value=Concession(**{k:v for k,v in original.items() if k in Concession.__dataclass_fields__}) if isinstance(original,dict) else deepcopy(original)
            if value.status!='pending':
                continue
            value.seller_entry-=offsets[value.seller]
            value.buyer_exit-=offsets[value.buyer]
            result.append(value)
        return result

    def forecast(self, index, candidate):
        started = time.perf_counter()
        world = self.public_world(index)
        if index not in self._baseline_forecasts:
            self._baseline_forecasts[index] = project(world,self.geometry['roundabout'],self.times[index],
                                                     existing=self.public_commitments(index))
        baseline = self._baseline_forecasts[index]
        alternative = project(world,self.geometry['roundabout'],self.times[index],self.instruction(index,candidate,True),
                              existing=self.public_commitments(index))
        z = next(z for z in world['zones'] if z['id'] == candidate['zone_id'])
        outcome = {}
        for o in (candidate['seller'],candidate['buyer']):
            target = z['ranges'][o][1]
            first = first_crossing(baseline['progress'],o,target)
            second = first_crossing(alternative['progress'],o,target)
            exact=first is not None and second is not None
            end=min(baseline['progress'][-1][0],alternative['progress'][-1][0])
            def end_progress(samples):
                i=max(0,bisect_right([v[0] for v in samples],end)-1)
                return samples[i][1][o]
            estimate=(first-second if exact else
                      (end_progress(alternative['progress'])-end_progress(baseline['progress']))/SPEED)
            outcome[o] = {'ordinary_clear':first,'concession_clear':second,
                'passage_seconds_saved':round(estimate,3),
                'estimate_method':'interpolated_crossing' if exact else 'common_horizon_progress',
                'estimate_confidence':'conditional' if exact else 'rough',
                'destination_seconds_saved':None,'destination_scope':'Unresolved beyond public-intent forecast.'}
        return {'outcomes':outcome,'horizon_seconds':HORIZON,'seconds':round(time.perf_counter()-started,4),
            'step_seconds':FORECAST_STEP,'method':'Projected lane motion using the same ordinary yielding policy; lateral tracking is approximated.',
            'concession_status':alternative['instruction_status'],'inputs':world['inputs'],
            'existing_commitments_preserved':all(s in ('pending','delivered') for s in alternative['existing_statuses']),
            'destination_scope':'Full-trip differences belong to the separate, full-state physical check.',
            'uncertainty':'Conditional forecast, not a calibrated probability or a safety proof.',
            '_branches':{name:{'frames':value['frames'],'arrivals':value['arrivals'],
                'termination':{'kind':'forecast_horizon'},'concession_status':value['instruction_status']}
                for name,value in [('ordinary',baseline),('concession',alternative)]}}

    def restored_world(self, index):
        if self._template is None:
            scenario = make_scenario(self.geometry,self.routes,self.meta)
            self._template = Simulation(scenario,record=False)
        sim = copy(self._template)
        sim.cars = deepcopy(self._template.cars)
        sim._time = self.times[index]
        sim.horizon = self.times[index]+400.
        sim.termination,sim.events,sim._step_events = None,[],[]
        sim._speeding = set()
        for o,c in zip(self.owners,self.record['frames'][index]['cars']):
            sim.cars[o].update(x=c[0],y=c[1],heading_rad=c[2],speed_mps=c[3],status=c[4],
                acceleration_mps2=c[8],steering_angle_rad=c[9])
            if c[4] == 'arrived':
                sim.cars[o]['arrival'] = self.record['arrivals'][o]
        driver = Driver(self.routes,self.meta,self.zones,self.geometry['roundabout'])
        driver.progress = dict(self.progress[index])
        return sim,driver

    def physical(self, index, candidate=None, cancel=None, progress=None):
        sim,driver = self.restored_world(index)
        instruction = self.instruction(index,candidate) if candidate else None
        frames,passages,ticks,stopped = [],[],0,None
        while not sim.termination:
            if cancel and cancel.is_set():
                sim._end('cancelled')
                break
            actions,states = driver.actions(sim,instruction)
            if ticks % 2 == 0:
                frames.append(physical_frame(sim,states,self.owners))
                passages.append((sim.time,dict(driver.progress)))
            if instruction and instruction.status not in ('pending','delivered'):
                sim._end('concession_fault',message=instruction.status)
                break
            sim.step(actions)
            ticks += 1
            if all(c['status'] == 'arrived' or c['speed_mps'] < .01 for c in sim.cars.values()):
                stopped = sim.time if stopped is None else stopped
                if not sim.termination and sim.time-stopped > 20.:
                    sim._end('stalled_traffic')
            else:
                stopped = None
            if progress and ticks % 50 == 0:
                progress(round(sim.time,1),sum(c['status']=='arrived' for c in sim.cars.values()))
        _,states = driver.actions(sim,instruction)
        frames.append(physical_frame(sim,states,self.owners))
        passages.append((sim.time,dict(driver.progress)))
        return {'frames':frames,'progress':passages,'termination':sim.termination,
            'arrivals':{o:c.get('arrival') for o,c in sim.cars.items()},
            'concession_status':instruction.status if instruction else None,
            'delivered_at':instruction.delivered_at if instruction else None}


def physical_frame(sim, states, owners):
    return {'t':round(sim.time,4),'cars':[[c['x'],c['y'],c['heading_rad'],c['speed_mps'],c['status'],
        *states[o],c['acceleration_mps2'],c['steering_angle_rad']] for o in owners for c in [sim.cars[o]]]}


def first_crossing(samples, owner, target):
    for i,(at,positions) in enumerate(samples):
        s = positions[owner]
        if s > target:
            if not i:
                return at
            previous,points = samples[i-1]
            return previous+(at-previous)*(target-points[owner])/max(1e-9,s-points[owner])
    return None


def project(public, roundabout, start, instruction=None, existing=None):
    class World:
        pass
    world = World()
    world.config,world.time,world.cars = VehicleConfig(),start,deepcopy(public['cars'])
    commitments=deepcopy(list(existing or []))
    controls=commitments+([instruction] if instruction is not None else [])
    driver = Driver(public['routes'],public['meta'],public['zones'],roundabout)
    driver.progress = {o:public['routes'][o].project(c['x'],c['y'],5.)[0] for o,c in world.cars.items()}
    samples,frames,arrivals = [],[],{}
    for _ in range(round(HORIZON/FORECAST_STEP)+1):
        actions,states = driver.actions(world,controls if commitments else instruction)
        samples.append((world.time,dict(driver.progress)))
        frames.append(physical_frame(world,states,list(public['cars'])))
        if (instruction and instruction.status not in ('pending','delivered')) or any(
                c.status not in ('pending','delivered') for c in commitments):
            break
        for o,a in actions.items():
            c = world.cars[o]
            acceleration = a.get('acceleration_mps2',1.5*(a.get('cruise_auto',{}).get('speed_mps',c['speed_mps'])-c['speed_mps']))
            acceleration = max(-6.,min(2.,acceleration))
            speed = max(0.,min(SPEED,c['speed_mps']+acceleration*FORECAST_STEP))
            s = driver.progress[o]+(speed+c['speed_mps'])*FORECAST_STEP/2
            route,goal = public['routes'][o],public['meta'][o]['goal_s']
            if goal < route.length and s >= goal-1.5 and speed < .1:
                c['status'],speed = 'arrived',0.
                arrivals.setdefault(o,world.time)
            x,y,h = route.at(min(s,route.length))
            c.update(x=x,y=y,heading_rad=h,speed_mps=speed,steering_angle_rad=a['steering_angle_rad'],acceleration_mps2=acceleration)
        world.time += FORECAST_STEP
    return {'progress':samples,'frames':frames,'arrivals':arrivals,
            'instruction_status':instruction.status if instruction else None,
            'existing_statuses':[c.status for c in commitments]}


class EncounterService:
    def __init__(self, context=None, directory=None):
        self._context = context
        self.directory = Path(directory or ROOT/'data/community/encounters')
        self.lock = threading.RLock()
        self.jobs = {}
        self.active = None

    @property
    def context(self):
        if self._context is None:
            self._context = Context()
        return self._context

    def candidates(self, owner, at):
        with self.lock:
            return self.context.candidates(owner,at)

    def preview(self, owner, at, candidate_id, source):
        with self.lock:
            c = self.context
            if source != c.source:
                raise ValueError('The recording changed; reopen the passage list')
            index,candidate = c.resolve(owner,at,candidate_id)
            key = digest([VERSION,c.source,c.runtime,index,candidate_id])
            if key not in self.jobs:
                if len(self.jobs) >= 12:
                    old = next((k for k in self.jobs if k != self.active),None)
                    if old:
                        self.jobs.pop(old)
                result = {'id':key,'source':c.source,'time':c.times[index],'snapshot':index,
                    'candidate':candidate,'forecast':c.forecast(index,candidate),'status':'predicted',
                    'model_calls':0,'scope':'Inspection only. Does not change the recorded day or create a live agreement.'}
                result['forecast_branches']=result['forecast'].pop('_branches')
                path = self.directory/(key+'.json.gz')
                if path.exists():
                    cached = json.loads(gzip.decompress(path.read_bytes()))
                    if cached.get('status') in ('complete','failed'):
                        result = cached
                self.jobs[key] = {'result':result,'cancel':threading.Event()}
            return self.summary(key)

    def summary(self, key):
        with self.lock:
            if key not in self.jobs:
                raise ValueError('Unknown comparison')
            return deepcopy({k:v for k,v in self.jobs[key]['result'].items() if k not in ('branches','forecast_branches')})

    def validate(self, key):
        with self.lock:
            job = self.jobs.get(key)
            if job is None:
                raise ValueError('Unknown comparison')
            if job['result']['status'] in ('running','complete','failed'):
                return self.summary(key)
            if self.active:
                raise ValueError('A physical comparison is already running; stop it first')
            job['cancel'].clear()
            job['result']['status'] = 'running'
            self.active = key
            threading.Thread(target=self._validate,args=(key,),daemon=True).start()
            return self.summary(key)

    def cancel(self, key):
        with self.lock:
            if key in self.jobs:
                self.jobs[key]['cancel'].set()
            return {'status':'cancelling'}

    def _validate(self, key):
        started = time.perf_counter()
        job = self.jobs[key]
        result,c = job['result'],self.context
        index,candidate = result['snapshot'],result['candidate']
        try:
            def report(stage):
                def update(at,arrived):
                    with self.lock:
                        result['progress'] = {'stage':stage,'time':at,'arrived':arrived,'cars':len(c.owners)}
                return update
            # Audit one full baseline against the retained physical recording.
            if c.baseline_audit is None:
                ordinary = c.physical(index,cancel=job['cancel'],progress=report('checking baseline'))
                if job['cancel'].is_set():
                    raise InterruptedError('Physical comparison stopped')
                error = max((abs(ordinary['arrivals'][o]-c.record['arrivals'][o])
                    for o in c.owners if ordinary['arrivals'][o] is not None),default=math.inf)
                c.baseline_audit = {'snapshot':index,'termination':ordinary['termination'],
                    'max_arrival_error_seconds':error if math.isfinite(error) else None,
                    'passed':ordinary['termination']['kind']=='completed' and error <= .1}
            else:
                ordinary = None
            if job['cancel'].is_set():
                raise InterruptedError('Physical comparison stopped')
            if not c.baseline_audit['passed']:
                raise ValueError('Baseline restoration did not reproduce the retained day; comparison withheld')
            if ordinary is None:
                ordinary = {'frames':c.record['frames'][index:],'arrivals':c.record['arrivals'],
                    'termination':c.record['termination'],'progress':[(t,p) for t,p in zip(c.times[index:],c.progress[index:])],
                    'concession_status':None,'delivered_at':None}
            alternative = c.physical(index,candidate,job['cancel'],report('checking concession'))
            if job['cancel'].is_set():
                raise InterruptedError('Physical comparison stopped')
            z = next(z for z in c.zones if z['id']==candidate['zone_id'])
            outcomes = {}
            valid = alternative['termination']['kind']=='completed' and alternative['concession_status']=='delivered'
            for o in c.owners:
                a,b = ordinary['arrivals'][o],alternative['arrivals'][o]
                first,second = (first_crossing(branch['progress'],o,z['ranges'][o][1])
                    for branch in (ordinary,alternative)) if o in z['ranges'] else (None,None)
                estimate = result['forecast']['outcomes'].get(o,{}).get('passage_seconds_saved')
                measured = first-second if first is not None and second is not None else None
                outcomes[o] = {'ordinary_arrival':a,'concession_arrival':b,
                    'destination_seconds_saved':round(a-b,3) if valid and a is not None and b is not None else None,
                    'passage_seconds_saved':round(measured,3) if valid and measured is not None else None,
                    'prediction_error_seconds':round(estimate-measured,3) if valid and estimate is not None and measured is not None else None}
            result.update(status='complete' if valid else 'failed',measured=outcomes,
                baseline_audit=c.baseline_audit,validation_seconds=round(time.perf_counter()-started,3),
                runtime_sha256=c.runtime,validation_scope='Full-state deterministic evaluator. Future scripted routes are known here; these outcomes are never forecast inputs.',
                concession_status=alternative['concession_status'],termination=alternative['termination'],
                branches={name:{k:v for k,v in branch.items() if k!='progress'} for name,branch in
                    [('ordinary',ordinary),('concession',alternative)]})
        except Exception as exc:
            result.update(status='cancelled' if isinstance(exc,InterruptedError) else 'failed',error=str(exc))
        finally:
            with self.lock:
                self.active = None
                write_record(self.directory/(key+'.json.gz'),result)

    def replay(self, key, predicted=False):
        result = self.jobs.get(key,{}).get('result',{})
        field='forecast_branches' if predicted else 'branches'
        if field not in result:
            raise ValueError('Physical comparison is not ready')
        return {'owners':self.context.owners,'branches':result[field],'candidate':result['candidate'],
            'source':result['source'],'time':result['time'],'kind':'forecast' if predicted else 'physical'}

"""Local traffic conflicts, rolling driving and physical delivery on the bicycle simulator.

East/north streets form a directed acyclic network, not a general urban driver.
Visible cars broadcast only their imminent movement and approach timestamp.
Geometry and those public signals govern rolling gap acceptance and following.
"""
from __future__ import annotations
from copy import deepcopy
from functools import lru_cache
import math

from .driving import Simulation, VehicleConfig, separation
from .driving_roads import nearest_on_path, path_length, point_at

SPEED = 5.
NEGOTIATION_RADIUS_M = 42.5

def _join(network, route):
    cache=getattr(network,'_joined_routes',None)
    if cache is None:cache=network._joined_routes={}
    key=tuple(route)
    if key in cache:return cache[key]
    points = []
    for pid in route:
        part = network.paths[pid]
        points.extend(part[1:] if points and math.dist(points[-1], part[0]) < .01 else part)
    cache[key]=points
    return points


def _movements(network):
    if not hasattr(network,'_junction_movements'):
        network._junction_movements={c['id']:c for c in network.connections if 'junction_id' in c}
    return network._junction_movements


def _own_intent(car,network):
    points=_join(network,car['path']);progress=nearest_on_path(points,car['x'],car['y'])[0];offset=0.
    movements=_movements(network)
    for pid in car['path']:
        length=path_length(network.paths[pid])
        if pid in movements and offset+length+3.>progress:
            return {'junction_id':movements[pid]['junction_id'],'movement':pid,
                    'entry_distance':offset-progress,'exit_distance':offset+length-progress}
        offset+=length
    return None



def observations(sim):
    """Lean local poses/signals. Public geometry is an explicit driver argument.

    The previous full engine map/road/config copy per car and tick was unused
    by the policy boundary. Announced movement geometry/priority is public;
    private destinations and unannounced future route segments stay private.
    """
    public=('id','x','y','heading_rad','speed_mps','steering_angle_rad','status')
    views={owner:{'time':sim.time,'self':deepcopy(car),
        'nearby':[{k:other[k] for k in public} for other in sim.cars.values()
                  if other['id']!=owner and math.hypot(other['x']-car['x'],other['y']-car['y'])<=sim.observation_radius]}
        for owner,car in sim.cars.items()}
    movements=_movements(sim.network)
    history=getattr(sim,'_network_approach_since',{});signals={}
    for owner,view in views.items():
        intent=_own_intent(view['self'],sim.network)
        if intent and intent['entry_distance']<=45. and view['self']['status']=='moving':
            key=(owner,intent['junction_id']);history.setdefault(key,sim.time)
            movement=movements[intent['movement']]
            signals[owner]={**intent,'since':history[key], 'priority':movement['priority'],
                            'points':deepcopy(sim.network.paths[intent['movement']])}
            view['intent']=signals[owner]
        else:view['intent']=None
    sim._network_approach_since=history
    for view in views.values():
        for peer in view['nearby']:peer['intent']=deepcopy(signals.get(peer['id']))
    return views


@lru_cache(maxsize=2048)
def _ranges(first,second):
    config=VehicleConfig();pa=[list(p) for p in first];pb=[list(p) for p in second]
    sa=[i*.75 for i in range(math.ceil(path_length(pa)/.75)+1)]
    sb=[i*.75 for i in range(math.ceil(path_length(pb)/.75)+1)]
    hits=[]
    for a in sa:
        x,y,h=point_at(pa,a);ca={'x':x,'y':y,'heading_rad':h}
        for b in sb:
            xx,yy,hh=point_at(pb,b)
            if math.hypot(x-xx,y-yy)>6.:continue
            if separation(ca,{'x':xx,'y':yy,'heading_rad':hh},config)<.7:hits.append((a,b))
    if not hits:return None
    return (min(a for a,b in hits)-1.,max(a for a,b in hits)+1.,
            min(b for a,b in hits)-1.,max(b for a,b in hits)+1.)


def conflict_ranges(network, first, second):
    movements=_movements(network)
    if first==second or first not in movements or second not in movements:return None
    a,b=movements[first],movements[second]
    if a['junction_id']!=b['junction_id'] or a['movement'][0]==b['movement'][0]:return None
    return _ranges(tuple(map(tuple,network.paths[first])),tuple(map(tuple,network.paths[second])))


def _crossing(first, second):
    """Different incoming approaches can exchange a physically conflicting slot.

    Actual swept-path overlap and before-entry checks are separate requirements.
    This includes conflicting turns/merges; following in one lane is not a trade.
    """
    a,b=first.rsplit('_',1)[-1],second.rsplit('_',1)[-1]
    return first!=second and a[0]!=b[0]


def _earliest_exit(distance, speed):
    """Public lower bound: immediate maximum acceleration up to the road cap."""
    acceleration=VehicleConfig().max_acceleration;speed=min(SPEED,max(0.,speed))
    distance=max(0.,distance);ramp_distance=(SPEED*SPEED-speed*speed)/(2*acceleration)
    if distance<=ramp_distance:return (math.sqrt(speed*speed+2*acceleration*distance)-speed)/acceleration
    return (SPEED-speed)/acceleration+(distance-ramp_distance)/SPEED


def _members(observation):
    car=observation['self'];intent=observation.get('intent')
    if not intent:return []
    members=[{**{k:car[k] for k in ('id','x','y','heading_rad','speed_mps','status')},'intent':intent}]
    members += [p for p in observation['nearby'] if p.get('intent') and p['intent']['junction_id']==intent['junction_id']]
    return sorted(members,key=lambda p:(p['intent']['since'],-p['intent']['priority'],p['id']))


def _order(members,commitments,time):
    order=[p['id'] for p in members]
    for c in commitments:
        if members and c.get('junction_id')!=members[0]['intent']['junction_id']:continue
        if c.get('status','running') not in ('running','active','accepted') or not c.get('accepted_at',0)<=time<c['expires_at']:continue
        if c['seller'] in order and c['buyer'] in order:
            seller,buyer=order.index(c['seller']),order.index(c['buyer'])
            movements={p['id']:p['intent']['movement'] for p in members}
            if buyer>seller and _crossing(movements[c['seller']],movements[c['buyer']]):
                order[seller],order[buyer]=order[buyer],order[seller]
    return order


def driver_actions(views,network,commitments=(),diagnostics=None):
    actions={}
    for owner,view in views.items():
        car=view['self']
        if car['status']!='moving':continue
        points=_join(network,car['path']);progress=nearest_on_path(points,car['x'],car['y'])[0]
        remaining=nearest_on_path(points,*car['goal'])[0]-progress
        speed=min(SPEED,math.sqrt(2*1.2*max(0.,remaining-1.5)));reason='free_driving';binding=None
        # Ordinary lane following remains active even without negotiation/signals.
        for peer in view['nearby']:
            ps,_,distance=nearest_on_path(points,peer['x'],peer['y']);gap=ps-progress-4.5
            aligned=math.cos(peer['heading_rad']-car['heading_rad'])>.65
            if aligned and distance<2.8 and gap>-.5:
                limit=max(0.,peer['speed_mps']+.7*(gap-5.))
                if limit<speed:speed=limit;reason='following_vehicle';binding=peer['id']
        members=_members(view);order=_order(members,commitments,view['time'])
        own=view.get('intent')
        if own:
            for peer in members:
                if peer['id']==owner:continue
                ranges=conflict_ranges(network,own['movement'],peer['intent']['movement'])
                if not ranges:continue
                ae,ax,be,bx=ranges;ad=own['entry_distance'];bd=peer['intent']['entry_distance']
                if own['movement'][-1]==peer['intent']['movement'][-1]:
                    # Virtual longitudinal positions meet at the common exit.
                    # Accept an existing 9.5m center gap irrespective of FIFO;
                    # otherwise establish it while rolling toward that exit.
                    od,pd=own['exit_distance'],peer['intent']['exit_distance']
                    if od/max(car['speed_mps'],.1)+9.5/SPEED<_earliest_exit(pd,peer['speed_mps']):continue
                    # Keep driving past a near-stationary upstream peer while
                    # the possible conflict remains beyond comfortable braking
                    # plus two seconds of observation. A moving merge leader
                    # already needs its rolling gap established farther away.
                    # Recheck each frame; a distant stopped car cannot reserve
                    # the exit through an infinite constant-speed ETA.
                    approach=max(12.,car['speed_mps']**2/(2*2.)+2*car['speed_mps'])
                    if peer['speed_mps']<.5 and ad+ae>approach and bd+be>0.:continue
                    agreed=next((c for c in commitments if {owner,peer['id']}=={c['seller'],c['buyer']}
                                 and c.get('junction_id')==own['junction_id'] and c.get('status','running')=='running'),None)
                    if agreed and -ad<ae and -bd<be:
                        leader=owner==agreed['seller']
                    elif od+9.5<=pd or pd+9.5<=od:leader=(pd<od)
                    elif -ad>=ae or -bd>=be:
                        leader=(pd<od) if (-ad>=ae and -bd>=be) else -bd>=be
                    else:leader=order.index(peer['id'])<order.index(owner)
                    if leader and od>0.:
                        wait=pd/max(peer['speed_mps'],.1)+9.5/SPEED
                        target=od/max(.2,wait) if wait>0. else SPEED
                        if target<speed:speed=target;reason='merging_headway';binding=peer['id']
                    continue
                if -ad>ax or -bd>bx:continue
                at=(ad+ae)/max(car['speed_mps'],.1);bt=(bd+be)/max(peer['speed_mps'],.1)
                own_clear=(ad+ax)/max(car['speed_mps'],.1);peer_clear=(bd+bx)/max(peer['speed_mps'],.1)
                if own_clear+.5<bt or peer_clear+.5<at:continue
                entered_peer=-bd>=be;entered_own=-ad>=ae
                yields=entered_peer or (not entered_own and order.index(peer['id'])<order.index(owner))
                if yields:
                    distance=max(0.,ad+ae)
                    target=distance/max(.2,peer_clear+.5)
                    # Do not creep into an occupied crossing when the leader stops.
                    target=min(target,math.sqrt(2*2.*max(0.,distance-1.)))
                    if target<speed:speed=target;reason='conflicting_trajectory';binding=peer['id']
        request={'steering_auto':{'path':car['path']},'cruise_auto':{'speed_mps':speed}}
        if remaining<3. and car['speed_mps']<.3:
            request.pop('cruise_auto');request['acceleration_mps2']=-3.
        actions[owner]=request
        if diagnostics is not None:diagnostics[owner]={'constraint':reason,'peer':binding,'junction_id':own['junction_id'] if own else None,'target_speed_mps':speed}
    return actions


def get_opportunities(observation,network):
    members=_members(observation)
    if len(members)<2:return []
    own=observation['self']['id'];result=[]
    for i,seller in enumerate(members[:-1]):
      for buyer in members[i+1:]:
        if own not in (seller['id'],buyer['id']):continue
        a,b=seller['intent'],buyer['intent'];ranges=conflict_ranges(network,a['movement'],b['movement'])
        if not ranges or not _crossing(a['movement'],b['movement']):continue
        ae,ax,be,bx=ranges
        if min(a['entry_distance']+ae,b['entry_distance']+be)<8.:continue
        st=(a['entry_distance']+ae)/max(seller['speed_mps'],.1);bt=(b['entry_distance']+be)/max(buyer['speed_mps'],.1)
        sc=(a['entry_distance']+ax)/max(seller['speed_mps'],.1);bc=(b['entry_distance']+bx)/max(buyer['speed_mps'],.1)
        gain=max(0.,sc+.5-bt);delay=max(0.,bc+.5-st)
        if min(gain,delay)<.1:continue
        # Any earlier conflicting third party makes this simple pair estimate
        # insufficient: decline to advertise an unproved buyer opportunity.
        result.append({'encounter_id':f"{a['junction_id']}:{seller['id']}:{buyer['id']}",
            'junction_id':a['junction_id'],'seller':seller['id'],'buyer':buyer['id'],
            'own_eta':[max(st,0.) if own==seller['id'] else max(bt,sc+.5),
                       max(st,bc+.5) if own==seller['id'] else max(bt,0.)],
            'seller_delay':delay,'buyer_gain':gain,'seller_movement':a['movement'],'buyer_movement':b['movement'],
            'conflict_ranges':list(ranges),'estimate_scope':'local constant-speed public maneuver estimate; not a counterfactual oracle'})
    return result


conflict_opportunities=get_opportunities


def _project_extended(points,pose):
    s=nearest_on_path(points,pose['x'],pose['y'])[0];length=path_length(points)
    if s>=length-1e-6:
        x,y,h=point_at(points,length);s+=max(0.,(pose['x']-x)*math.cos(h)+(pose['y']-y)*math.sin(h))
    return s


def commitment_progress(views,commitment,memory=None):
    memory={} if memory is None else memory
    seller,buyer=commitment['seller'],commitment['buyer'];view=views[seller];time=view['time']
    poses={seller:view['self'],**{p['id']:p for p in view['nearby']}}
    result={'status':'running','time':time,'hold_seconds':memory.get('hold_seconds',0.),'yield_observed':bool(memory.get('held'))}
    if time>=commitment['expires_at']:result['status']='expired';return result
    if time<commitment.get('accepted_at',0.):return result
    if buyer not in poses:result['status']='blocked';return result
    # Capture announced public paths; derive proof geometry independently of
    # any descriptive ranges included by the commitment's caller.
    if 'seller_path' not in memory:
        for role in ('seller','buyer'):
            owner=commitment[role];intent=view.get('intent') if owner==seller else poses[owner].get('intent')
            if not intent or intent['junction_id']!=commitment['junction_id']:result['status']='blocked';return result
            memory[role+'_movement']=intent['movement'];memory[role+'_path']=intent['points']
        if not _crossing(memory['seller_movement'],memory['buyer_movement']):
            result['status']='blocked';return result
        ranges=_ranges(tuple(map(tuple,memory['seller_path'])),tuple(map(tuple,memory['buyer_path'])))
        if not ranges:result['status']='blocked';return result
        memory['ranges']=ranges
    ae,ax,be,bx=memory['ranges'];sp=_project_extended(memory['seller_path'],poses[seller]);bp=_project_extended(memory['buyer_path'],poses[buyer])
    if 'initial_order_valid' not in memory:
        order=[p['id'] for p in _members(view)]
        memory['initial_order_valid']=(seller in order and buyer in order and order.index(seller)<order.index(buyer)
            and sp<ae and bp<be)
    if not memory['initial_order_valid']:result['status']='blocked';return result
    fields=('id','x','y','heading_rad','speed_mps')
    witness=lambda:{'time':time,'seller':{k:poses[seller][k] for k in fields},'buyer':{k:poses[buyer][k] for k in fields}}
    merging=math.dist(memory['seller_path'][-1],memory['buyer_path'][-1])<.05
    if merging:
        # A merge promises order through the common exit, not exclusive use
        # of an entire shared lane. Ordinary safe following remains possible.
        sr=path_length(memory['seller_path'])-sp
        br=path_length(memory['buyer_path'])-bp
        if sr<=0 and br>=0:result['status']='blocked'
        elif bp>=be and sr>br:memory.setdefault('held',witness())
        if memory.get('held') and br < -VehicleConfig().length/2-.5 and sr>br and result['status']=='running':
            memory.setdefault('delivered',witness());result['status']='delivered'
    else:
        if sp>=ae and bp<=bx:result['status']='blocked'
        elif bp>=be and sp<ae:memory.setdefault('held',witness())
        if memory.get('held') and bp>bx and result['status']=='running':
            memory.setdefault('delivered',witness());result['status']='delivered'
    memory['hold_seconds']=memory.get('hold_seconds',0.)+max(0.,time-memory.get('time',time))*(1-min(1.,poses[seller]['speed_mps']/SPEED));memory['time']=time
    result.update(hold_seconds=memory['hold_seconds'],yield_observed=bool(memory.get('held')),
        evidence={'junction_id':commitment['junction_id'],'delivery_kind':'merge_exit_order' if merging else 'exclusive_crossing',
                  'bounds':commitment.get('bounds'),
                  'movement_ids':{'seller':memory.get('seller_movement'),'buyer':memory.get('buyer_movement')},
                  'conflict_ranges':{'seller':list(memory['ranges'][:2]),'buyer':list(memory['ranges'][2:])},'held':memory.get('held'),'delivered':memory.get('delivered')})
    return result

"""Live requests and independent passage contracts on separated crossings."""
from concurrent.futures import Future
from copy import deepcopy
import gzip
import hashlib
import json

import pytest

from vehicle_days.driving import Simulation
from vehicle_days.encounters import Context, physical_frame
from vehicle_days.interactive import InteractiveDemo
from vehicle_days.layout_traffic import Driver, Route, conflicts, make_scenario


class ImmediateExecutor:
    def submit(self, fn, *args):
        result=Future()
        try:result.set_result(fn(*args))
        except Exception as exc:result.set_exception(exc)
        return result


def context_for_crossings(tmp_path,spacing=120.):
    nodes={};roads=[];routes={};meta={}
    for suffix,x,owners in [('one',0.,('A','B')),('two',spacing,('C','D'))]:
        for name,dx,dy in [('w',-40.,0.),('e',40.,0.),('s',0.,-40.),('n',0.,40.)]:
            nodes[suffix+name]={'x':x+dx,'y':dy}
        roads.extend([{'id':suffix+'h','a':suffix+'w','b':suffix+'e'},
                      {'id':suffix+'v','a':suffix+'s','b':suffix+'n'}])
        routes[owners[0]]=Route([[x-40.,0.],[x+40.,0.]])
        routes[owners[1]]=Route([[x,-40.],[x,40.]])
        meta[owners[0]]={'goal':[x+35.,0.],'goal_s':75.}
        meta[owners[1]]={'goal':[x,35.],'goal_s':75.}
    geometry={'nodes':nodes,'roads':roads,'roundabout':{'x':1000.,'y':1000.,'radius':30.}}
    zones=conflicts(routes)
    sim=Simulation(make_scenario(geometry,routes,meta,horizon=80.),record=False)
    driver=Driver(routes,meta,zones,geometry['roundabout'])
    _,states=driver.actions(sim)
    path=tmp_path/'map.json';path.write_text(json.dumps(geometry))
    record={'geometry_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'negotiations':0,
        'owners':list(routes),'routes':{o:{'points':r.points,**meta[o]} for o,r in routes.items()},
        'conflicts':zones,'frames':[physical_frame(sim,states,list(routes))],'arrivals':{},'termination':None}
    source=tmp_path/'reference.json.gz';source.write_bytes(gzip.compress(json.dumps(record).encode()))
    return Context(source,path)


@pytest.fixture
def separated(tmp_path):
    return context_for_crossings(tmp_path)


def demo(context):
    return InteractiveDemo(context,ImmediateExecutor(),workers=1)


def command(d,action,**fields):
    return d.command({'action':action,'revision':d.revision,**fields})


def test_two_independent_live_agreements_remain_binding_and_settle_once(separated):
    d=demo(separated)
    first=command(d,'options',car='A',at=0.)['candidates'][0]
    command(d,'apply',car='A',at=0.,candidate=first['id'])
    second_packet=command(d,'options',car='C',at=.2)
    assert second_packet['active_contracts'] and second_packet['candidates']
    second=next(c for c in second_packet['candidates'] if c['seller']=='C' and c['buyer']=='D')
    accepted=command(d,'apply',car='C',at=.2,candidate=second['id'])
    assert len(accepted['active_contracts'])==2
    assert {(c.seller,c.buyer) for c in d.instructions}=={('A','B'),('C','D')}
    command(d,'advance',to=10.,after=0.)
    command(d,'advance',to=16.,after=10.)
    assert not d.failure
    assert [c.status for c in d.instructions]==['delivered','delivered']
    assert len([e for e in d.events if e['kind']=='accepted'])==2
    delivered=[e for e in d.events if e['kind']=='delivered']
    assert len(delivered)==2 and len({e['id'] for e in delivered})==2
    command(d,'advance',to=20.,after=16.)
    assert len([e for e in d.events if e['kind']=='delivered'])==2


def test_npc_request_is_created_without_selecting_or_commanding_the_requester(separated):
    d=demo(separated)
    command(d,'advance',to=4.,after=0.)
    packet=command(d,'options',car='A',at=4.)
    requests=packet['incoming_requests']
    assert requests
    request=next(r for r in requests if r['requester']=='B')
    assert request['recipient']=='A' and request['origin']=='heuristic_delay_request'
    assert request['created_at']<=4. and request['actionable']
    assert any(e['kind']=='request_created' and e['requester']=='D' and e['recipient']=='C' for e in d.events)
    assert d.selected=='A'
    # Observing or receiving a request never accepts it on the driver's behalf.
    assert d.instructions==[]
    assert not any(e['kind']=='accepted' for e in d.events)


def test_declining_request_preserves_ordinary_motion_and_suppresses_repeated_exchange(separated):
    d=demo(separated)
    command(d,'advance',to=4.,after=0.)
    packet=command(d,'options',car='A',at=4.)
    request=next(r for r in packet['incoming_requests'] if r['requester']=='B')
    before=deepcopy((d.world.cars,d.frames,d.revision,d.instructions))
    reply=command(d,'decline',car='A',at=4.,request_id=request['request_id'])
    assert reply['status']=='declined'
    assert before==(d.world.cars,d.frames,d.revision,d.instructions)
    event_count=len(d.events)
    assert command(d,'decline',car='A',at=4.,request_id=request['request_id'])==reply
    assert len(d.events)==event_count
    again=command(d,'options',car='A',at=4.)
    assert all(r['candidate_id']!=request['candidate_id'] for r in again['incoming_requests'])
    assert all(c['id']!=request['candidate_id'] for c in again['candidates'])
    command(d,'advance',to=5.,after=4.)
    assert d.world.time==pytest.approx(5.) and not d.instructions


def test_incoming_request_id_can_be_accepted_as_the_actual_contract(separated):
    d=demo(separated)
    command(d,'advance',to=4.,after=0.)
    request=command(d,'options',car='A',at=4.)['incoming_requests'][0]
    result=command(d,'apply',car='A',at=4.,request_id=request['request_id'])
    assert result['accepted']['id']==request['candidate_id']
    assert d.instructions[0].seller=='A'
    assert any(r['status']=='accepted' for r in d.requests.values())


def test_conflicting_participants_rejected_before_state_or_promises_change(separated):
    d=demo(separated)
    first=command(d,'options',car='A',at=0.)['candidates'][0]
    command(d,'apply',car='A',at=0.,candidate=first['id'])
    before=deepcopy((d.world.cars,d.frames,d.events,d.instructions,d.revision))
    reason=d._conflict_reason(first,0)
    assert 'participant' in reason
    with pytest.raises(ValueError):
        command(d,'apply',car='A',at=0.,candidate=first['id'])
    assert before==(d.world.cars,d.frames,d.events,d.instructions,d.revision)


def test_disjoint_cars_cannot_reserve_overlapping_junction_space(tmp_path):
    context=context_for_crossings(tmp_path,spacing=22.)
    d=demo(context)
    first=next(c for c in command(d,'options',car='A',at=0.)['candidates'] if c['seller']=='A' and c['buyer']=='B')
    command(d,'apply',car='A',at=0.,candidate=first['id'])
    current=__import__('vehicle_days.interactive',fromlist=['snapshot_context']).snapshot_context(d._payload(0))
    second=next(c for c in context.candidates('C',0.)['candidates'] if c['seller']=='C' and c['buyer']=='D')
    assert all(c['id']!=second['id'] for c in current.candidates('C',0.)['candidates'])
    assert 'overlapping road space' in d._conflict_reason(second,0)


def test_wait_chain_rejects_known_deadlock_before_a_promise(separated):
    d=demo(separated)
    choice=command(d,'options',car='A',at=0.)['candidates'][0]
    # B cannot pass because it follows C, and C is waiting for A. Holding A
    # for B would close a real causal cycle even though A/B approach separately.
    index=0
    rows=d.frames[index]['cars']
    rows[separated.indices['B']][5:7]=['following','C']
    rows[separated.indices['C']][5:7]=['yielding','A']
    assert 'circular wait' in d._conflict_reason(choice,index)
    assert not d.instructions


def test_checked_rejection_retires_request_without_repeating_and_can_reopen(separated):
    d=demo(separated)
    command(d,'advance',to=4.,after=0.)
    original=command(d,'options',car='A',at=4.)['incoming_requests'][0]
    index=d._index(4.)
    key=(d.revision,'A',index)
    checked=deepcopy(d.ready[key])
    checked['candidates']=[c for c in checked['candidates'] if c['id']!=original['candidate_id']]
    checked['rejected'].append({'id':original['candidate_id'],'reason':'blocking_cycle'})
    d.ready[key]=checked
    packet=command(d,'options',car='A',at=4.)
    assert all(r['candidate_id']!=original['candidate_id'] for r in packet['incoming_requests'])
    request=next(r for r in d.requests.values() if r['request_id']==original['request_id'])
    assert request['status']=='unavailable'
    count=len(d.events)
    command(d,'options',car='A',at=4.)
    assert len(d.events)==count
    # At a later state the actual detailed forecaster again finds a deliverable
    # passage. The same request can become actionable without a new NPC spam.
    command(d,'advance',to=4.4,after=4.)
    reopened=command(d,'options',car='A',at=4.4)
    assert any(r['request_id']==original['request_id'] and r['actionable']
               for r in reopened['incoming_requests'])
    assert any(e['kind']=='request_reopened' for e in d.events)


def test_newer_checked_rejection_blocks_stale_apply_without_mutation(separated):
    d=demo(separated)
    command(d,'advance',to=4.,after=0.)
    old=command(d,'options',car='A',at=4.)['candidates'][0]
    command(d,'advance',to=4.2,after=4.)
    command(d,'options',car='A',at=4.2)
    key=(d.revision,'A',d._index(4.2))
    checked=deepcopy(d.ready[key])
    checked['candidates']=[c for c in checked['candidates'] if c['id']!=old['id']]
    checked['rejected'].append({'id':old['id'],'reason':'blocking_cycle'})
    d.ready[key]=checked
    before=deepcopy((d.world.cars,d.frames,d.events,d.requests,d.instructions,d.revision))
    with pytest.raises(ValueError,match='newer forecast'):
        command(d,'apply',car='A',at=4.2,candidate=old['id'])
    assert before==(d.world.cars,d.frames,d.events,d.requests,d.instructions,d.revision)


class PendingExecutor:
    def submit(self, *args):
        return Future()


def test_valid_older_estimate_stays_available_while_refresh_runs(separated):
    d=demo(separated)
    original=command(d,'options',car='A',at=0.)['candidates'][0]
    d.executor=PendingExecutor()
    command(d,'advance',to=4.,after=0.)
    packet=command(d,'options',car='A',at=4.)
    assert packet['status']=='ready' and packet['refreshing']
    assert packet['estimate_age_seconds']==4.
    assert any(c['id']==original['id'] for c in packet['candidates'])
    assert packet['stats']['forecast_mean_job_seconds']>0
    # Reusing the estimate never reuses its old pose: the contract begins at
    # today's observed control tick and goes through current feasibility checks.
    result=command(d,'apply',car='A',at=4.,candidate=original['id'])
    assert result['at']==4. and d.instructions[0].start==4.


def test_estimates_expire_after_ten_simulated_seconds(separated):
    d=demo(separated)
    command(d,'options',car='A',at=0.)
    d.executor=PendingExecutor()
    command(d,'advance',to=8.,after=0.)
    command(d,'advance',to=11.,after=8.)
    packet=command(d,'options',car='A',at=11.)
    assert packet['status']=='pending' and not packet.get('candidates')


def test_prefetch_spreads_slow_jobs_across_sixteen_second_window(separated):
    d=InteractiveDemo(separated,PendingExecutor(),workers=4)
    d.forecast_seconds=[4.]
    command(d,'advance',to=8.,after=0.)
    command(d,'advance',to=16.,after=8.)
    command(d,'options',car='A',at=0.,pace=4.)
    prepared=sorted(d.frames[key[2]]['t'] for key in d.jobs)
    assert len(prepared)==4
    assert prepared[-1]>10. and prepared[-1]<=16.
    assert all(b-a>=5.1 for a,b in zip(prepared,prepared[1:]))

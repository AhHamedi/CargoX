"""Independent behavioral invariants for live avoidable-delay negotiation."""
from copy import deepcopy
from dataclasses import asdict
import math

import pytest

from test_encounters import context
from test_interactive_review import ImmediateExecutor
from vehicle_days.driving import Simulation
from vehicle_days.encounters import EncounterService, physical_frame
from vehicle_days.interactive import InteractiveDemo, snapshot_context
from vehicle_days.layout_traffic import Driver, Route, conflicts, make_scenario
from vehicle_days.reasoning import TrafficReasoner, passage_evidence


def at_pose(context, owner, progress, speed=0.):
    """Set one observed pose without changing any future recording rows."""
    i=context.indices[owner]
    x,y,h=context.routes[owner].at(progress)
    row=context.record['frames'][0]['cars'][i]
    row[:5]=[x,y,h,speed,'moving']
    context.progress[0][owner]=progress
    context._public.clear();context._baseline_forecasts.clear()


def command(demo, action, **fields):
    return demo.command({'action':action, 'revision':demo.revision, **fields})


def candidates(demo, owner, at):
    result=command(demo,'options',car=owner,at=at)
    assert result['status']=='ready'
    return result


def instructions(demo):
    return demo.instructions


def four_car_district(context):
    """Separated crossings on one connected road network, not just actor IDs."""
    context.geometry['nodes'].update(w2={'x':160.,'y':0.},e2={'x':240.,'y':0.},
                                      s2={'x':200.,'y':-40.},n2={'x':200.,'y':40.})
    context.geometry['roads'] += [{'id':'h2','a':'w2','b':'e2'}, {'id':'v2','a':'s2','b':'n2'},
                                  {'id':'link','a':'e','b':'w2'}]
    context.routes.update(C=Route([[160.,0.],[240.,0.]]),D=Route([[200.,-40.],[200.,40.]]))
    context.meta.update(C={'points':context.routes['C'].points,'goal':[235.,0.],'goal_s':75.},
                        D={'points':context.routes['D'].points,'goal':[200.,35.],'goal_s':75.})
    context.owners=['A','B','C','D'];context.indices={o:i for i,o in enumerate(context.owners)}
    context.zones=conflicts(context.routes)
    sim=Simulation(make_scenario(context.geometry,context.routes,context.meta,horizon=60.),record=False)
    driver=Driver(context.routes,context.meta,context.zones,context.geometry['roundabout'])
    _,states=driver.actions(sim)
    context.record['frames']=[physical_frame(sim,states,context.owners)]
    context.times=[0.];context.progress=[dict(driver.progress)]
    context._template=None;context._public.clear();context._baseline_forecasts.clear()
    return context


def test_stopped_requester_at_normal_five_metre_line_can_request_passage(context):
    zone=context.zones[0]
    at_pose(context,'A',5.,2.)
    at_pose(context,'B',zone['ranges']['B'][0]-5.,0.)
    offered=context.candidates('B',0.)['candidates']
    assert any(c['seller']=='A' and c['buyer']=='B' for c in offered)


def test_fast_seller_without_braking_clearance_cannot_promise_passage(context):
    zone=context.zones[0]
    at_pose(context,'A',zone['ranges']['A'][0]-2.,5.)
    at_pose(context,'B',zone['ranges']['B'][0]-5.,0.)
    offered=context.candidates('B',0.)['candidates']
    assert not any(c['seller']=='A' and c['buyer']=='B' for c in offered)


def test_car_already_occupying_crossing_cannot_sell_a_reversed_order(context):
    zone=context.zones[0]
    at_pose(context,'A',zone['ranges']['A'][0]+1.,0.)
    at_pose(context,'B',zone['ranges']['B'][0]-5.,0.)
    offered=context.candidates('B',0.)['candidates']
    assert not any(c['seller']=='A' and c['buyer']=='B' for c in offered)


def test_two_independent_real_agreements_coexist_and_deliver_once(context):
    demo=InteractiveDemo(four_car_district(context),ImmediateExecutor(),workers=1)
    first=next(c for c in candidates(demo,'A',0.)['candidates']
               if c['seller']=='A' and c['buyer']=='B')
    command(demo,'apply',car='A',at=0.,candidate=first['id'])
    now=demo.world.time
    second=next(c for c in candidates(demo,'C',now)['candidates']
                if c['seller']=='C' and c['buyer']=='D')
    first_before=deepcopy(asdict(next(c for c in instructions(demo) if c.seller=='A')))
    accepted=command(demo,'apply',car='C',at=now,candidate=second['id'])
    assert len([c for c in instructions(demo) if c.status=='pending'])==2
    first_after=asdict(next(c for c in instructions(demo) if c.seller=='A'))
    assert first_after==first_before
    assert len([e for e in accepted['events'] if e['kind']=='accepted'])==2
    for until in (12.,20.,28.):
        command(demo,'advance',to=until,after=-1.)
    delivered=[e for e in demo.events if e['kind']=='delivered']
    assert {(e['seller'],e['buyer']) for e in delivered}=={('A','B'),('C','D')}
    assert len(delivered)==2
    assert not demo.failure


def test_new_forecast_baseline_and_alternative_both_honor_old_promise(context):
    demo=InteractiveDemo(four_car_district(context),ImmediateExecutor(),workers=1)
    first=next(c for c in candidates(demo,'A',0.)['candidates'] if c['seller']=='A')
    command(demo,'apply',car='A',at=0.,candidate=first['id'])
    original=deepcopy([asdict(c) for c in instructions(demo)])
    payload=demo._payload(len(demo.frames)-1)
    observed=snapshot_context(payload)
    assessed=TrafficReasoner(EncounterService(context=observed)).assess('C',payload['frame']['t'])
    second=next(c for c in assessed['candidates'] if c['seller']=='C' and c['buyer']=='D')
    future=observed.forecast(0,second)
    for name,branch in future['_branches'].items():
        proof=passage_evidence(observed,branch['frames'],first,0)
        assert proof['order_observed'], name+' dropped the first accepted passage order'
    assert original==[asdict(c) for c in instructions(demo)]


def test_npc_request_is_initiated_for_receiver_and_can_be_declined(context):
    demo=InteractiveDemo(context,ImmediateExecutor(),workers=1)
    command(demo,'advance',to=4.,after=0.)
    poses=deepcopy(demo.world.cars)
    time=demo.world.time
    packet=candidates(demo,'A',time)
    incoming=next(r for r in packet['incoming_requests']
                  if r['requester']=='B' and r['recipient']=='A')
    assert incoming['origin']=='heuristic_delay_request'
    assert incoming['created_at']<=time
    assert demo.world.cars==poses and demo.world.time==time
    assert not instructions(demo)
    assert not [e for e in demo.events if e['kind']=='accepted']
    identity=incoming['request_id']
    count=len([e for e in demo.events if e['kind']=='request_created' and e['request_id']==identity])
    again=candidates(demo,'A',time)
    assert any(r['request_id']==identity for r in again['incoming_requests'])
    assert len([e for e in demo.events if e['kind']=='request_created' and e['request_id']==identity])==count==1
    declined=command(demo,'decline',car='A',at=time,request_id=identity)
    assert declined['status']=='declined'
    assert not any(r['request_id']==identity for r in candidates(demo,'A',time)['incoming_requests'])
    assert demo.world.cars==poses and demo.world.time==time
    assert not instructions(demo)


def test_prefetched_future_snapshot_cannot_send_a_request_early(context):
    from vehicle_days.interactive import assess_snapshot
    demo=InteractiveDemo(context,ImmediateExecutor(),workers=1)
    command(demo,'advance',to=4.,after=0.)
    future_index=len(demo.frames)-1
    descriptor=assess_snapshot(demo._payload(future_index),'A')
    # Caching a future observation is permitted; publishing its intents is not.
    with demo.lock:
        demo._publish_predicted_requests(descriptor,0)
    assert not demo.requests
    assert not [e for e in demo.events if e['kind']=='request_created']


def test_declined_request_survives_an_unrelated_accepted_deal(context):
    demo=InteractiveDemo(four_car_district(context),ImmediateExecutor(),workers=1)
    command(demo,'advance',to=4.,after=0.)
    first=candidates(demo,'A',4.)
    request=next(r for r in first['incoming_requests'] if r['requester']=='B')
    command(demo,'decline',car='A',at=4.,request_id=request['request_id'])
    other=next(c for c in candidates(demo,'C',4.)['candidates'] if c['seller']=='C' and c['buyer']=='D')
    command(demo,'apply',car='C',at=4.,candidate=other['id'])
    # Rendering remains at the observed decision time while new continuation is buffered.
    later=candidates(demo,'A',4.)
    assert not any(r['candidate_id']==request['candidate_id'] for r in later['incoming_requests'])
    assert any(r['request_id']==request['request_id'] and r['status']=='declined' for r in demo.requests.values())


def test_same_order_gap_reservation_changes_early_braking_without_fake_overtake(context):
    """A faster follower later donates priority; a promise avoids early braking.

    B initially yields to A. Once 00 reaches B's queue, priority inheritance
    would make B pass first anyway. A binding early-gap agreement retains that
    final order while changing both drivers' early controls.
    """
    context.geometry['nodes']['w']['x']=-80.
    context.routes={'A':Route([[0.,-40.],[0.,40.]]),
                    'B':Route([[-80.,0.],[30.,0.],[35.,7.],[35.,15.]]),
                    '00':Route([[-80.,0.],[60.,0.]])}
    context.meta={o:{'points':r.points,'goal':list(r.at(r.length-5.)[:2]),'goal_s':r.length-5.}
                  for o,r in context.routes.items()}
    context.owners=['00','A','B'];context.indices={o:i for i,o in enumerate(context.owners)}
    context.zones=conflicts(context.routes)
    scenario=make_scenario(context.geometry,context.routes,context.meta,horizon=60.)
    poses={'A':(0.,-35.,math.pi/2),'B':(-20.,0.,0.),'00':(-55.,0.,0.)}
    for car in scenario['vehicles']:
        x,y,h=poses[car['id']];car.update(x=x,y=y,heading_rad=h,speed_mps=5.)
    sim=Simulation(scenario,record=False)
    driver=Driver(context.routes,context.meta,context.zones,context.geometry['roundabout'])
    _,states=driver.actions(sim)
    assert states['B'][:2]==('yielding','A')
    context.record['frames']=[physical_frame(sim,states,context.owners)]
    context.times=[0.];context.progress=[dict(driver.progress)]
    context._template=sim;context._public.clear();context._baseline_forecasts.clear()
    reasoner=TrafficReasoner(EncounterService(context=context))
    assessment=reasoner.assess('B',0.)
    gap=next(c for c in assessment['candidates'] if c.get('action_type')=='early_gap')
    normal=next(o for o in assessment['normal_order'] if o['zone_id']==gap['zone_id'])
    assert normal['first']=='B' and normal['second']=='A'
    assert gap['seller']=='A' and gap['buyer']=='B' and gap['changes_order'] is False
    forecast=context.forecast(0,gap)
    assert forecast['outcomes']['B']['passage_seconds_saved']>0
    assert gap['delivery_evidence']['order_observed']
    # The follower benefits from clearing the same queue, never an overtake.
    assert not any(c['buyer']=='00' and c['seller']=='B' for c in assessment['candidates'])


@pytest.mark.parametrize('distance,speed',[(.3,1.),(.8,1.),(2.,5.)])
def test_low_speed_servo_overrun_is_not_mistaken_for_max_braking_feasibility(distance,speed):
    from vehicle_days.encounters import can_hold_before_entry
    assert not can_hold_before_entry(distance,speed)


def test_cached_offer_is_revalidated_when_seller_can_no_longer_hold(context):
    demo=InteractiveDemo(context,ImmediateExecutor(),workers=1)
    command(demo,'advance',to=4.,after=0.)
    candidate=next(c for c in candidates(demo,'B',4.)['candidates'] if c['seller']=='A' and c['buyer']=='B')
    command(demo,'advance',to=6.,after=4.)
    before=deepcopy((demo.revision,demo.world.time,demo.world.cars,demo.frames,
                     demo.events,demo.requests,[asdict(c) for c in instructions(demo)]))
    with pytest.raises(ValueError):
        command(demo,'apply',car='B',at=6.,candidate=candidate['id'])
    after=(demo.revision,demo.world.time,demo.world.cars,demo.frames,
           demo.events,demo.requests,[asdict(c) for c in instructions(demo)])
    assert before==after


def test_distinct_actor_pairs_at_same_junction_are_related_resources(context):
    context.routes.update(C=Route([[40.,4.],[-40.,4.]]),
                          D=Route([[4.,40.],[4.,-40.]]))
    context.meta.update(C={'points':context.routes['C'].points,'goal':[-35.,4.],'goal_s':75.},
                        D={'points':context.routes['D'].points,'goal':[4.,-35.],'goal_s':75.})
    context.owners=['A','B','C','D'];context.indices={o:i for i,o in enumerate(context.owners)}
    context.zones=conflicts(context.routes)
    sim=Simulation(make_scenario(context.geometry,context.routes,context.meta,horizon=60.),record=False)
    driver=Driver(context.routes,context.meta,context.zones,context.geometry['roundabout'])
    _,states=driver.actions(sim)
    context.record['frames']=[physical_frame(sim,states,context.owners)]
    context.times=[0.];context.progress=[dict(driver.progress)]
    context._template=None;context._public.clear();context._baseline_forecasts.clear()
    demo=InteractiveDemo(context,ImmediateExecutor(),workers=1)
    related=next(c for c in candidates(demo,'C',0.)['candidates'] if c['seller']=='C' and c['buyer']=='D')
    first=next(c for c in candidates(demo,'A',0.)['candidates'] if c['seller']=='A' and c['buyer']=='B')
    command(demo,'apply',car='A',at=0.,candidate=first['id'])
    index=len(demo.frames)-1
    public=snapshot_context(demo._payload(index))
    assert not any(c['id']==related['id'] for c in public.candidates('C',demo.world.time)['candidates'])
    assert not {related['seller'],related['buyer']}.intersection({'A','B'})
    before=deepcopy((demo.world.time,demo.world.cars,demo.events,[asdict(c) for c in instructions(demo)]))
    assert 'overlapping road space' in demo._conflict_reason(related,index)
    assert before==(demo.world.time,demo.world.cars,demo.events,[asdict(c) for c in instructions(demo)])
    offered=candidates(demo,'C',demo.world.time)['candidates']
    assert not any(c['id']==related['id'] for c in offered)


def test_queue_member_requests_upstream_release_without_passing_its_head(context):
    context.geometry['nodes']['w']['x']=-80.
    context.routes={'A':Route([[0.,-40.],[0.,40.]]),
                    'B':Route([[-80.,0.],[30.,0.],[35.,7.],[35.,15.]]),
                    'Z':Route([[-80.,0.],[60.,0.]])}
    context.meta={o:{'points':r.points,'goal':list(r.at(r.length-5.)[:2]),'goal_s':r.length-5.}
                  for o,r in context.routes.items()}
    context.owners=['A','B','Z'];context.indices={o:i for i,o in enumerate(context.owners)}
    context.zones=conflicts(context.routes)
    scenario=make_scenario(context.geometry,context.routes,context.meta,horizon=60.)
    poses={'A':(0.,-35.,math.pi/2),'B':(-20.,0.,0.),'Z':(-26.,0.,0.)}
    for car in scenario['vehicles']:
        x,y,h=poses[car['id']];car.update(x=x,y=y,heading_rad=h,speed_mps=5.)
    sim=Simulation(scenario,record=False)
    driver=Driver(context.routes,context.meta,context.zones,context.geometry['roundabout'])
    _,states=driver.actions(sim)
    assert states['Z'][:2]==('following','B') and states['B'][:2]==('yielding','A')
    context.record['frames']=[physical_frame(sim,states,context.owners)]
    context.times=[0.];context.progress=[dict(driver.progress)]
    context._template=sim;context._public.clear();context._baseline_forecasts.clear()
    demo=InteractiveDemo(context,ImmediateExecutor(),workers=1)
    menu=candidates(demo,'Z',0.)
    release=next(c for c in menu['candidates'] if c['seller']=='A' and c['buyer']=='B')
    assert release['initiator']=='Z'
    assert set(release['beneficiaries'])=={'B','Z'}
    assert release['causal_chain']==['Z','B','A']
    assert not any(c['buyer']=='Z' and c['seller']=='B' for c in menu['candidates'])
    future=command(demo,'future',car='Z',at=0.,candidate=release['id'])
    assert future['status']=='ready' and set(future['owners'])=={'A','B','Z'}
    assert all(math.isfinite(e['seconds_saved']) for e in future['effects'].values())
    applied=command(demo,'apply',car='Z',at=0.,candidate=release['id'])
    assert applied['accepted']['buyer']=='B' and applied['accepted']['seller']=='A'
    assert {c.seller for c in instructions(demo)}=={'A'}

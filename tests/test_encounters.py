from copy import deepcopy
import gzip
import hashlib
import json
import threading

import pytest

from vehicle_days.driving import Simulation
from vehicle_days.encounters import Context, Concession, physical_frame, first_crossing
from vehicle_days.layout_traffic import Driver, Route, conflicts, make_scenario


@pytest.fixture
def context(tmp_path):
    geometry = {'nodes':{'w':{'x':-40.,'y':0.},'e':{'x':40.,'y':0.},
                         's':{'x':0.,'y':-40.},'n':{'x':0.,'y':40.}},
        'roads':[{'id':'h','a':'w','b':'e'},{'id':'v','a':'s','b':'n'}],
        'roundabout':{'x':1000.,'y':1000.,'radius':30.}}
    routes = {'A':Route([[-40.,0.],[40.,0.]]),'B':Route([[0.,-40.],[0.,40.]])}
    meta = {'A':{'goal':[35.,0.],'goal_s':75.},'B':{'goal':[0.,35.],'goal_s':75.}}
    zones = conflicts(routes)
    sim = Simulation(make_scenario(geometry,routes,meta,horizon=60.),record=False)
    driver = Driver(routes,meta,zones,geometry['roundabout'])
    frames=[];tick=0
    while not sim.termination:
        actions,states=driver.actions(sim)
        if tick%2==0:frames.append(physical_frame(sim,states,['A','B']))
        sim.step(actions);tick+=1
    _,states=driver.actions(sim);frames.append(physical_frame(sim,states,['A','B']))
    path=tmp_path/'map.json';path.write_text(json.dumps(geometry))
    record={'geometry_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'negotiations':0,
        'owners':['A','B'],'routes':{o:{'points':r.points,**meta[o]}for o,r in routes.items()},
        'conflicts':zones,'frames':frames,'arrivals':{o:c['arrival']for o,c in sim.cars.items()},
        'termination':sim.termination}
    source=tmp_path/'record.json.gz';source.write_bytes(gzip.compress(json.dumps(record).encode()))
    return Context(source,path)


def test_counterfactual_restores_both_cars_and_preserves_the_source(context):
    before=deepcopy(context.record)
    candidate=next(c for c in context.candidates('A',0)['candidates'] if c['seller']=='A')
    ordinary=context.physical(0)
    concession=context.physical(0,candidate)
    assert ordinary['termination']['kind']==concession['termination']['kind']=='completed'
    assert concession['concession_status']=='delivered'
    assert concession['arrivals']['B']<ordinary['arrivals']['B']
    assert concession['arrivals']['A']>ordinary['arrivals']['A']
    assert context.record==before


def test_normal_order_concession_can_have_zero_benefit(context):
    candidate=next(c for c in context.candidates('A',0)['candidates'] if c['seller']=='B')
    ordinary=context.physical(0)
    concession=context.physical(0,candidate)
    assert concession['concession_status']=='delivered'
    assert concession['arrivals']==ordinary['arrivals']


def test_public_forecast_does_not_read_future_recording_frames(context):
    index=context.frame_index(2.)
    inputs=deepcopy(context.public_world(index)['inputs'])
    for f in context.record['frames'][index+1:]:
        for c in f['cars']:c[0]+=10000.
    context.record['arrivals']={'A':100000.,'B':999999.}
    context._public.clear()
    assert context.public_world(index)['inputs']==inputs
    def keys(value):
        if isinstance(value,dict):
            return set(value).union(*(keys(v)for v in value.values()))
        if isinstance(value,list):
            return set().union(*(keys(v)for v in value))
        return set()
    assert not keys(inputs).intersection({'wallet','deadline','memory','destination','goal'})


def test_snapshot_replay_reproduces_the_recorded_baseline(context):
    index=context.frame_index(2.)
    sim,driver=context.restored_world(index)
    for _ in range(2):sim.step(driver.actions(sim)[0])
    expected=context.record['frames'][index+1]
    for owner,car in zip(context.owners,expected['cars']):
        assert sim.cars[owner]['x']==pytest.approx(car[0],abs=1e-8)
        assert sim.cars[owner]['y']==pytest.approx(car[1],abs=1e-8)


def test_cancelled_branch_never_reports_delivery(context):
    candidate=context.candidates('A',0)['candidates'][0]
    cancel=threading.Event();cancel.set()
    result=context.physical(0,candidate,cancel)
    assert result['termination']['kind']=='cancelled'
    assert result['concession_status']=='pending'


def test_commitment_detects_wrong_order_and_timeout():
    c=Concession('A','B',0,10.,20.,0.,5.)
    c.observe({'A':11.,'B':5.},1.)
    assert c.status=='violated_order'
    c=Concession('A','B',0,10.,20.,0.,5.)
    c.observe({'A':5.,'B':5.},6.)
    assert c.status=='delivery_timeout'


def test_forecast_returns_signed_passage_effect_and_unresolved_destination(context):
    listing=context.candidates('A',0)
    candidate=next(c for c in listing['candidates']if c['seller']=='A')
    forecast=context.forecast(listing['snapshot'],candidate)
    assert forecast['outcomes']['A']['passage_seconds_saved']<0
    assert forecast['outcomes']['B']['passage_seconds_saved']>0
    assert all(o['destination_seconds_saved'] is None for o in forecast['outcomes'].values())


def test_passage_times_are_interpolated():
    assert first_crossing([(0.,{'A':0.}),(1.,{'A':2.})],'A',1.)==.5


def test_upcoming_passage_can_be_discovered_outside_the_old_radius(context):
    candidates=context.candidates('A',0)['candidates']
    assert candidates
    assert all(c['distance_between_cars_m']>42.5 for c in candidates)


def test_forecast_replay_is_bounded_and_starts_from_the_same_observed_state(context):
    candidate=context.candidates('A',0)['candidates'][0]
    forecast=context.forecast(0,candidate)
    for branch in forecast['_branches'].values():
        assert branch['frames'][0]['t']==0.
        assert branch['frames'][-1]['t']<=25.
        assert branch['frames'][0]['cars'][0][:4]==context.record['frames'][0]['cars'][0][:4]

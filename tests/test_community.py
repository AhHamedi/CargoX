"""Behavioural checks for the actual one-day product, without paid inference."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import threading
import time
import math

import pytest

from vehicle_days.community import (OWNERS as ALL_OWNERS, Day, schedule, planner_context,
                                     validate_plan, permitted)
from district_fixture import tiny_district as district
from vehicle_days.community_models import comparison_cases, rendered
from vehicle_days.driving import replay
from vehicle_days.negotiation import Ledger, Contract


OWNERS=ALL_OWNERS[:4]

def day(seed=17, case='contention', bypass=()):
    scene=district(seed,case); profiles=schedule(seed,1,scene['route_options'],case)
    plans={o:dict(route='bypass' if o in bypass else scene['route_options'][o][0]['id'],
                  strategy='Own journeys',guidance='Choose voluntarily') for o in OWNERS}
    return Day(scene,profiles,plans,{o:Contract(4,allow_gift=True,max_yield_seconds=20) for o in OWNERS},
               Ledger({o:4 for o in OWNERS}),{o:{'memory':'','receipts':[]} for o in OWNERS})


def first_request(d):
    while not d.sim.termination:
        v=d.next_request()
        if v:return v
        d.step()
    raise AssertionError('Expected a real conflict')


def finish(d, action=lambda v:'decline'):
    while not d.sim.termination and not d.failure:
        v=d.next_request()
        if v:d.apply({'action':action(v)})
        d.step()
    return d


@pytest.mark.parametrize('seed',[17,29])
@pytest.mark.parametrize('bypass',[(),('C00',),('C02',),('C00','C02')])
def test_all_real_route_combinations_complete_without_negotiation(seed,bypass):
    d=finish(day(seed,bypass=bypass))
    assert d.sim.termination['kind']=='completed' and d.failure is None
    assert all(c['status']=='arrived' for c in d.sim.cars.values())
    assert d.accounts.audit()['ok'] and d.accounts.balances==dict.fromkeys(OWNERS,4)
    assert len(d.inspection_frames)==len(d.sim.frames)


def test_no_conflict_continues_without_unnecessary_negotiation():
    d=finish(day(case='no-conflict'))
    assert not d.decisions and not d.failure


def test_half_sized_contact_radius_filters_negotiation_but_preserves_driving_observation():
    from vehicle_days.district import make_district
    scene=make_district(count=8,size=3);owners=sorted(scene['route_options'])
    profiles=schedule(17,1,scene['route_options'])
    plans={o:dict(route=('early','middle','late')[int(o[1:])%3],strategy='test',guidance='test') for o in owners}
    d=Day(scene,profiles,plans,{o:Contract(4,allow_gift=True,max_yield_seconds=60) for o in owners},
          Ledger({o:4 for o in owners}),{o:{'receipts':[]} for o in owners},record=False)
    v=first_request(d)
    a=d.sim.cars[v['owner']];b=d.sim.cars[v['negotiation']['peer']]
    distance=math.hypot(a['x']-b['x'],a['y']-b['y'])
    uid=d.pending[0]
    assert d.negotiation_radius_m==42.5 and distance<=42.5
    d.negotiation_radius_m=distance-1e-6
    views,opportunities=d.opportunities()
    assert uid not in opportunities
    assert b['id'] in [p['id'] for p in views[a['id']]['nearby']]
    assert d.sim.observation_radius==85
    d.negotiation_radius_m=distance
    assert uid in d.opportunities()[1]
    assert d.recording()['negotiation_radius_m']==distance


def test_conflicting_merge_delivers_common_exit_order_without_stopping_the_day():
    from vehicle_days.district import make_district
    scene=make_district(count=8,size=3);owners=sorted(scene['route_options'])
    profiles=schedule(17,1,scene['route_options'])
    plans={o:dict(route=('early','middle','late')[int(o[1:])%3],strategy='test',guidance='test') for o in owners}
    d=Day(scene,profiles,plans,{o:Contract(4,allow_gift=True,max_yield_seconds=60) for o in owners},
          Ledger({o:4 for o in owners}),{o:{'receipts':[]} for o in owners},record=False)
    chosen=None
    while not d.sim.termination and not d.failure:
        v=d.next_request()
        if v:
            n=d.encounters[d.pending[0]]
            merge=n['seller_movement'].rsplit('_',1)[-1][-1]==n['buyer_movement'].rsplit('_',1)[-1][-1]
            if chosen is None and merge:chosen=n['id'];action='quote_1'
            elif n['id']==chosen and n['offer']:action='accept'
            else:action='decline'
            d.apply({'action':action})
        d.step()
    assert chosen and not d.failure and d.sim.termination['kind']=='completed'
    assert d.accounts.reservations[chosen]['status']=='delivered' and d.accounts.audit()['ok']
    event=next(e for e in d.timeline if e.get('kind')=='delivered')
    assert event['proof']['evidence']['delivery_kind']=='merge_exit_order'


def test_real_binding_deals_deliver_and_measured_effects_keep_each_owner():
    d=finish(day(),lambda v:'quote_1' if v['negotiation']['offer'] is None else 'accept')
    assert d.failure is None
    assert sum(e['kind']=='settle' for e in d.accounts.events)>=2
    assert not d.active() and d.accounts.audit()['ok']
    assert sum(d.accounts.balances.values())==16
    effects=d.measure_effects()
    assert len(effects)==sum(e['kind']=='settle' for e in d.accounts.events)
    assert all(set(e['seconds_saved'])==set(OWNERS) for e in effects)
    assert effects[0]['seconds_saved'][effects[0]['buyer']]>0
    assert effects[0]['seconds_saved'][effects[0]['seller']]<0
    assert 'Not a factual policy effect' in effects[0]['scope']


def test_physical_recording_replays_without_model_calls():
    d=finish(day(bypass=('C00',)))
    r=replay(d.sim.recording())
    assert r['termination']==d.sim.recording()['termination']
    for a,b in zip(r['frames'][-1]['cars'],d.sim.frames[-1]['cars']):
        assert (a['x'],a['y'],a['status'])==(b['x'],b['y'],b['status'])


def test_wide_prices_conserve_and_refund_exactly_once():
    ledger=Ledger({'buyer':16,'seller':0})
    ledger.reserve('high','buyer','seller',16,20,now=0)
    assert ledger.available('buyer')==0
    ledger.settle('high',5,evidence='physical witness')
    ledger.settle('high',6,evidence='duplicate')
    assert ledger.balances=={'buyer':0,'seller':16} and ledger.audit()['ok']
    ledger.reserve('refund','seller','buyer',12,30,now=10)
    ledger.refund('refund','cancelled',now=11);ledger.settle('refund',12)
    assert ledger.balances=={'buyer':0,'seller':16} and ledger.audit()['ok']
    with pytest.raises(ValueError):ledger.reserve('too-high','seller','buyer',17,40)


def test_private_views_do_not_change_when_peer_private_state_changes():
    d=day();v=first_request(d);own=v['owner'];peer=v['negotiation']['peer']
    uid=d.pending[0];n=d.encounters[uid]
    before=deepcopy(d.view(own,n))
    d.profiles[peer]['goal']['missed_cost']=999
    d.histories[peer]['memory']='secret'
    others=[o for o in OWNERS if o!=own]
    d.accounts.balances[others[0]]+=1;d.accounts.balances[others[1]]-=1
    assert d.view(own,n)==before
    context=planner_context(own,1,d.scenario,d.profiles,d.accounts,d.histories,{o:[] for o in OWNERS})
    assert 'secret' not in str(context)
    assert context['own_next_journey']['goal']==schedule(17,2,d.scenario['route_options'])[own]['goal']


def test_raw_forbidden_request_is_retained_without_contract():
    d=day();v=first_request(d)
    row=d.apply({'action':'quote_16','probabilities':{'quote_16':1}})
    assert row['requested']=='quote_16' and row['applied'] is None and not row['valid']
    assert not d.accounts.events


def test_counteroffer_and_current_proposer_consent_are_required():
    d=day();first_request(d);d.apply({'action':'quote_1'})
    d.next_request();d.apply({'action':'quote_3'})
    d.next_request();row=d.apply({'action':'accept'})
    assert row['valid'] and d.active()[0]['price']==3
    d.stop();assert not d.active() and d.accounts.audit()['ok']


def test_changed_proposer_authority_invalidates_old_offer():
    d=day();v=first_request(d);owner=v['owner'];d.apply({'action':'quote_1'})
    d.contracts[owner]=Contract(4,allow_gift=True,max_yield_seconds=20,version=2)
    d.next_request();row=d.apply({'action':'accept'})
    assert not row['valid'] and 'authority changed' in row['reason']
    assert not d.active()


def test_final_turn_does_not_offer_an_unacceptable_quote():
    d=day();v=first_request(d);v['negotiation']['turn']=3
    assert not any(a.startswith('quote_') for a in permitted(v))


def test_zero_yield_is_a_valid_private_plan_and_invalid_routes_are_rejected():
    d=day();ctx=planner_context('C00',1,d.scenario,d.profiles,d.accounts,d.histories,{o:[] for o in OWNERS})
    p=dict(route='shared',strategy='Be urgent',guidance='No yielding',spend_limit=4,reserve=0,max_yield_seconds=0)
    _,c=validate_plan(p,ctx,1);assert c.max_yield_seconds==0 and not c.allow_gift
    with pytest.raises(ValueError):validate_plan({**p,'route':'invented'},ctx,1)


def test_sixteen_disclosed_cases_expose_same_input_and_do_not_contain_peer_objectives():
    d=day();cases=comparison_cases(first_request(d))
    assert len(cases)==16
    for c in cases:
        assert isinstance(rendered(c['view']),str)
        assert c['expected'] is None or set(c['expected'])<=set(c['view']['permitted'])


@pytest.fixture
def server(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parents[1]/'src/vehicle_days/app.py'
    spec=importlib.util.spec_from_file_location('community_server',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'DATA',tmp_path)
    monkeypatch.setattr(module,'OWNERS',OWNERS)
    monkeypatch.setattr(module,'district',district)
    session=module.Session(tmp_path/'run');yield session
    session.close()


def test_stopped_epoch_cannot_apply_late_native_response(server):
    server.world=day();server.plans=server.world.plans;server.contracts=server.world.contracts
    server.accounts=server.world.accounts;first_request(server.world)
    entered=threading.Event();release=threading.Event()
    class Slow:
        def choose(self,view):entered.set();release.wait(3);return {'action':'quote_1'}
        def close(self):pass
    server.model_name='jev';server.models['jev']=Slow();server.status='plans_ready'
    server.command({'action':'run'});assert entered.wait(2)
    server.command({'action':'stop'});release.set();server.thread.join(3)
    assert server.status=='stopped' and not server.world.decisions and not server.world.active()


def test_pause_freezes_time_and_next_day_is_explicitly_review_gated(server):
    server.world=day();server.plans=server.world.plans;server.contracts=server.world.contracts
    server.accounts=server.world.accounts;server.phase='drive';server.status='running'
    server.command({'action':'pause'});clock=server.world.sim.time
    assert server.paused and server.snapshot()['frame']['time']==clock
    with pytest.raises(ValueError,match='review milestone'):server.command({'action':'next'})


def test_decision_step_advances_the_same_physical_tick_as_run(server):
    server.world=day();server.plans=server.world.plans;server.contracts=server.world.contracts
    server.accounts=server.world.accounts;server.speed=16
    class Decline:
        def choose(self,view):return {'action':'decline'}
        def close(self):pass
    server.model_name='jev';server.models['jev']=Decline();server.status='plans_ready'
    server.command({'action':'step'})
    deadline=time.monotonic()+5
    while server.status!='paused' and time.monotonic()<deadline:time.sleep(.01)
    assert server.status=='paused' and len(server.world.decisions)==1
    assert server.world.sim.time==pytest.approx(server.world.decisions[0]['time']+.1)

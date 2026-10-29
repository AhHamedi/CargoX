from copy import deepcopy

import pytest

from test_encounters import context
from vehicle_days.encounters import EncounterService
from vehicle_days.reasoning import TrafficReasoner, RollingContext, passage_evidence


def test_normal_priority_is_not_a_negotiable_move(context):
    reasoner=TrafficReasoner(EncounterService(context=context))
    result=reasoner.assess('A',0)
    assert result['cars_assessed']==2
    assert len(result['candidates'])==1
    action=result['candidates'][0]
    assert action['seller']=='A' and action['buyer']=='B'
    assert action['normal_order']==['A','B']
    assert action['delivery_evidence']['order_observed']
    assert all(a['seller']!='B' for a in result['candidates'])


def test_each_car_gets_the_same_joint_current_state(context):
    reasoner=TrafficReasoner(EncounterService(context=context))
    global_state=reasoner.assess(None,0)
    own=reasoner.assess('A',0)
    assert global_state['all_agents']==own['all_agents']
    assert global_state['candidates']==[]
    assert global_state['cars_assessed']==len(context.owners)
    assert reasoner.assess('B',0)['candidates'][0]['id']==own['candidates'][0]['id']


def test_options_change_after_the_crossing(context):
    reasoner=TrafficReasoner(EncounterService(context=context))
    assert reasoner.assess('A',0)['candidates']
    assert reasoner.assess('A',context.times[-1])['candidates']==[]


def test_unfulfilled_alternative_is_rejected_not_offered(context):
    forecast=context.forecast
    def timeout(index,candidate):
        f=forecast(index,candidate)
        f['concession_status']='delivery_timeout'
        return f
    context.forecast=timeout
    r=TrafficReasoner(EncounterService(context=context)).assess('A',0)
    assert r['candidates']==[]
    assert r['rejected'][0]['reason']=='delivery_timeout'


def test_action_execution_rechecks_current_snapshot_and_default(context):
    reasoner=TrafficReasoner(EncounterService(context=context))
    r=reasoner.assess('A',0);a=r['candidates'][0]
    assert reasoner.require_action('A',0,a['id'],r['snapshot'])==a
    with pytest.raises(ValueError,match='recalculate'):
        reasoner.require_action('A',0,a['id'],r['snapshot']+1)
    default=next(a for a in context.candidates('A',0)['candidates'] if a['seller']=='B')
    with pytest.raises(ValueError,match='normal priority'):
        reasoner.require_action('A',0,default['id'],r['snapshot'])


def test_reasoner_does_not_read_recorded_future_to_choose_normal_order(context):
    r1=TrafficReasoner(EncounterService(context=context)).assess('A',0)
    for f in context.record['frames'][1:]:
        for car in f['cars']:car[0]+=10000
    context.record['arrivals']={o:999999 for o in context.owners}
    r2=TrafficReasoner(EncounterService(context=context)).assess('A',0)
    assert r1['candidates']==r2['candidates']


def test_native_evidence_checks_clearance_before_entry(context):
    a=next(a for a in context.candidates('A',0)['candidates'] if a['seller']=='A')
    native=context.physical(0,a)
    proof=passage_evidence(context,native['frames'],a,0)
    assert proof['order_observed']
    assert proof['timing']['B']['clears']<=proof['timing']['A']['enters']
    normal=context.physical(0)
    assert not passage_evidence(context,normal['frames'],a,0)['order_observed']


def test_delivery_filter_does_not_reward_or_require_profit(context):
    original=context.forecast
    def harmful(index,candidate):
        f=original(index,candidate)
        for outcome in f['outcomes'].values():
            outcome['passage_seconds_saved']=-100.
        return f
    context.forecast=harmful
    result=TrafficReasoner(EncounterService(context=context)).assess('A',0)
    assert result['candidates']
    assert all(v['passage_seconds_saved']==-100 for v in result['candidates'][0]['predicted_effects'].values())

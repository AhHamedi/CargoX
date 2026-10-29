"""Avoidable-delay discovery, independently of presentation or trained policies."""
from copy import deepcopy
import math

from test_encounters import context
from vehicle_days.encounters import (Concession, EncounterService, can_hold_before_entry,
    following_chain, project)
from vehicle_days.reasoning import TrafficReasoner, _early_gap_changes, persistent_wait_cycle


def place(context, owner, s, speed=0., reason='driving', peer=None):
    row=context.record['frames'][0]['cars'][context.indices[owner]]
    row[:4]=[*context.routes[owner].at(s),speed]
    row[4:7]=['moving',reason,peer]
    row[8]=0.
    context.progress[0][owner]=s
    context._public.clear();context._baseline_forecasts.clear()


def test_stopped_requester_at_five_metre_yield_line_can_negotiate(context):
    zone=context.zones[0]
    place(context,'A',zone['ranges']['A'][0]-20.,3.)
    place(context,'B',zone['ranges']['B'][0]-5.,0.,'yielding','A')
    candidates=context.candidates('B',0.)['candidates']
    action=next(a for a in candidates if a['seller']=='A' and a['buyer']=='B')
    assessed=TrafficReasoner(EncounterService(context=context)).assess('B',0.)
    assert any(a['id']==action['id'] for a in assessed['candidates'])
    assert context.progress[0]['B']==zone['ranges']['B'][0]-5.


def test_stopped_seller_need_not_reverse_to_nominal_hold_line(context):
    zone=context.zones[0]
    place(context,'A',zone['ranges']['A'][0]-2.,0.)
    place(context,'B',zone['ranges']['B'][0]-12.,2.)
    assert any(a['seller']=='A' for a in context.candidates('B',0.)['candidates'])
    assert can_hold_before_entry(2.,0.)


def test_moving_seller_that_cannot_stop_is_not_advertised(context):
    zone=context.zones[0]
    place(context,'A',zone['ranges']['A'][0]-2.,5.)
    place(context,'B',zone['ranges']['B'][0]-12.,2.)
    assert not can_hold_before_entry(2.,5.)
    assert all(a['seller']!='A' for a in context.candidates('B',0.)['candidates'])
    assert can_hold_before_entry(20.,5.)


def test_already_occupied_passage_cannot_be_sold_retroactively(context):
    zone=context.zones[0]
    place(context,'A',zone['ranges']['A'][0]+.5,0.)
    place(context,'B',zone['ranges']['B'][0]-5.,0.)
    assert context.candidates('B',0.)['candidates']==[]


def test_public_commitments_are_copied_and_rebased_not_mutated(context):
    instruction=context.instruction(0,context.candidates('A',0.)['candidates'][0])
    context.commitments=[instruction]
    before=deepcopy(instruction)
    public=context.public_commitments(0)
    offset=context.public_world(0)['offsets']
    assert public[0].seller_entry==instruction.seller_entry-offset[instruction.seller]
    assert public[0].buyer_exit==instruction.buyer_exit-offset[instruction.buyer]
    public[0].status='delivered'
    assert instruction==before
    assert not context.candidates('A',0.)['candidates']


def test_projection_baseline_respects_preexisting_promise_and_keeps_it_private(context):
    action=next(a for a in context.candidates('A',0.)['candidates'] if a['seller']=='A')
    instruction=context.instruction(0,action,public=True)
    before=deepcopy(instruction)
    world=context.public_world(0)
    result=project(world,context.geometry['roundabout'],0.,existing=[instruction])
    assert instruction==before
    assert result['existing_statuses']==['delivered']
    zone=next(z for z in world['zones'] if z['id']==action['zone_id'])
    from vehicle_days.encounters import first_crossing
    buyer=first_crossing(result['progress'],'B',zone['ranges']['B'][1])
    seller=first_crossing(result['progress'],'A',zone['ranges']['A'][0])
    assert buyer is not None and seller is not None and buyer<seller


def test_incomplete_horizon_is_a_conditional_option_not_an_absent_option(context):
    original=context.forecast
    def incomplete(index,candidate):
        result=original(index,candidate)
        result['concession_status']='pending'
        return result
    context.forecast=incomplete
    assessment=TrafficReasoner(EncounterService(context=context)).assess('B',0.)
    assert assessment['candidates']
    assert assessment['candidates'][0]['delivery_status']=='conditional'
    assert assessment['candidates'][0]['assessment']=='conditional_delay_exchange'
    assert all(math.isfinite(v['passage_seconds_saved']) for v in assessment['candidates'][0]['predicted_effects'].values())


def test_stable_wait_cycle_is_different_from_merely_unresolved_passage():
    def row(reason,peer):return[0.,0.,0.,0.,'moving',reason,peer,[],0.,0.]
    frames=[{'t':t,'cars':[row('yielding','B'),row('following','A')]} for t in (0.,1.,2.,3.)]
    assert persistent_wait_cycle(frames,['A','B'],['A'])==['A','B','A']
    frames[-1]['cars'][1][3]=1.
    assert persistent_wait_cycle(frames,['A','B'],['A'])==[]


def test_same_order_gap_requires_buyer_motion_but_not_added_donor_motion_cost():
    def row(x,speed,acc=0.):return[x,0.,0.,speed,'moving','driving',None,[],acc,0.]
    ordinary=[{'t':0.,'cars':[row(0.,0.),row(20.,0.)]},
              {'t':1.,'cars':[row(0.,0.),row(20.,0.)]}]
    agreed=deepcopy(ordinary);agreed[1]['cars'][0]=row(1.,2.,2.)
    changes=_early_gap_changes(ordinary,agreed,{'A':0,'B':1},'A','B')
    assert changes=={'buyer_motion_changes':True,'donor_motion_changes':False}
    assert not _early_gap_changes(ordinary,ordinary,{'A':0,'B':1},'A','B')['buyer_motion_changes']


def test_following_chain_reaches_upstream_cause_without_fabricating_overtake():
    def row(reason,peer):return[0.,0.,0.,0.,'moving',reason,peer,[],0.,0.]
    cars={'A':row('following','B'),'B':row('following','C'),'C':row('yielding','D'),'D':row('driving',None)}
    assert following_chain(cars,'A')==['A','B','C']


def test_announced_parking_stop_cannot_be_promised_as_a_future_clearance(context):
    context.meta['B']['goal_s']=context.zones[0]['ranges']['B'][1]-.5
    context._public.clear()
    assert all(c['buyer']!='B' for c in context.candidates('A',0.)['candidates'])


def test_stationary_queue_ending_at_a_parked_body_is_not_horizon_uncertainty():
    from vehicle_days.reasoning import terminal_parked_queue
    def row(reason,peer,status='moving'):return[0.,0.,0.,0.,status,reason,peer,[],0.,0.]
    frames=[{'t':t,'cars':[row('following','B'),row('parked_vehicle','C'),row('arrived',None,'arrived')]} for t in (0.,1.,2.,3.)]
    assert terminal_parked_queue(frames,['A','B','C'],'A')==['A','B','C']
    frames[-1]['cars'][1][3]=1.
    assert terminal_parked_queue(frames,['A','B','C'],'A')==[]


def test_delivered_exchange_is_not_hidden_by_a_later_following_relationship(context):
    original=context.forecast
    def later_following(index,candidate):
        value=original(index,candidate)
        if value['concession_status']=='delivered':
            # Local passage has already completed; a later control relationship
            # is not grounds to pretend that exchange never existed.
            row=value['_branches']['concession']['frames'][-1]['cars'][context.indices[candidate['buyer']]]
            row[4:7]=['moving','following',candidate['seller']]
        return value
    context.forecast=later_following
    value=TrafficReasoner(EncounterService(context=context)).assess('B',0.)
    assert value['candidates'] and value['candidates'][0]['delivery_evidence']['order_observed']

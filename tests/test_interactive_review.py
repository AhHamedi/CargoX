"""Independent checks of the interactive world's externally observable semantics."""
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import asdict
import math

import pytest

from test_encounters import context
from vehicle_days.interactive import InteractiveDemo, assess_snapshot, effect_estimate


class ImmediateExecutor:
    """Run the real forecast function deterministically, without a worker process."""
    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future


def new_demo(context):
    demo = InteractiveDemo(context, ImmediateExecutor(), workers=1)
    return demo, demo.command({'action':'new'})


def observed(demo):
    return deepcopy({'revision':demo.revision, 'time':demo.world.time,
        'cars':demo.world.cars, 'frames':demo.frames, 'progress':demo.progress,
        'events':demo.events, 'arrivals':demo.arrivals,
        'native_states':demo.native_states, 'contracts':demo.contracts,
        'instructions':[asdict(c) for c in demo.instructions],
        'ended':demo.ended, 'failure':demo.failure})


def command(demo, action, **fields):
    return demo.command({'action':action, 'revision':demo.revision, **fields})


def available(demo, owner='A', at=0.):
    packet = command(demo, 'options', car=owner, at=at)
    assert packet['status']=='ready'
    return packet


def test_hover_future_is_read_only_and_returned_frames_are_detached(context):
    demo, _ = new_demo(context)
    physics_before = deepcopy((demo.world.time, demo.world.cars, demo.contracts))
    action = available(demo)['candidates'][0]
    # Observing traffic may now publish autonomous NPC requests, but may not
    # advance physics or create a binding agreement. Speculation stays pure.
    assert physics_before == (demo.world.time, demo.world.cars, demo.contracts)
    before = observed(demo)
    future = command(demo, 'future', car='A', at=0., candidate=action['id'])
    assert future['status']=='ready'
    assert future['owners']==['A','B']
    assert len(future['frames'])>1
    assert observed(demo)==before
    # An animation consumer cannot accidentally mutate authoritative reality.
    future['frames'][0]['cars'][0][0] += 1000.
    assert observed(demo)==before


def test_stale_revision_and_invalid_apply_are_atomic(context):
    demo, initial = new_demo(context)
    action = available(demo)['candidates'][0]
    demo.command({'action':'new'})
    before = observed(demo)
    with pytest.raises(ValueError, match='older world'):
        demo.command({'action':'apply','revision':initial['revision'], 'car':'A',
                      'at':0., 'candidate':action['id']})
    assert observed(demo)==before
    with pytest.raises(ValueError):
        command(demo, 'apply', car='A', at=0., candidate='not-an-option')
    assert observed(demo)==before


def test_finite_rough_effect_does_not_claim_an_observed_crossing():
    baseline = {'progress':[(2.,{'A':0.}), (7.,{'A':10.})]}
    alternative = {'progress':[(2.,{'A':0.}), (7.,{'A':5.})]}
    estimate = effect_estimate(baseline, alternative, 'A', target=100.)
    assert math.isfinite(estimate['seconds_saved'])
    assert estimate['seconds_saved']<0
    assert estimate['confidence']=='rough'
    assert estimate['scope']=='time equivalent over the forecast horizon'
    assert estimate['horizon_seconds']==5.
    assert 'progress difference' in estimate['method']
    zero = effect_estimate(baseline, baseline, 'A', target=100.)
    assert zero['seconds_saved']==0. and zero['confidence']=='rough'


def test_crossing_estimate_uses_interpolated_times_when_available():
    baseline = {'progress':[(0.,{'A':0.}), (4.,{'A':20.})]}
    alternative = {'progress':[(0.,{'A':0.}), (4.,{'A':10.})]}
    estimate = effect_estimate(baseline, alternative, 'A', target=5.)
    assert estimate['seconds_saved']==pytest.approx(-1.)
    assert estimate['confidence']=='conditional'
    assert estimate['scope']=='at this crossing'


def test_recorded_future_and_arrivals_are_not_live_inputs(context):
    first, _ = new_demo(context)
    first_frames = command(first, 'advance', to=3., after=-1.)['frames']
    for frame in context.record['frames'][1:]:
        for car in frame['cars']:
            car[0] += 10000.
            car[1] -= 10000.
    context.record['arrivals'] = {o:100000. for o in context.owners}
    second, _ = new_demo(context)
    second_frames = command(second, 'advance', to=3., after=-1.)['frames']
    assert first_frames==second_frames
    assert second.arrivals=={}


def test_accepted_choice_changes_continuation_and_later_forecast_origin(context):
    changed, start = new_demo(context)
    baseline, _ = new_demo(context)
    action = next(a for a in available(changed)['candidates'] if a['seller']=='A')
    accepted = command(changed, 'apply', car='A', at=0., candidate=action['id'])
    assert accepted['revision']==start['revision']+1
    assert accepted['active_contract']['seller']=='A'
    assert accepted['active_contract']['buyer']=='B'
    assert accepted['frames'][0]['t']==0.
    for until in (8.,16.,24.):
        command(changed, 'advance', to=until, after=-1.)
        command(baseline, 'advance', to=until, after=-1.)
    assert changed.frames!=baseline.frames
    assert len([e for e in changed.events if e['kind']=='accepted'])==1
    assert len([e for e in changed.events if e['kind']=='delivered'])==1
    # Later forecasts start from the changed reality, with no restoration to
    # an ordinary recording or to the end of the previously displayed future.
    index = len(changed.frames)-1
    payload = changed._payload(index)
    before = observed(changed)
    assessment = assess_snapshot(payload, 'A')
    assert assessment['time']==changed.frames[index]['t']
    assert payload['frame']==changed.frames[index]
    assert observed(changed)==before
    command(changed, 'advance', to=32., after=-1.)
    assert len([e for e in changed.events if e['kind']=='delivered'])==1


def test_a_later_decision_uses_reality_changed_by_the_first_agreement(context):
    """Three cars, two crossings: accepting one move must not end the demo."""
    from vehicle_days.driving import Simulation
    from vehicle_days.encounters import physical_frame
    from vehicle_days.layout_traffic import Driver, Route, conflicts, make_scenario

    context.geometry['nodes'].update(cs={'x':20.,'y':-75.}, cn={'x':20.,'y':45.})
    context.geometry['roads'].append({'id':'v2','a':'cs','b':'cn'})
    context.routes['C'] = Route([[20.,-75.],[20.,45.]])
    context.meta['C'] = {'points':context.routes['C'].points, 'goal':[20.,40.], 'goal_s':115.}
    context.owners = ['A','B','C']
    context.indices = {o:i for i,o in enumerate(context.owners)}
    context.zones = conflicts(context.routes)
    sim = Simulation(make_scenario(context.geometry,context.routes,context.meta,horizon=60.),record=False)
    driver = Driver(context.routes,context.meta,context.zones,context.geometry['roundabout'])
    _, states = driver.actions(sim)
    context.record['frames'] = [physical_frame(sim,states,context.owners)]
    context.times = [0.]
    context.progress = [dict(driver.progress)]
    context._template = None

    demo, _ = new_demo(context)
    initial = available(demo)
    first = next(a for a in initial['candidates'] if a['seller']=='A' and a['buyer']=='B')
    command(demo,'apply',car='A',at=0.,candidate=first['id'])
    second = None
    for _ in range(90):
        command(demo,'advance',to=demo.world.time+.2,after=demo.world.time)
        if demo.instruction.status=='delivered':
            choices = available(demo,at=demo.world.time)
            second = next((a for a in choices['candidates'] if a['id']!=first['id']),None)
            if second:
                break
    assert second is not None, 'The second crossing must remain interactive.'
    effective_state = deepcopy(demo.frames[-1]['cars'])
    when, revision = demo.world.time, demo.revision
    packet = command(demo,'apply',car='A',at=when,candidate=second['id'])
    assert packet['revision']==revision+1
    assert packet['frames'][0]['t']==pytest.approx(when)
    assert [r[:5] for r in packet['frames'][0]['cars']]==[r[:5] for r in effective_state]
    assert len([e for e in packet['events'] if e['kind']=='accepted'])==2
    assert len([e for e in packet['events'] if e['kind']=='delivered'])==1


def test_forecast_uses_the_same_unbroken_promise_as_live_execution(context):
    from vehicle_days.interactive import snapshot_context
    demo, _ = new_demo(context)
    public = snapshot_context(demo._payload(0))
    candidate = public.candidates('A', 0.)['candidates'][0]
    promise = public.instruction(0, candidate)
    promise.observe({promise.seller:promise.seller_entry-1.,
                     promise.buyer:promise.buyer_exit-1.}, 21.)
    assert promise.status=='pending'
    assert promise.expires>25.

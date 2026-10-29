from concurrent.futures import Future
from copy import deepcopy

import pytest

from test_encounters import context
from vehicle_days.interactive import InteractiveDemo, assess_snapshot, effect_estimate


class ImmediateExecutor:
    def submit(self, function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future


def demo_for(context):
    return InteractiveDemo(context, ImmediateExecutor(), workers=1)


def test_ordinary_reality_preserves_the_existing_physical_engine(context):
    demo = demo_for(context)
    new = demo.command({'action':'new'})
    packet = demo.command({'action':'advance','revision':new['revision'],'to':1.,'after':0.})
    assert packet['model']=='Kinematic bicycle physics'
    for frame, expected in zip(packet['frames'], context.record['frames'][1:]):
        assert frame['t']==pytest.approx(expected['t'])
        for actual, recorded in zip(frame['cars'], expected['cars']):
            assert actual[:4]==pytest.approx(recorded[:4],abs=1e-8)


def test_future_recording_is_never_used_to_advance_the_demonstration(context):
    demo = demo_for(context)
    new = demo.command({'action':'new'})
    for frame in context.record['frames'][1:]:
        for car in frame['cars']:
            car[0] += 10000.
    context.record['arrivals'] = {'A':9999.,'B':9999.}
    packet = demo.command({'action':'advance','revision':new['revision'],'to':1.,'after':0.})
    assert all(abs(car[0])<100 for f in packet['frames'] for car in f['cars'])
    assert packet['arrivals']=={}


def test_hover_future_is_speculative_and_does_not_change_reality(context):
    demo = demo_for(context)
    new = demo.command({'action':'new'})
    options = demo.command({'action':'options','revision':new['revision'],'car':'A','at':0.})
    action = options['candidates'][0]
    before = deepcopy((demo.frames,demo.events,demo.revision,demo.world.cars))
    future = demo.command({'action':'future','revision':new['revision'],'car':'A','at':0.,'candidate':action['id']})
    assert future['status']=='ready'
    assert future['owners']==['A','B']
    assert all(set(effect)>= {'seconds_saved','confidence','method'} for effect in future['effects'].values())
    assert before==(demo.frames,demo.events,demo.revision,demo.world.cars)


def test_accepting_an_option_changes_continuing_physics_not_just_a_clip(context):
    demo = demo_for(context)
    new = demo.command({'action':'new'})
    options = demo.command({'action':'options','revision':new['revision'],'car':'A','at':0.})
    action = options['candidates'][0]
    ordinary = demo.command({'action':'advance','revision':new['revision'],'to':6.,'after':0.})
    applied = demo.command({'action':'apply','revision':new['revision'],'car':'A','at':0.,'candidate':action['id']})
    assert applied['revision']==new['revision']+1
    assert applied['frames'][0]['t']==0
    changed = demo.command({'action':'advance','revision':applied['revision'],'to':6.,'after':2.})
    assert changed['frames'][-1]['cars']!=ordinary['frames'][-1]['cars']
    assert len([e for e in applied['events'] if e['kind']=='accepted'])==1
    assert demo.instruction.status=='pending'
    later = demo.command({'action':'advance','revision':applied['revision'],'to':14.,'after':6.})
    assert later['revision']==applied['revision']
    assert len([e for e in later['events'] if e['kind']=='accepted'])==1
    assert len([e for e in later['events'] if e['kind']=='delivered'])==1
    demo.command({'action':'advance','revision':applied['revision'],'to':18.,'after':14.})
    assert len([e for e in demo.events if e['kind']=='delivered'])==1


def test_invalid_or_stale_apply_cannot_mutate_the_world(context):
    demo = demo_for(context)
    new = demo.command({'action':'new'})
    before = deepcopy((demo.frames,demo.events,demo.world.cars,demo.revision))
    for revision, candidate in ((new['revision']-1,'0:A:B'),(new['revision'],'wrong')):
        with pytest.raises(ValueError):
            demo.command({'action':'apply','revision':revision,'car':'A','at':0.,'candidate':candidate})
        assert before==(demo.frames,demo.events,demo.world.cars,demo.revision)


def test_rough_effect_is_finite_without_claiming_an_observed_crossing():
    base = {'progress':[(0.,{'A':0.}),(25.,{'A':30.})]}
    alternative = {'progress':[(0.,{'A':0.}),(25.,{'A':45.})]}
    estimate = effect_estimate(base,alternative,'A',100.)
    assert estimate['seconds_saved']==3.
    assert estimate['confidence']=='rough'
    assert estimate['scope']=='time equivalent over the forecast horizon'


def test_pending_agreement_survives_car_change_and_forecast_timeout(context):
    demo = demo_for(context)
    new = demo.command({'action':'new'})
    action = demo.command({'action':'options','revision':new['revision'],'car':'A','at':0.})['candidates'][0]
    applied = demo.command({'action':'apply','revision':new['revision'],'car':'A','at':0.,'candidate':action['id']})
    other = demo.command({'action':'options','revision':applied['revision'],'car':'B','at':.2})
    assert other['active_contract']['status']=='pending'
    assert other['candidates']==[]
    assert demo.instruction.expires>1e8


def test_background_work_is_bounded_and_owner_change_cancels_queued_work(context):
    class PendingExecutor:
        def __init__(self):
            self.calls = []
        def submit(self, function, *args):
            future = Future()
            self.calls.append(future)
            return future
    executor = PendingExecutor()
    demo = InteractiveDemo(context,executor,workers=2)
    revision = demo.revision
    demo.command({'action':'advance','revision':revision,'to':4.,'after':0.})
    packet = demo.command({'action':'options','revision':revision,'car':'A','at':0.})
    assert packet['status']=='pending'
    assert packet['stats']['pending']<=4
    old = list(executor.calls)
    demo.command({'action':'options','revision':revision,'car':'B','at':0.})
    assert all(f.cancelled() for f in old)
    assert len(demo.jobs)<=4


def test_forecast_worker_failure_does_not_stop_physics(context):
    class BrokenExecutor:
        def submit(self, *args):
            raise RuntimeError('Worker unavailable')
    demo = InteractiveDemo(context,BrokenExecutor(),workers=1)
    packet = demo.command({'action':'options','revision':demo.revision,'car':'A','at':0.})
    assert packet['status']=='failed'
    assert packet['error']=='Worker unavailable'
    advanced = demo.command({'action':'advance','revision':demo.revision,'to':1.,'after':0.})
    assert advanced['time']==pytest.approx(1.)
    assert not advanced['ended']


def test_obsolete_background_reads_are_superseded_without_a_request_failure(context):
    demo = demo_for(context)
    old = demo.revision
    demo.command({'action':'new'})
    for action in ('advance','options','future'):
        result = demo.command({'action':action,'revision':old,'to':2.,'car':'A','at':0.,'candidate':'0:A:B'})
        assert result['status']=='superseded'
        assert result['revision']==demo.revision
        assert result['frames']==[]
    assert demo.world.time==0.
    with pytest.raises(ValueError):
        demo.command({'action':'apply','revision':old,'car':'A','at':0.,'candidate':'0:A:B'})


def test_reset_can_interrupt_a_buffer_fill_without_error_or_old_frames(context):
    import threading
    demo = demo_for(context)
    old = demo.revision
    frame_published, reset_finished = threading.Event(), threading.Event()
    # Arrange reset when the loop publishes and explicitly yields its lock,
    # without relying on thread timing.
    original_sleep = __import__('vehicle_days.interactive',fromlist=['time']).time.sleep
    import vehicle_days.interactive as module
    results = []
    def yield_once(seconds):
        if threading.current_thread().name=='buffer-fill':
            frame_published.set()
            assert reset_finished.wait(2.)
        else:
            original_sleep(seconds)
    from unittest.mock import patch
    with patch.object(module.time,'sleep',yield_once):
        thread=threading.Thread(name='buffer-fill',target=lambda:results.append(demo.command({'action':'advance','revision':old,'to':4.,'after':0.})))
        thread.start()
        assert frame_published.wait(2.)
        new=demo.command({'action':'new'})
        reset_finished.set()
        thread.join(2.)
    assert not thread.is_alive()
    assert results[0]['status']=='superseded'
    assert results[0]['revision']==new['revision']
    assert results[0]['frames']==[]
    assert demo.world.time==0.


def test_invalid_pace_is_rejected_before_any_world_mutation(context):
    demo = demo_for(context)
    action = demo.command({'action':'options','revision':demo.revision,'car':'A','at':0.})['candidates'][0]
    before = deepcopy((demo.frames,demo.events,demo.native_states,demo.world.cars,demo.revision,demo.latest_observed))
    for pace in (0,4.1,float('nan'),float('inf'),True,'2',None):
        with pytest.raises(ValueError,match='pace'):
            demo.command({'action':'apply','revision':demo.revision,'car':'A','at':0.,'candidate':action['id'],'pace':pace})
        assert before==(demo.frames,demo.events,demo.native_states,demo.world.cars,demo.revision,demo.latest_observed)


def test_pace_changes_buffer_length_without_changing_physical_trajectory(context):
    ordinary, faster = demo_for(context), demo_for(context)
    def accept(demo, pace):
        action=demo.command({'action':'options','revision':demo.revision,'car':'A','at':0.})['candidates'][0]
        return demo.command({'action':'apply','revision':demo.revision,'car':'A','at':0.,'candidate':action['id'],'pace':pace})
    slow=accept(ordinary,1.)
    fast=accept(faster,4.)
    assert slow['time']==pytest.approx(2.)
    assert fast['time']==pytest.approx(4.)
    assert slow['frames']==[frame for frame in fast['frames'] if frame['t']<=2.]
    assert slow['events']==fast['events']


def test_reset_preserves_observed_drive_but_not_unused_prebuffer(context,tmp_path):
    import gzip,json
    directory=tmp_path/'drives'
    demo=InteractiveDemo(context,ImmediateExecutor(),workers=1,directory=directory)
    demo.command({'action':'advance','revision':demo.revision,'to':2.,'after':0.})
    demo.command({'action':'new'})
    assert not directory.exists()
    demo.command({'action':'advance','revision':demo.revision,'to':2.,'after':0.})
    demo.command({'action':'options','revision':demo.revision,'car':'A','at':1.})
    revision=demo.revision
    expected=deepcopy(demo.frames)
    demo.command({'action':'new'})
    files=list(directory.glob('*.json.gz'))
    assert len(files)==1
    value=json.loads(gzip.decompress(files[0].read_bytes()))
    assert value['frames']==expected
    assert value['revision']==revision and value['owners']==context.owners
    assert value['observed_until']==1. and value['buffered_until']==2.
    assert value['save_reason']=='reset'
    assert value['model']=='Kinematic bicycle physics'
    assert 'prepared continuation' in value['observation_scope']
    assert demo.selected is None and demo.latest_observed==0.
    assert not list(directory.glob('*.tmp'))


def test_physical_fault_is_preserved_atomically_before_reset(context,tmp_path):
    import gzip,json
    directory=tmp_path/'drives'
    demo=InteractiveDemo(context,ImmediateExecutor(),workers=1,directory=directory)
    demo.world._end('road_departure',['A'])
    demo._capture()
    files=list(directory.glob('*.json.gz'))
    assert len(files)==1
    value=json.loads(gzip.decompress(files[0].read_bytes()))
    assert value['failure']=='road_departure' and value['ended']
    assert value['events'][-1]['kind']=='physical_fault'
    assert value['frames'][-1]['t']==demo.world.time
    assert value['save_reason']=='fault'
    demo.close()
    closed=json.loads(gzip.decompress(files[0].read_bytes()))
    assert closed['failure']=='road_departure' and closed['save_reason']=='close'
    assert not list(directory.glob('*.tmp'))

from collections import OrderedDict
from concurrent.futures import Future
import time

import pytest

from test_encounters import context
from vehicle_days.encounters import EncounterService
from vehicle_days.reasoning import TrafficReasoner
from vehicle_days.forecast_stream import ForecastStream


class Executor:
    def __init__(self):self.calls=[]
    def submit(self, fn, owner, at):
        f=Future();self.calls.append((f,owner,at));return f
    def shutdown(self,**kwargs):pass


def make(context):
    context.forecasts=OrderedDict()
    service=EncounterService(context=context)
    executor=Executor()
    return ForecastStream(service,TrafficReasoner(service),workers=2,executor=executor),executor


def complete(context,call):
    f,owner,at=call;i=context.frame_index(at)
    result={'source':context.source,'snapshot':i,'time':context.times[i],
        'candidates':[],'seconds':.1,'cars_assessed':len(context.owners)}
    f.set_result({'assessment':result,'forecasts':[],'worker_seconds':.1,'worker_pid':123})


def test_slow_calculation_returns_pending_without_waiting(context):
    stream,e=make(context);began=time.perf_counter()
    r=stream.poll('A',0)
    assert r['status']=='pending'
    assert time.perf_counter()-began<.1
    assert len(e.calls)<=stream.workers+2


def test_future_results_never_exposed_early(context):
    stream,e=make(context);stream.poll('A',0)
    later=next(call for call in e.calls if call[2]>0)
    complete(context,later)
    assert stream.poll('A',0)['status']=='pending'
    complete(context,e.calls[0])
    assert stream.poll('A',0)['result']['time']==0
    assert stream.poll('A',later[2])['result']['time']==later[2]


def test_exact_fork_requires_the_requested_snapshot(context):
    stream,e=make(context);stream.poll('A',0,exact=True)
    complete(context,e.calls[0])
    assert stream.poll('A',0,exact=True)['status']=='ready'
    assert stream.poll('A',.2,exact=True)['status']=='pending'
    assert stream.poll('A',.2)['result']['time']<=.2


def test_car_switch_cancels_obsolete_queued_work(context):
    stream,e=make(context);stream.poll('A',0);old=list(e.calls)
    stream.poll('B',0)
    assert all(f.cancelled()for f,_,_ in old)
    assert stream.poll('B',0)['status']=='pending'
    assert stream.discarded>0


def test_worker_failure_is_visible_without_stalling_playback(context):
    stream,e=make(context);stream.poll('A',0,exact=True)
    e.calls[0][0].set_exception(RuntimeError('test worker failed'))
    r=stream.poll('A',0,exact=True)
    assert r['status']=='failed' and 'test worker' in r['error']


def test_closed_scheduler_rejects_new_work(context):
    stream,e=make(context);stream.poll('A',0);stream.close()
    with pytest.raises(ValueError,match='stopped'):stream.poll('A',0)


def test_completed_work_is_reused_by_exact_action_revalidation(context):
    stream,e=make(context);stream.poll('A',0,exact=True)
    complete(context,e.calls[0])
    def unexpected_recalculation(index):raise AssertionError('Completed forecast was recalculated')
    stream.reasoner._joint=unexpected_recalculation
    result=stream.reasoner.assess('A',0)
    assert result['cache_hit'] and result['time']==0

"""Physical lane order must not be sold as a negotiable crossing order."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from test_encounters import context
from test_interactive import ImmediateExecutor
from vehicle_days.encounters import Concession, EncounterService, shared_hold_obstruction
from vehicle_days.interactive import InteractiveDemo, snapshot_context
from vehicle_days.layout_traffic import Route
from vehicle_days.reasoning import TrafficReasoner


def reported_snapshot(name):
    value=json.loads((Path(__file__).parent/'fixtures/c66-shared-approach.json').read_text())[name]
    candidate=value.pop('candidate')
    value['source']='C66-public-regression'
    value['routes']={o:Route(p) for o,p in value['routes'].items()}
    return snapshot_context(value),candidate


def test_reported_c28_deal_is_impossible_even_when_c66_reports_driving():
    c,candidate=reported_snapshot('blocked_second')
    assert c.record['frames'][0]['cars'][c.indices['C66']][5]=='driving'
    proof=c.hold_obstruction(0,candidate)
    assert proof['reason']=='shared_lane_order'
    assert proof['buyer_at_hold']<proof['buyer_clearance']-20.
    result=TrafficReasoner(EncounterService(context=c)).assess('C66',c.times[0])
    assert all(a['id']!=candidate['id'] for a in result['candidates'])
    # Admission must also reject a previously cached option independently of
    # whether this reduced two-car rollout resolves ordinary order in 25 s.
    demo=InteractiveDemo.__new__(InteractiveDemo)
    demo.session='regression';demo.revision=1
    demo.context=c;demo.frames=c.record['frames'];demo.progress=c.progress;demo.contracts=[[]]
    assert 'shared approach' in demo._conflict_reason(candidate,0)


def test_reported_first_c30_deal_has_different_approaches():
    c,candidate=reported_snapshot('valid_first')
    assert c.hold_obstruction(0,candidate) is None
    result=TrafficReasoner(EncounterService(context=c)).assess('C66',c.times[0])
    assert any(a['id']==candidate['id'] for a in result['candidates'])


def test_the_shared_corridor_includes_bends_not_only_current_heading():
    a=Route([[0.,0.],[30.,0.],[30.,60.]])
    assert shared_hold_obstruction(a,a,20.,5.,70.,80.)


def test_buyer_ahead_of_seller_does_not_need_an_overtake():
    a=Route([[0.,0.],[150.,0.]])
    assert shared_hold_obstruction(a,a,20.,40.,100.,120.) is None


def test_clipping_at_a_rounded_world_space_vertex_is_stable():
    route=Route([[1000.,0.],[1050.,0.],[1150.,0.]])
    assert shared_hold_obstruction(route,route,70.,50.-1e-14,120.,135.)


@pytest.mark.parametrize('other',[
    [[60.,-80.],[60.,80.]],                  # Independent crossing approach.
    [[150.,0.],[0.,0.]],                    # Opposite direction.
    [[0.,4.],[150.,4.]],                    # Separate parallel lane.
    [[0.,0.],[45.,0.],[45.,80.]],            # Buyer turns before the hold site.
    [[0.,0.],[45.,0.],[45.,20.],[75.,20.],[75.,0.],[150.,0.]],
])
def test_crossings_and_bypasses_are_not_banned_by_a_future_hold_point(other):
    a=Route([[0.,0.],[150.,0.]])
    b=Route(other)
    assert shared_hold_obstruction(a,b,25.,5.,105.,b.length-2.) is None


def test_current_public_geometry_is_used_without_private_tails():
    c,candidate=reported_snapshot('blocked_second')
    before=deepcopy(c.public_world(0)['inputs'])
    c.routes['C28'].points.append([90000.,90000.])
    assert c.hold_obstruction(0,candidate)
    assert c.public_world(0)['inputs']==before


def test_an_existing_impossible_promise_stops_and_records_a_fault(context,tmp_path):
    # A legacy promise must never be quietly released or counted as delivery.
    context.routes['B']=Route([[-60.,0.],[40.,0.]])
    context.meta['B']=deepcopy(context.meta['A'])
    context.meta['B']['goal_s']=95.
    context.zones=[{'id':0,'a':'A','b':'B','x':10.,'y':0.,'ranges':{'A':[50.,60.],'B':[70.,80.]}}]
    for owner,s in [('A',30.),('B',30.)]:
        row=context.record['frames'][0]['cars'][context.indices[owner]]
        row[:4]=[*context.routes[owner].at(s),0.]
        row[4:7]=['moving','driving',None]
        context.progress[0][owner]=s
    context._template=None;context._public.clear();context._baseline_forecasts.clear()
    demo=InteractiveDemo(context,ImmediateExecutor(),directory=tmp_path)
    promise=Concession('A','B',0,50.,80.,0.,1e9)
    demo.instructions=[promise]
    demo._capture()
    assert demo.ended and 'shared lane' in demo.failure
    assert promise.status=='unfulfillable_passage'
    assert [e['kind'] for e in demo.events]==['execution_fault']
    assert list(tmp_path.glob('*.json.gz'))

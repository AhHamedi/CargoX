"""A donor's binding stop is independent of its displayed driving reason."""
from types import SimpleNamespace

from vehicle_days.encounters import Concession, EncounterService
from vehicle_days.interactive import InteractiveContext
from vehicle_days.layout_traffic import Route
from vehicle_days.reasoning import (TrafficReasoner, held_promise_edges,
                                    persistent_wait_cycle)


OWNERS=['A','Q','B','X']
PROPOSAL={'seller':'A','buyer':'B','zone_id':0}


def row(x, reason, peer, speed=0.):
    return [x,0.,0.,speed,'moving',reason,peer,[],0.,0.]


def queue_frames(donor=60., buyer=41.):
    return [{'t':float(t),'cars':[row(donor,'yielding','X'),
            row(donor-9.5,'following','A'),row(buyer,'following','Q'),
            row(100.,'driving',None)]} for t in range(4)]


def public_context():
    route=Route([[0.,0.],[120.,0.]])
    return SimpleNamespace(
        indices={o:i for i,o in enumerate(OWNERS)},
        public_world=lambda index:{'routes':{o:route for o in OWNERS}},
        public_commitments=lambda index:[],
        instruction=lambda index,candidate,public:Concession('A','B',0,65.,90.,0.,1e9))


def test_hard_hold_cycle_survives_a_different_displayed_primary_reason():
    frames=queue_frames()
    # Ordinary displayed edges end at X, which is not part of the cycle.
    assert persistent_wait_cycle(frames,OWNERS,['B'])==[]
    promises=held_promise_edges(public_context(),0,frames,PROPOSAL)
    cycle=persistent_wait_cycle(frames,OWNERS,['B'],promises)
    assert cycle==['B','Q','A','B']
    assert frames[-1]['cars'][0][6]=='X'


def test_following_a_donor_far_before_its_hold_does_not_imply_deadlock():
    # The donor can still advance to a later hold while the buyer turns away
    # at an earlier fork. A pending promise alone is not a stationary edge.
    frames=queue_frames(donor=25.,buyer=6.)
    promises=held_promise_edges(public_context(),0,frames,PROPOSAL)
    assert not any(promises.values())
    assert persistent_wait_cycle(frames,OWNERS,['B'],promises)==[]


def test_a_donor_still_moving_to_the_boundary_is_not_a_held_edge():
    frames=queue_frames()
    for frame in frames:
        frame['cars'][0][3]=1.
    assert not any(held_promise_edges(public_context(),0,frames,PROPOSAL).values())


def test_a_cleared_buyer_does_not_leave_a_phantom_hard_dependency():
    frames=queue_frames(buyer=95.)
    assert not any(held_promise_edges(public_context(),0,frames,PROPOSAL).values())


def test_c66_forecast_rejects_transitive_queue_even_without_current_geometry_rejection():
    context=InteractiveContext()
    obstruction=context.hold_obstruction
    # Exercise the future safeguard separately from current-pose admission.
    context.hold_obstruction=lambda index,candidate,frame=None:(
        obstruction(index,candidate,frame=frame) if frame is not None else None)
    result=TrafficReasoner(EncounterService(context=context)).assess('C66',65.)
    rejected=next(r for r in result['rejected'] if r['id']=='1200:C28:C66')
    assert rejected['reason']=='buyer_queued_behind_seller'
    assert rejected['blocking_queue']==['C66','C30','C28']
    assert not any(c['id']=='1200:C28:C66' for c in result['candidates'])

"""Positive containment hints must preserve full road-union decisions."""
from copy import deepcopy
import math

import pytest

from vehicle_days.driving import VehicleConfig
from vehicle_days.driving_roads import Network


def network():
    return Network({'roads':[
        {'id':'horizontal','points':[[-40.,0.],[40.,0.]],'lanes':1,'lane_width':6.},
        {'id':'vertical','points':[[0.,-40.],[0.,40.]],'lanes':1,'lane_width':6.},
        {'id':'remote','points':[[70.,0.],[110.,0.]],'lanes':1,'lane_width':6.},
    ]})


def test_hint_rechecks_current_pose_before_using_positive_certificate():
    net=network();config=VehicleConfig()
    car={'id':'A','x':0.,'y':0.,'heading_rad':0.}
    assert net.contains_body(car,config)
    assert 'A' in net._body_surface_cache
    class CountingGrid(dict):
        calls=0
        def get(self,*args):
            self.calls+=1
            return super().get(*args)
    net._grid=CountingGrid(net._grid)
    assert net.contains_body({**car,'x':1.},config)
    assert net._grid.calls==0
    assert not net.contains_body({**car,'x':55.},config)
    assert net._grid.calls>0
    assert net.contains_body({**car,'x':80.},config)
    assert not net.contains_body({**car,'x':80.,'y':4.},config)


def test_hint_and_full_union_test_agree_at_boundaries_and_turns():
    cached,uncached=network(),network()
    for config in (VehicleConfig(),VehicleConfig(length=6.,width=2.2)):
        for tolerance in (0.,.05,.2,10.):
            for x,y in [(0.,0.),(1.,2.),(0.,8.),(8.,0.),(39.,0.),(40.,2.),
                        (0.,3.),(3.,3.),(3.05,3.05),(55.,0.),(80.,0.),(80.,3.),
                        (110.,0.),(-39.,2.8),(-40.,4.),(40.+1e-12,3.)]:
                for heading in (0.,math.pi/8,math.pi/4,math.pi/2,math.pi):
                    car={'id':'A','x':x,'y':y,'heading_rad':heading}
                    uncached._body_surface_cache.clear()
                    uncached._body_pose_cache.clear()
                    uncached._body_patch_cache.clear()
                    uncached._body_patch_attempt.clear()
                    assert cached.contains_body(car,config,tolerance)==uncached.contains_body(car,config,tolerance)


@pytest.mark.parametrize("at",[20.,60.,90.])
def test_short_eighty_car_physical_continuation_is_identical_without_hints(at,monkeypatch):
    from vehicle_days.encounters import Context
    import vehicle_days.layout_traffic as layout
    context=Context();index=context.frame_index(at)
    cached,driver_a=context.restored_world(index)
    uncached,driver_b=context.restored_world(index)
    uncached.network=deepcopy(uncached.network)
    full_test=uncached.network.contains_body
    def without_hints(*args,**kwargs):
        uncached.network._body_surface_cache.clear()
        uncached.network._body_pose_cache.clear()
        uncached.network._body_patch_cache.clear()
        uncached.network._body_patch_attempt.clear()
        return full_test(*args,**kwargs)
    uncached.network.contains_body=without_hints
    uncached.network._certify_body_patch=lambda *args: None
    for _ in range(20):
        a,state_a=driver_a.actions(cached)
        with monkeypatch.context() as original:
            original.setattr(layout,"_parked_body_can_limit",lambda *args: True)
            b,state_b=driver_b.actions(uncached)
        assert (a,state_a)==(b,state_b)
        cached.step(a);uncached.step(b)
        assert cached.snapshot()==uncached.snapshot()
    assert cached.termination is None


def test_far_parked_broadphase_cannot_skip_a_limiting_sat_gap():
    from vehicle_days.driving import separation
    from vehicle_days.layout_traffic import _parked_body_can_limit
    for config in (VehicleConfig(),VehicleConfig(length=6.,width=2.2)):
        limit=2*(math.hypot(config.length,config.width)+1.2+1e-7)**2
        for angle in range(32):
            bearing=angle*math.tau/32
            for gap in (math.sqrt(limit)+1e-6,12.,30.,55.):
                peer={'x':gap*math.cos(bearing),'y':gap*math.sin(bearing),'heading_rad':angle*.371}
                assert not _parked_body_can_limit(0.,0.,peer,limit)
                for heading in (0.,.37,math.pi/4,math.pi/2):
                    own={'x':0.,'y':0.,'heading_rad':heading}
                    assert separation(own,peer,config)>1.2

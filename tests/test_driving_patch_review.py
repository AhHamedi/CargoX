"""Independent geometry checks for positive-only pavement certificates."""
import math
from types import SimpleNamespace

import pytest

from vehicle_days.driving import VehicleConfig
from vehicle_days.driving_roads import Network, _EPS, _cross, _subtract, _strict_subtract


def pose(x=0.,y=0.,heading=0.,owner='review'):
    return {'id':owner,'x':x,'y':y,'heading_rad':heading}


def full_union_without_memos(network,car,config,tolerance=.05):
    """The original complete union test, without any cache or spatial shortcut."""
    inset=tolerance/math.sqrt(2.)
    hl,hw=max(0.,config.length/2-inset),max(0.,config.width/2-inset)
    c,s=math.cos(car['heading_rad']),math.sin(car['heading_rad'])
    body=[(car['x']+u*hl*c-v*hw*s,car['y']+u*hl*s+v*hw*c)
          for u,v in ((-1,-1),(1,-1),(1,1),(-1,1))]
    uncovered=[body]
    bounds=(min(p[0] for p in body),min(p[1] for p in body),max(p[0] for p in body),max(p[1] for p in body))
    for surface,box in network._surfaces:
        if box[0]>bounds[2] or box[2]<bounds[0] or box[1]>bounds[3] or box[3]<bounds[1]:
            continue
        if all(_cross(a,b,p)>=-_EPS for a,b in zip(surface,surface[1:]+surface[:1]) for p in body):
            return True
        uncovered=[piece for poly in uncovered for piece in _subtract(poly,surface)]
        if not uncovered:
            return True
    return False


@pytest.mark.parametrize('offset',(0.,10000.))
@pytest.mark.parametrize('angle',(0.,.31,math.pi/2,2.4))
def test_warmed_certificates_preserve_analytic_flat_road_boundaries(offset,angle):
    c,s=math.cos(angle),math.sin(angle)
    def xy(x,y):return [offset+x*c-y*s,offset+x*s+y*c]
    network=Network({'roads':[{'id':'flat','points':[xy(-30.,0.),xy(30.,0.)],
                              'lanes':1,'lane_width':6.}]})
    config=VehicleConfig()
    assert network.contains_body(pose(*xy(0.,0.),heading=angle),config,tolerance=0.)
    for tolerance in (0.,.05,.2):
        for relative in (0.,math.pi/12,math.pi/4,math.pi/2):
            hl,hw=config.length/2-tolerance/math.sqrt(2),config.width/2-tolerance/math.sqrt(2)
            ex=abs(math.cos(relative))*hl+abs(math.sin(relative))*hw
            ey=abs(math.sin(relative))*hl+abs(math.cos(relative))*hw
            for sign in (-1,1):
                for delta in (-.01,.01):
                    for x,y in ((sign*(30.-ex+delta),0.),(0.,sign*(3.-ey+delta))):
                        actual=network.contains_body(pose(*xy(x,y),heading=angle+relative),config,tolerance)
                        assert actual is (delta<0.), (offset,angle,tolerance,relative,x,y)


def test_pose_memo_rechecks_angle_dimensions_and_tolerance():
    network=Network({'roads':[{'id':'flat','points':[[-30.,0.],[30.,0.]],
                              'lanes':1,'lane_width':6.}]})
    normal=VehicleConfig()
    assert network.contains_body(pose(y=1.8),normal,tolerance=0.)
    assert not network.contains_body(pose(y=1.8,heading=math.pi/2),normal,tolerance=0.)
    assert not network.contains_body(pose(y=1.8),SimpleNamespace(length=4.5,width=3.),tolerance=0.)
    assert network.contains_body(pose(y=2.9),normal,tolerance=1.2)
    assert not network.contains_body(pose(y=2.9),normal,tolerance=0.)


def test_a_concave_crossing_cannot_be_certified_from_four_paved_corners():
    network=Network({'roads':[
        {'id':'h','points':[[-30.,0.],[30.,0.]],'lanes':1,'lane_width':6.},
        {'id':'v','points':[[0.,-30.],[0.,30.]],'lanes':1,'lane_width':6.}]})
    config=VehicleConfig()
    assert network.contains_body(pose(),config,tolerance=0.)
    # Every corner lies in one arm, but the front edge crosses the missing
    # concave quadrant (x>3 and y>3); corner membership alone is insufficient.
    rejected=pose(2.,2.,math.pi/4)
    c,s=math.cos(rejected['heading_rad']),math.sin(rejected['heading_rad'])
    corners=[(2.+u*2.25*c-v*.9*s,2.+u*2.25*s+v*.9*c)
             for u,v in ((-1,-1),(1,-1),(1,1),(-1,1))]
    assert all(abs(x)<=3. or abs(y)<=3. for x,y in corners)
    assert not network.contains_body(rejected,config,tolerance=0.)


def test_warmed_body_cannot_bridge_a_hole_between_paved_components():
    network=Network({'roads':[
        {'id':'lower','points':[[-30.,-2.],[30.,-2.]],'lanes':1,'lane_width':2.2},
        {'id':'upper','points':[[-30.,2.],[30.,2.]],'lanes':1,'lane_width':2.2},
        {'id':'bridge','points':[[-20.,-4.],[-20.,4.]],'lanes':1,'lane_width':7.}]})
    config=SimpleNamespace(length=4.5,width=5.)
    assert network.contains_body(pose(-20.,0.),config,tolerance=0.)
    assert not network.contains_body(pose(0.,0.),config,tolerance=0.)


def curved_network():
    return Network({'roads':[{'id':'arc','points':[
        [30.*math.cos(t),30.*math.sin(t)] for t in [-.6+i*.03 for i in range(41)]],
        'lanes':1,'lane_width':3.5}]})


def test_expanded_patch_is_exercised_and_matches_uncached_union_on_turns():
    network=curved_network();config=VehicleConfig()
    assert network.contains_body(pose(30.,0.,math.pi/2),config)
    assert network._body_patch_cache, 'Fixture must exercise expanded union certificates.'
    class CountingGrid(dict):
        calls=0
        def get(self,*args):
            self.calls+=1
            return super().get(*args)
    network._grid=CountingGrid(network._grid)
    angle=.002
    assert network.contains_body(pose(30.*math.cos(angle),30.*math.sin(angle),angle+math.pi/2),config)
    assert network._grid.calls==0, 'A genuine patch hit should avoid the full union search.'
    for angle in [-.3+i*.05 for i in range(13)]:
        for lateral in (0.,.3,-.3,1.1,-1.1):
            car=pose((30.+lateral)*math.cos(angle),(30.+lateral)*math.sin(angle),angle+math.pi/2)
            assert network.contains_body(car,config)==full_union_without_memos(network,car,config)


def test_strict_coverage_retains_subthreshold_slivers_at_translated_coordinates():
    origin=1e6
    top=origin+5.
    almost_top=math.nextafter(top,-math.inf)
    patch=[(origin,origin),(origin+5.,origin),(origin+5.,top),(origin,top)]
    pavement=[(origin-1.,origin-1.),(origin+6.,origin-1.),
              (origin+6.,almost_top),(origin-1.,almost_top)]
    remainder=_strict_subtract(patch,pavement,0.)
    assert remainder, 'A real uncovered sliver must prevent a positive certificate.'
    assert any(any(p[1]>almost_top for p in piece) for piece in remainder)


def test_strict_certificate_insets_pavement_before_accepting_a_patch():
    square=[(-2.,-2.),(2.,-2.),(2.,2.),(-2.,2.)]
    assert not _strict_subtract(square,square,0.)
    assert _strict_subtract(square,square,1e-7)


def test_declining_an_expensive_certificate_preserves_the_original_decision(monkeypatch):
    import vehicle_days.driving_roads as roads
    strict_subtract=roads._strict_subtract
    def exhausted(poly,cutter,margin,budget=None):
        if budget is not None:
            budget[0]=0
        return strict_subtract(poly,cutter,margin,budget)
    monkeypatch.setattr(roads,'_strict_subtract',exhausted)
    network=curved_network();config=VehicleConfig()
    assert network.contains_body(pose(30.,0.,math.pi/2),config)
    assert not network._body_patch_cache
    outside=pose(31.1,0.,math.pi/2)
    assert not network.contains_body(outside,config)
    assert not network._body_patch_cache

"""Small physical fixture for focused protocol tests; not an application scenario."""
import math
import random
from vehicle_days.driving_roads import Network,nearest_on_path
from vehicle_days.network_driving import _join,SPEED

def tiny_district(seed=17, case='contention'):
    """Two shared crossings; a curved bypass ends beyond both crossings.

    Northbound destinations are off the bypass, and horizontal cars have
    separate parking bays. The bypass is physically longer, not a label.
    """
    scenario = {'version': 3, 'name': 'Two crossings', 'seed': seed, 'horizon': 150.,
        'observation_radius_m': 85., 'negotiation_radius_m': 85.,
        'roads': [], 'connections': [], 'signals': [],
        'junctions': {f'J{i}0': {'center': [x, -60.], 'bounds': [x-15, -75, x+15, -45]}
                      for i, x in enumerate((-60., 0.))}, 'vehicles': []}
    def road(name, pts):
        scenario['roads'].append(dict(id=name, points=pts, lanes=1, lane_width=5., speed_limit=SPEED))
    def link(name, src, dst, pts, **kw):
        scenario['connections'].append(dict(id=name, **{'from': [src, 0], 'to': [dst, 0]},
                                             points=pts, width=5., priority=0, **kw))
    road('west', [[-220, -60], [-112, -60]])
    road('approach', [[-108, -60], [-75, -60]])
    road('between', [[-45, -60], [-15, -60]])
    road('east', [[15, -60], [108, -60]])
    road('out', [[112, -60], [145, -60]])
    link('fork', 'west', 'approach', [[-112, -60], [-108, -60]])
    link('rejoin', 'east', 'out', [[108, -60], [112, -60]])
    for i, x in enumerate((-60., 0.)):
        road(f'south{i}', [[x, -270], [x, -75]])
        road(f'north{i}', [[x, -45], [x, -28]])
        link(f'J{i}0_WE', 'approach' if i == 0 else 'between',
             'between' if i == 0 else 'east', [[x-15, -60], [x+15, -60]],
             junction_id=f'J{i}0', movement='WE')
        link(f'J{i}0_SN', f'south{i}', f'north{i}', [[x, -75], [x, -45]],
             junction_id=f'J{i}0', movement='SN')
    def arc(cx, cy, a, b):
        return [[cx+15*math.cos(a+(b-a)*i/16), cy+15*math.sin(a+(b-a)*i/16)] for i in range(17)]
    bypass = arc(-112, -45, -math.pi/2, 0)
    bypass += [[-97, -40]] + arc(-82, -40, math.pi, math.pi/2)[1:]
    bypass += [[82, -25]] + arc(82, -40, math.pi/2, 0)[1:]
    bypass += [[97, -45]] + arc(112, -45, math.pi, 1.5*math.pi)[1:]
    link('bypass', 'west', 'out', bypass)
    for owner, y in (('C00', -45), ('C02', -75)):
        pts = [[145+60*u, -60+(y+60)*(3*u*u-2*u*u*u)] for u in [i/32 for i in range(33)]]
        road('bay'+owner, [[205, y], [230, y]])
        link('park'+owner, 'out', 'bay'+owner, pts)
    rng = random.Random(seed)
    initial = {'C00': [-160., -60., 0.], 'C02': [-186., -60., 0.],
               'C01': [-60., -156., math.pi/2], 'C03': [0., -238., math.pi/2]}
    if case == 'no-conflict':
        initial['C01'][1] = initial['C03'][1] = -100.
    routes = {}
    for owner in ('C00', 'C01', 'C02', 'C03'):
        x, y, heading = initial[owner]
        if heading == 0: x += rng.uniform(-1., 1.)
        else: y += rng.uniform(-1., 1.)
        if owner in ('C00', 'C02'):
            ending = ['out:0', 'park'+owner, 'bay'+owner+':0']
            options = [('shared', ['west:0', 'fork', 'approach:0', 'J00_WE', 'between:0', 'J10_WE', 'east:0', 'rejoin']+ending),
                       ('bypass', ['west:0', 'bypass']+ending)]
            goal = [215., -45. if owner == 'C00' else -75.]
        else:
            i = 0 if owner == 'C01' else 1
            options = [('crossing', [f'south{i}:0', f'J{i}0_SN', f'north{i}:0'])]
            goal = [x, -35.]
        scenario['vehicles'].append(dict(id=owner, x=x, y=y, heading_rad=heading, speed_mps=SPEED,
            goal=goal, goal_radius=2., cruise_cap_mps=SPEED, path=options[0][1], route_choice=options[0][0]))
        routes[owner] = [dict(id=rid, path=path) for rid, path in options]
    network = Network(scenario)
    for car in scenario['vehicles']:
        for route in routes[car['id']]:
            points = _join(network, route['path'])
            route['distance_m'] = round(nearest_on_path(points, *car['goal'])[0]-nearest_on_path(points, car['x'], car['y'])[0], 3)
            route['free_seconds'] = round(route['distance_m']/SPEED+2.4, 3)
    scenario['route_options'] = routes
    return scenario


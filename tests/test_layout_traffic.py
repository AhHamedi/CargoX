"""Physical checks for the model-free district driver, with no model providers."""
from vehicle_days.driving import Simulation
from vehicle_days.layout_traffic import Driver, Route, conflicts
from vehicle_days.driving_roads import nearest_on_path


def world(paths, goals, starts=None):
    routes = {owner: Route(points) for owner, points in paths.items()}
    scenario = {'version': 3, 'horizon': 55., 'roads': [], 'vehicles': []}
    meta = {}
    for owner, route in routes.items():
        x, y, h = route.at((starts or {}).get(owner, 5.))
        scenario['roads'].append({'id': owner, 'points': route.points, 'lanes': 1, 'lane_width': 5.})
        scenario['vehicles'].append({'id': owner, 'x': x, 'y': y, 'heading_rad': h,
            'goal': goals[owner], 'goal_radius': 2., 'cruise_cap_mps': 5.})
        meta[owner] = {'goal_s': nearest_on_path(route.points, *goals[owner])[0]}
    sim = Simulation(scenario, record=False)
    driver = Driver(routes, meta, conflicts(routes), {'x': 1000., 'y': 1000.})
    return sim, driver


def test_simultaneous_crossing_yields_and_completes_without_contact():
    sim, driver = world({'A': [[-40., 0.], [40., 0.]], 'B': [[0., -40.], [0., 40.]]},
                        {'A': [35., 0.], 'B': [0., 35.]})
    yielded = set()
    while not sim.termination:
        actions, states = driver.actions(sim)
        yielded.update(o for o, state in states.items() if state[0] == 'yielding')
        sim.step(actions)
    assert sim.termination['kind'] == 'completed'
    assert yielded
    assert all(c['status'] == 'arrived' for c in sim.cars.values())


def test_opposing_lanes_pass_without_reserving_a_false_conflict():
    sim, driver = world({'A': [[-40., -1.9], [40., -1.9]], 'B': [[40., 1.9], [-40., 1.9]]},
                        {'A': [35., -1.9], 'B': [-35., 1.9]})
    assert not driver.zones
    while not sim.termination:
        actions, states = driver.actions(sim)
        assert all(state[0] != 'yielding' for state in states.values())
        sim.step(actions)
    assert sim.termination['kind'] == 'completed'


def test_queue_leader_does_not_yield_to_its_follower_for_a_future_turn():
    # An overlapping future turn must not reverse cars on the same approach.
    paths = {'A': [[-40., 0.], [40., 0.]], 'B': [[-40., 0.], [40., 0.]]}
    routes = {o: Route(p) for o, p in paths.items()}
    zone = {'id': 0, 'a': 'A', 'b': 'B', 'ranges': {'A': [35., 50.], 'B': [35., 50.]}, 'x': 0., 'y': 0.}
    driver = Driver(routes, {o: {'goal_s': 75.} for o in routes}, [zone], {'x': 1000., 'y': 1000.})
    from types import SimpleNamespace
    sim = SimpleNamespace(time=0., cars={
        'A': {'x': -20., 'y': 0., 'heading_rad': 0., 'speed_mps': 2., 'status': 'moving'},
        'B': {'x': -10., 'y': 0., 'heading_rad': 0., 'speed_mps': 2., 'status': 'moving'}})
    actions, states = driver.actions(sim)
    assert states['A'][0] == 'following'
    assert states['B'][0] == 'driving'
    assert actions['B']['cruise_auto']['speed_mps'] > sim.cars['B']['speed_mps']
    assert not states['B'][2]


def test_parking_approaches_do_not_reverse_across_the_last_junction():
    import json
    import math
    from vehicle_days.layout_traffic import MAP, make_routes
    routes, _ = make_routes(json.loads(MAP.read_text()))
    for route in routes.values():
        heading = route.at(route.length)[2]
        for a, b in zip(route.points[-26:-1], route.points[-25:]):
            forward = (b[0]-a[0])*math.cos(heading)+(b[1]-a[1])*math.sin(heading)
            assert forward >= -1e-6, 'The pull-in must stay on the final approach'


def test_same_path_at_different_times_is_not_a_competing_passage():
    from vehicle_days.layout_traffic import passage_overlap
    # The reported C18/C00 case: the leader clears in ~1 s, follower enters in ~6 s.
    assert not passage_overlap([335.,351.5], [297.5,320.], 305.4896,315.0244,5.,5.)
    assert passage_overlap([0.,20.], [0.,20.], -5.,-8.,0.,3.)


def test_three_car_priority_inversion_cannot_freeze_a_junction():
    sim, driver = world({'A': [[-40.,0.],[40.,0.]], 'B': [[0.,-40.],[0.,40.]],
                         'C': [[-40.,0.],[40.,0.]]},
                        {'A': [20.,0.], 'B': [0.,35.], 'C': [35.,0.]},
                        starts={'A':15.,'B':25.,'C':25.})
    while not sim.termination:
        actions, _ = driver.actions(sim)
        sim.step(actions)
    assert sim.termination['kind'] == 'completed'


def test_queue_head_does_not_wait_for_a_follower_more_than_thirty_metres_back():
    horizontal = [[-80.,0.],[80.,0.]]
    paths = {o: horizontal for o in ('A','C','D','E','F')}
    paths['B'] = [[0.,-80.],[0.,80.]]
    goals = {o:[x,0.] for o,x in {'A':10.,'F':22.,'E':34.,'D':46.,'C':58.}.items()}
    goals['B'] = [0.,60.]
    sim, driver = world(paths, goals, starts={'A':34.,'F':43.,'E':52.,'D':61.,'C':70.,'B':68.})
    while not sim.termination:
        actions, _ = driver.actions(sim)
        sim.step(actions)
    assert sim.termination['kind'] == 'completed'


def test_parked_body_remains_an_obstacle_but_its_future_route_does_not():
    from types import SimpleNamespace
    from vehicle_days.driving import VehicleConfig
    routes = {'C':Route([[0.,0.],[100.,0.]]),'A':Route([[40.,8.],[40.,-20.]])}
    zone = {'id':0,'a':'A','b':'C','ranges':{'A':[0.,10.],'C':[25.,45.]},'x':40.,'y':0.}
    driver = Driver(routes,{'C':{'goal_s':95.},'A':{'goal_s':0.}},[zone],{'x':1000.,'y':1000.})
    sim = SimpleNamespace(time=0.,config=VehicleConfig(),cars={
        'C':dict(x=20.,y=0.,heading_rad=0.,speed_mps=2.,status='moving'),
        'A':dict(x=40.,y=8.,heading_rad=-1.57,speed_mps=0.,status='arrived')})
    actions, states = driver.actions(sim)
    assert states['C'][0] == 'driving'
    assert actions['C']['cruise_auto']['speed_mps'] == 5.
    sim.cars['C']['x'] = 35.
    sim.cars['A']['y'] = 0.
    actions, states = driver.actions(sim)
    assert states['C'][0] == 'parked_vehicle'
    assert actions['C']['cruise_auto']['speed_mps'] == 0.


def test_binding_yield_survives_the_ordinary_same_approach_shortcut():
    from types import SimpleNamespace
    from vehicle_days.encounters import Concession
    routes={o:Route([[-40.,0.],[40.,0.]]) for o in ['A','B']}
    zone={'id':0,'a':'A','b':'B','ranges':{'A':[35.,50.],'B':[35.,50.]},'x':0.,'y':0.}
    driver=Driver(routes,{o:{'goal_s':75.} for o in routes},[zone],{'x':1000.,'y':1000.})
    sim=SimpleNamespace(time=0.,cars={
        'A':{'x':-20.,'y':0.,'heading_rad':0.,'speed_mps':2.,'status':'moving'},
        'B':{'x':-10.,'y':0.,'heading_rad':0.,'speed_mps':2.,'status':'moving'}})
    ordinary,_=driver.actions(sim)
    promise=Concession('B','A',0,35.,50.,0.,20.)
    actions,states=driver.actions(sim,promise)
    assert ordinary['B']['cruise_auto']['speed_mps']>0
    assert actions['B']['cruise_auto']['speed_mps']==0
    assert states['B'][:2]==('yielding','A')
    assert promise.status=='pending'

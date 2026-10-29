import copy
import math
from types import SimpleNamespace

import pytest

from vehicle_days.driving_autos import reference_actions, resolve_action
from vehicle_days.driving_roads import Network

CONFIG = SimpleNamespace(wheelbase=2.7, max_acceleration=2., max_braking=6., max_speed=13., max_steering_angle=.5)


def setup():
    network = Network({"roads": [{"id": "r", "points": [[0, 0], [200, 0]], "lanes": 1}]})
    car = {"id": "c", "x": 10., "y": 0., "heading_rad": 0., "speed_mps": 5., "steering_angle_rad": 0., "path": ["r:0"]}
    return network, car


def test_autos_control_only_their_axis_and_preserve_state():
    network, car = setup()
    before = copy.deepcopy(car)
    controls, metadata = resolve_action(car, {"steering_auto": {"path": "r:0"}, "acceleration_mps2": -4.}, network, CONFIG)
    assert controls == {"steering_angle_rad": 0., "acceleration_mps2": -4.}
    assert metadata["steering"]["auto"] == "path"
    controls, _ = resolve_action(car, {"cruise_auto": {"speed_mps": 8.}, "steering_angle_rad": .2}, network, CONFIG)
    assert controls == {"steering_angle_rad": .2, "acceleration_mps2": 2.}
    assert car == before


def test_pure_pursuit_turns_toward_path_without_changing_speed():
    network, car = setup()
    car["y"] = 1.
    controls, _ = resolve_action(car, {"steering_auto": {"path": ["r:0"]}}, network, CONFIG)
    assert -.5 <= controls["steering_angle_rad"] < 0
    assert controls["acceleration_mps2"] == 0
    car["y"] = -1.
    controls, _ = resolve_action(car, {"steering_auto": {"path": ["r:0"]}}, network, CONFIG)
    assert controls["steering_angle_rad"] > 0


def test_cruise_brakes_but_has_no_collision_avoidance():
    network, car = setup()
    controls, _ = resolve_action(car, {"cruise_auto": {"speed_mps": 0}}, network, CONFIG)
    assert controls["acceleration_mps2"] == -6
    controls, _ = resolve_action(car, {"cruise_auto": {"speed_mps": 10}}, network, CONFIG)
    assert controls["acceleration_mps2"] == 2
    assert controls["steering_angle_rad"] == 0


@pytest.mark.parametrize("action", [
    {"steering_auto": {"path": "r:0"}, "steering_angle_rad": 0},
    {"cruise_auto": {"speed_mps": 0}, "acceleration_mps2": 0},
    {"steering_auto": {"path": "missing"}}, {"steering_auto": {}},
    {"cruise_auto": {"speed_mps": -1}}, {"cruise_auto": {"speed_mps": 14}},
    {"acceleration_mps2": float("nan")}, {"steering_angle_rad": True},
    {"cruise_auto": {"speed_mps": math.inf}}, {"mystery": 3},
])
def test_bad_requests_fail_explicitly(action):
    network, car = setup()
    with pytest.raises(ValueError):
        resolve_action(car, action, network, CONFIG)


def test_disconnected_route_rejected():
    network, car = setup()
    network.paths["other"] = [[300, 0], [400, 0]]
    with pytest.raises(ValueError, match="connected"):
        resolve_action(car, {"steering_auto": {"path": ["r:0", "other"]}}, network, CONFIG)


def test_reference_driver_follows_and_stops_at_goal():
    network, car = setup()
    car.update(goal=[100, 0], cruise_speed_mps=6.)
    frame = {"time": 0, "cars": [car]}
    assert reference_actions(frame, network)["c"]["cruise_auto"]["speed_mps"] == 6
    leader = {**car, "id": "leader", "x": 18., "speed_mps": 0.}
    frame["cars"].append(leader)
    assert reference_actions(frame, network)["c"]["cruise_auto"]["speed_mps"] == 0
    frame["cars"] = [{**car, "x": 99}]
    assert reference_actions(frame, network)["c"]["cruise_auto"]["speed_mps"] == 0


def test_nested_unknown_options_rejected():
    network, car = setup()
    for action in ({"steering_auto": {"path": "r:0", "avoid_collisions": True}},
                   {"cruise_auto": {"speed_mps": 2, "avoid_collisions": True}}):
        with pytest.raises(ValueError):
            resolve_action(car, action, network, CONFIG)


def test_reference_signal_and_priority_decisions_are_driver_policy():
    from vehicle_days.driving_roads import make_driving_scenario
    scenario = make_driving_scenario("merge", 2)
    network = Network(scenario)
    main, ramp = copy.deepcopy(scenario["vehicles"])
    main.update(x=-40., y=0., speed_mps=6.)
    ramp.update(x=-60., y=-35., speed_mps=6.)
    actions = reference_actions({"time": 0, "cars": [main, ramp]}, network)
    assert actions[main["id"]]["cruise_auto"]["speed_mps"] == 6
    assert actions[ramp["id"]]["cruise_auto"]["speed_mps"] < 6
    # Cruise alone still accelerates when its requested speed exceeds current speed.
    controls, _ = resolve_action(ramp, {"cruise_auto": {"speed_mps": 8}}, network, CONFIG)
    assert controls["acceleration_mps2"] > 0
    network, car = setup()
    network.signals = [{"id": "signal", "path": "r:0", "s": 15., "green_seconds": 5., "red_seconds": 10., "offset": 0.}]
    assert reference_actions({"time": 6, "cars": [car]}, network)["c"]["cruise_auto"]["speed_mps"] == 0
    assert reference_actions({"time": 0, "cars": [car]}, network)["c"]["cruise_auto"]["speed_mps"] == 6


@pytest.mark.parametrize("name", ["boulevard", "junction", "merge", "curves", "loop"])
def test_cautious_driver_completes_actual_eight_car_journeys(name):
    from vehicle_days.driving import Simulation
    from vehicle_days.driving_roads import make_driving_scenario
    simulation = Simulation(make_driving_scenario(name, 8), record=False)
    while not simulation.termination:
        simulation.step(reference_actions(simulation.snapshot(), simulation.network))
    assert simulation.termination["kind"] == "completed", simulation.termination
    assert all(car["status"] == "arrived" and car["speed_mps"] == 0
               for car in simulation.cars.values())

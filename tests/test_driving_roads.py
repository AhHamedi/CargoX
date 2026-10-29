"""Geometry checks independent of vehicle integration."""
import copy
import math
from types import SimpleNamespace

import pytest

from vehicle_days.driving_roads import Network, SCENARIO_NAMES, make_driving_scenario, nearest_on_path, path_length, point_at

CONFIG = SimpleNamespace(length=4.5, width=1.8)


def scenario(points=None, lanes=2):
    return {"roads": [{"id": "r", "points": points or [[0, 0], [100, 0]], "lanes": lanes}], "connections": [], "signals": []}


def car(x=20, y=0, heading=0):
    return {"id": "c", "x": x, "y": y, "heading_rad": heading, "speed_mps": 0}


def test_arc_length_and_projection():
    points = [[0, 0], [3, 0], [3, 4]]
    assert path_length(points) == 7
    assert point_at(points, 5) == pytest.approx((3, 2, math.pi/2))
    assert point_at(points, -10) == (0, 0, 0)
    assert point_at(points, 10) == pytest.approx((3, 4, math.pi/2))
    assert nearest_on_path(points, 2, 1) == pytest.approx((2, 1, 1))
    assert nearest_on_path(points, 4, 2) == pytest.approx((5, -1, 1))


def test_lane_offsets_and_public_map_are_copies():
    network = Network(scenario())
    assert network.paths["r:0"] == [[0, 1.75], [100, 1.75]]
    assert network.paths["r:1"] == [[0, -1.75], [100, -1.75]]
    view = network.public_map()
    view["roads"][0]["points"][0][0] = -999
    assert network.roads[0]["points"][0][0] == 0
    observation = network.observe(car(25, -1.5), 0)
    assert observation["path"] == "r:1"
    assert observation["s"] == 25


def test_full_body_including_flat_road_ends():
    network = Network(scenario())
    assert network.contains_body(car(), CONFIG)
    assert not network.contains_body(car(1, 0), CONFIG)
    assert network.contains_body(car(y=2.5), CONFIG)
    assert not network.contains_body(car(y=2.7), CONFIG)
    assert not network.contains_body(car(y=3, heading=math.pi/4), CONFIG)


def test_body_cannot_bridge_gap_even_if_all_corners_on_roads():
    value = {"roads": [{"id": "a", "points": [[0, -1.5], [100, -1.5]], "lanes": 1, "lane_width": 1.5},
                       {"id": "b", "points": [[0, 1.5], [100, 1.5]], "lanes": 1, "lane_width": 1.5}]}
    config = SimpleNamespace(length=4.5, width=3.2)
    assert not Network(value).contains_body(car(), config)


def test_union_covers_body_across_touching_roads():
    value = {"roads": [{"id": "a", "points": [[0, 0], [20, 0]], "lanes": 1},
                       {"id": "b", "points": [[20, 0], [40, 0]], "lanes": 1}]}
    assert Network(value).contains_body(car(), CONFIG, tolerance=0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True])
def test_nonfinite_geometry_rejected(bad):
    value = scenario()
    value["roads"][0]["points"][1][0] = bad
    with pytest.raises(ValueError, match="finite"):
        Network(value)


def test_degenerate_or_ambiguous_network_rejected():
    for edit in (lambda v: v["roads"].append(copy.deepcopy(v["roads"][0])),
                 lambda v: v["roads"][0].update(lanes=0),
                 lambda v: v["roads"][0].update(points=[[0, 0], [0, 0]]),
                 lambda v: v["roads"][0].update(lane_width=-1)):
        value = scenario()
        edit(value)
        with pytest.raises(ValueError):
            Network(value)


def test_connection_and_signal_validation():
    value = scenario(lanes=1)
    value["roads"].append({"id": "s", "points": [[110, 0], [200, 0]], "lanes": 1})
    value["connections"] = [{"id": "join", "from": ["r", 0], "to": ["s", 0], "points": [[100, 0], [110, 0]]}]
    value["signals"] = [{"id": "stop", "road": "r", "lane": 0, "s": 90, "green_seconds": 5, "red_seconds": 10}]
    network = Network(value)
    assert network.contains_body(car(x=105), CONFIG)
    assert network.signal_states(4.99)[0]["state"] == "green"
    assert network.signal_states(5)[0]["state"] == "red"
    assert network.signal_states(15)[0]["state"] == "green"
    value["connections"][0]["points"][0][0] = 99
    with pytest.raises(ValueError, match="endpoints"):
        Network(value)
    value["connections"][0]["points"][0][0] = 100
    value["signals"][0]["s"] = 200
    with pytest.raises(ValueError, match="stop line"):
        Network(value)


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_fixtures_have_valid_spawn_clearance_and_determinism(name):
    value = make_driving_scenario(name, 64, 17)
    assert value == make_driving_scenario(name, 64, 17)
    network = Network(value)
    for vehicle in value["vehicles"]:
        assert network.contains_body(vehicle, CONFIG, tolerance=0), vehicle
        for other in value["vehicles"]:
            if vehicle["id"] != other["id"]:
                assert math.dist((vehicle["x"], vehicle["y"]), (other["x"], other["y"])) > 3.4
    if name == "loop":
        assert network.paths["ring:0"][0] == network.paths["ring:0"][-1]


def test_connection_rounds_intermediate_road_end_without_open_gap():
    value = {"roads": [{"id": "a", "points": [[0, 0], [100, 0]], "lanes": 1},
                       {"id": "b", "points": [[120, 1], [200, 1]], "lanes": 1}],
             "connections": [{"id": "turn", "from": ["a", 0], "to": ["b", 0],
                              "points": [[100, 0], [110, 1], [120, 1]]}]}
    assert Network(value).contains_body(car(x=99), CONFIG, tolerance=0)


def test_one_incoming_lane_can_split_into_multiple_connected_choices():
    value = {"roads": [{"id": "in", "points": [[0, 0], [50, 0]], "lanes": 1},
                       {"id": "left", "points": [[60, 10], [100, 50]], "lanes": 1},
                       {"id": "right", "points": [[60, -10], [100, -50]], "lanes": 1}],
             "connections": [{"id": "choose_left", "from": ["in", 0], "to": ["left", 0],
                              "points": [[50, 0], [60, 10]], "priority": 1},
                             {"id": "choose_right", "from": ["in", 0], "to": ["right", 0],
                              "points": [[50, 0], [60, -10]], "priority": 0}]}
    network = Network(value)
    assert network.path_meta["choose_left"]["from"] == network.path_meta["choose_right"]["from"]
    assert network.path_meta["choose_left"]["to"] != network.path_meta["choose_right"]["to"]
    assert network.public_map()["connections"][0]["priority"] == 1


def test_goal_must_fit_parked_body_and_connect_to_spawn_surface():
    network = Network(scenario(lanes=1))
    vehicle = {**car(), "goal": [80., 0.], "goal_radius": 3.}
    network.validate_goal(vehicle, CONFIG)
    # The destination point may be outside; its acceptance radius must include
    # an actual parked body on the road.
    network.validate_goal({**vehicle, "goal": [80., 3.5]}, CONFIG)
    network.validate_goal({**vehicle, "goal": [100., 0.]}, CONFIG)
    with pytest.raises(ValueError, match="parking pose"):
        network.validate_goal({**vehicle, "goal": [1e6, 1e6]}, CONFIG)
    with pytest.raises(ValueError, match="parking pose"):
        network.validate_goal({**vehicle, "goal": [100., 0.], "goal_radius": 1.}, CONFIG)
    value = scenario(lanes=1)
    value["roads"].append({"id": "island", "points": [[200., 0.], [300., 0.]], "lanes": 1})
    with pytest.raises(ValueError, match="reachable"):
        Network(value).validate_goal({**vehicle, "goal": [250., 0.]}, CONFIG)


def test_physical_crossing_connects_surfaces_without_legal_turn_connector():
    value = scenario(lanes=1)
    value["roads"].append({"id": "cross", "points": [[80., -50.], [80., 50.]], "lanes": 1})
    network = Network(value)
    network.validate_goal({**car(), "goal": [80., 30.], "goal_radius": 3.}, CONFIG)
    # Connectivity is cached across vehicle validation.
    components = network._components
    network.validate_goal({**car(), "goal": [80., -30.], "goal_radius": 3.}, CONFIG)
    assert network._components is components


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_all_generated_goals_have_connected_feasible_stopping_poses(name):
    value = make_driving_scenario(name, 64)
    network = Network(value)
    for vehicle in value["vehicles"]:
        network.validate_goal(vehicle, CONFIG)

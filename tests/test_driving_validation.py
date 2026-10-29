"""Independent oracle falsification, including a collision between frames."""
import copy
import math

import pytest

from vehicle_days.driving_audit import audit_recording, command_pose, constant_steer_pose, linear_contact_time, separation


def car(owner="a", x=0., y=0., **kwargs):
    return {"id": owner, "x": x, "y": y, "heading_rad": 0., "speed_mps": 0.,
            "steering_angle_rad": 0., **kwargs}


def stationary():
    cars = [car(), car("b", x=20.)]
    return {"frames": [{"time": 0., "cars": cars}, {"time": .1, "cars": copy.deepcopy(cars)}]}


def test_stationary_record_and_separating_axis_geometry():
    assert audit_recording(stationary())["passed"]
    assert separation(car(), car(x=4.5)) == pytest.approx(0.)
    assert separation(car(), car(x=4.6)) > 0
    assert separation(car(), car(x=1., heading_rad=math.pi/4)) < 0


@pytest.mark.parametrize("corruption,kind", [
    ("overlap", "unexplained overlap"), ("clock", "clock"), ("nan", "finite state"),
    ("acceleration", "acceleration bound"), ("steer_rate", "steering rate"),
    ("teleport", "position jump"), ("missing", "conservation"),
])
def test_oracle_detects_corrupted_recordings(corruption, kind):
    run = stationary()
    current = run["frames"][1]["cars"][0]
    if corruption == "overlap":
        run["frames"][1]["cars"][1]["x"] = 1.
    elif corruption == "clock":
        run["frames"][1]["time"] = .3
    elif corruption == "nan":
        current["x"] = float("nan")
    elif corruption == "acceleration":
        current["speed_mps"] = 1.
    elif corruption == "steer_rate":
        current["steering_angle_rad"] = .1
    elif corruption == "teleport":
        current["x"] = 5.
    else:
        run["frames"][1]["cars"].pop()
    report = audit_recording(run)
    assert not report["passed"]
    assert kind in {row["kind"] for row in report["examples"]}


def test_contact_claim_must_have_a_physical_witness():
    run = stationary()
    run["termination"] = {"kind": "collision", "participants": ["a", "b"]}
    # The broad phase can omit separated pairs, so terminal contact gets an
    # explicit witness check regardless of their distance.
    assert not audit_recording(run)["passed"]


def test_every_named_collision_participant_needs_contact():
    cars = [car(), car("b", x=4.5), car("c", x=30.)]
    run = {"frames": [{"time": 0., "cars": cars}],
           "termination": {"kind": "collision", "participants": ["a", "b", "c"]}}
    assert not audit_recording(run)["passed"]


def test_oracle_detects_a_collision_between_clear_endpoints():
    a = car(x=-.6, speed_mps=12.)
    b = car("b", y=-.6, speed_mps=12., heading_rad=math.pi/2)
    run = {"frames": [{"time": 0., "cars": [a, b]},
                      {"time": .1, "cars": [{**a, "x": .6}, {**b, "y": .6}]}]}
    report = audit_recording(run, length=.1, width=.1)
    assert not report["passed"]
    assert "missed intervening contact" in {row["kind"] for row in report["examples"]}


def test_closed_form_straight_braking_stops_partway():
    result = constant_steer_pose(car(speed_mps=3.), 1., acceleration=-6.)
    assert result["x"] == pytest.approx(.75)
    assert result["speed_mps"] == 0.


def test_exact_swept_sat_finds_crossing_and_rejects_close_parallel_miss():
    assert linear_contact_time(car(x=-10., speed_mps=8.),
                               car("b", y=-10., heading_rad=math.pi/2, speed_mps=8.), 4.) == pytest.approx(.85625)
    assert linear_contact_time(car(speed_mps=8.), car("b", x=10., y=1.80001, speed_mps=4.), 4.) is None


def test_closed_form_turn_matches_circle_with_sideslip_offset():
    initial = car(speed_mps=2., steering_angle_rad=.2)
    result = constant_steer_pose(initial, 1.)
    beta = math.atan(.5*math.tan(.2))
    radius = 2.7/(math.cos(beta)*math.tan(.2))
    center = (-radius*math.sin(beta), radius*math.cos(beta))
    assert math.dist(center, (result["x"], result["y"])) == pytest.approx(radius)
    assert result["heading_rad"] == pytest.approx(2./radius)


def test_oracle_rejects_frozen_position_despite_recorded_forward_speed():
    a = car(speed_mps=2., applied_action={"acceleration_mps2": 0., "steering_angle_rad": 0.})
    run = {"frames": [{"time": 0., "cars": [a]}, {"time": .1, "cars": [copy.deepcopy(a)]}]}
    report = audit_recording(run)
    assert not report["passed"]
    assert "command motion mismatch" in {row["kind"] for row in report["examples"]}


def test_oracle_detects_a_lateral_peak_between_safe_endpoints():
    initial = car(speed_mps=5.78196090786571, steering_angle_rad=.2394053029006593)
    control = {"acceleration_mps2": -6., "steering_angle_rad": .5}
    after = command_pose(initial, control, .02)
    after["applied_action"] = control
    run = {"frames": [{"time": 0., "cars": [initial]}, {"time": .02, "cars": [after]}]}
    report = audit_recording(run)
    assert not report["passed"]
    assert "intervening lateral bound" in {row["kind"] for row in report["examples"]}


@pytest.mark.parametrize("speed,steering,acceleration", [(0., 0., 2.), (3., 0., -6.), (2., .2, 0.)])
def test_engine_motion_agrees_with_independent_closed_form(speed, steering, acceleration):
    from vehicle_days.driving import Simulation

    initial = car(speed_mps=speed, steering_angle_rad=steering)
    scenario = {"version": 3, "horizon": 1.,
                "roads": [{"id": "r", "points": [[-100., 0.], [100., 0.]], "lane_width": 40.}],
                "vehicles": [initial]}
    sim = Simulation(scenario)
    while not sim.termination:
        sim.step({"a": {"acceleration_mps2": acceleration, "steering_angle_rad": steering}})
    expected = constant_steer_pose(initial, 1., acceleration)
    actual = sim.snapshot()["cars"][0]
    for key in ("x", "y", "heading_rad", "speed_mps"):
        assert actual[key] == pytest.approx(expected[key], abs=1e-8)
    assert audit_recording(sim.recording())["passed"]

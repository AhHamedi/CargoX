"""Analytical motion, risky controls and deterministic first-impact episodes."""
import math

import pytest

from vehicle_days.driving import Simulation, replay


def scene(cars=None, horizon=10., width=1000.):
    return {"version": 3, "horizon": horizon, "roads": [{"id": "road", "points": [[-100., 0.], [2000., 0.]],
            "lanes": 1, "lane_width": width, "speed_limit": 13.}], "connections": [], "signals": [],
            "vehicles": cars or [{"id": "A", "x": 0., "y": 0., "heading_rad": 0.}]}


def drive(sim, action, steps):
    for _ in range(steps):
        frame = sim.step({owner: action for owner in sim.cars})
        if frame["terminated"]:
            break
    return sim.cars["A"]


def test_straight_constant_acceleration_and_in_tick_stop():
    sim = Simulation(scene())
    car = drive(sim, {"acceleration_mps2": 2.}, 20)
    assert car["x"] == pytest.approx(4., abs=1e-9)
    assert car["speed_mps"] == pytest.approx(4.)
    car = drive(sim, {"acceleration_mps2": -6.}, 7)
    assert car["x"] == pytest.approx(4.+4**2/12, abs=1e-9)
    assert car["speed_mps"] == 0
    assert sim.termination is None


def test_constant_steering_exact_bicycle_arc():
    delta, v, duration = .1, 4., 2.
    spec = scene([{"id": "A", "x": 0., "y": 0., "heading_rad": 0., "speed_mps": v,
                   "steering_angle_rad": delta}])
    sim = Simulation(spec)
    car = drive(sim, {"steering_angle_rad": delta}, round(duration/.1))
    beta = math.atan(.5*math.tan(delta))
    k = math.cos(beta)*math.tan(delta)/2.7
    yaw = v*duration*k
    assert car["heading_rad"] == pytest.approx(yaw, abs=1e-9)
    assert car["x"] == pytest.approx((math.sin(beta+yaw)-math.sin(beta))/k, abs=1e-9)
    assert car["y"] == pytest.approx((math.cos(beta)-math.cos(beta+yaw))/k, abs=1e-9)


def test_stationary_steering_changes_wheels_only():
    sim = Simulation(scene())
    car = drive(sim, {"steering_angle_rad": .5}, 3)
    assert car["steering_angle_rad"] == pytest.approx(.15)
    assert (car["x"], car["y"], car["heading_rad"]) == (0., 0., 0.)


def test_wheels_reach_near_target_at_rate_then_hold():
    from vehicle_days.driving import _Motion, VehicleConfig
    car = {"x": 0., "y": 0., "heading_rad": 0., "speed_mps": 0.,
           "steering_angle_rad": .1, "cruise_cap_mps": 13.}
    motion = _Motion(car, {"acceleration_mps2": 0., "steering_angle_rad": .103}, .02, VehicleConfig())
    assert motion.at(.004)["steering_angle_rad"] == pytest.approx(.102)
    assert motion.at(.006)["steering_angle_rad"] == pytest.approx(.103)
    assert motion.at(.02)["steering_angle_rad"] == pytest.approx(.103)


def test_physical_limits_visible_but_no_automatic_following():
    sim = Simulation(scene([{"id": "A", "x": 0., "y": 0., "heading_rad": 0., "speed_mps": 10.},
                            {"id": "B", "x": 10., "y": 0., "heading_rad": 0.}]))
    for _ in range(10):
        result = sim.step({"A": {"acceleration_mps2": 20., "steering_angle_rad": 0.}})
        if result["terminated"]:
            break
    assert result["termination"]["kind"] == "collision"
    assert sim.cars["A"]["action_result"]["limited"]
    assert result["termination"]["time"] < .55
    assert sim.cars["A"]["speed_mps"] > 10.


@pytest.mark.parametrize("kind", ["rear", "head", "cross", "side", "graze"])
def test_first_contact_between_ticks(kind):
    if kind == "rear":
        cars = [{"id": "A", "x": 0., "y": 0., "heading_rad": 0., "speed_mps": 13.},
                {"id": "B", "x": 5., "y": 0., "heading_rad": 0.}]
        expected = .5/13
    elif kind == "head":
        cars = [{"id": "A", "x": 0., "y": 0., "heading_rad": 0., "speed_mps": 13.},
                {"id": "B", "x": 5., "y": 0., "heading_rad": math.pi, "speed_mps": 13.}]
        expected = .5/26
    elif kind == "cross":
        cars = [{"id": "A", "x": -4., "y": 0., "heading_rad": 0., "speed_mps": 13.},
                {"id": "B", "x": 0., "y": -4., "heading_rad": math.pi/2, "speed_mps": 13.}]
        expected = .85/13
    else:
        y = 1.8 if kind == "graze" else 1.7
        cars = [{"id": "A", "x": 0., "y": 0., "heading_rad": 0., "speed_mps": 13.},
                {"id": "B", "x": 5., "y": y, "heading_rad": 0.}]
        expected = .5/13
    sim = Simulation(scene(cars))
    frame = sim.step({})
    assert frame["termination"]["kind"] == "collision"
    assert frame["time"] == pytest.approx(expected, abs=1e-5)
    assert frame["termination"]["participants"] == ["A", "B"]
    with pytest.raises(RuntimeError, match="ended"):
        sim.step({})


def test_close_parallel_cars_do_not_collide():
    cars = [{"id": "A", "x": 0., "y": 0., "heading_rad": 0., "speed_mps": 13.},
            {"id": "B", "x": 0., "y": 1.801, "heading_rad": 0., "speed_mps": 13.}]
    sim = Simulation(scene(cars))
    drive(sim, {}, 10)
    assert sim.termination is None


def test_low_gap_is_not_contact():
    cars = [{"id": "A", "x": 0., "y": 0., "heading_rad": 0.},
            {"id": "B", "x": 4.51, "y": 0., "heading_rad": 0.}]
    sim = Simulation(scene(cars))
    sim.step({})
    assert sim.termination is None


def test_steer_too_fast_is_valid_driving_failure():
    sim = Simulation(scene([{"id": "A", "x": 0., "y": 0., "heading_rad": 0., "speed_mps": 13.}]))
    drive(sim, {"steering_angle_rad": .5}, 10)
    assert sim.termination["kind"] == "loss_of_control"
    assert sim.time < .2


def test_turning_demand_peak_during_braking_is_not_missed():
    spec = scene([{"id": "A", "x": 0., "y": 0., "heading_rad": 0.,
                   "speed_mps": 5.78196090786571, "steering_angle_rad": .2394053029006593}], horizon=.02)
    sim = Simulation(spec)
    frame = sim.step({"A": {"acceleration_mps2": -6., "steering_angle_rad": .5}})
    assert frame["termination"]["kind"] == "loss_of_control"
    assert 0 < frame["time"] < .01
    assert frame["cars"][0]["lateral_acceleration_mps2"] == pytest.approx(3., abs=1e-6)


def test_leaving_road_is_recorded():
    spec = scene(width=3.5)
    sim = Simulation(spec)
    drive(sim, {"acceleration_mps2": 1., "steering_angle_rad": .3}, 100)
    assert sim.termination["kind"] == "road_departure"


def test_reset_replay_and_order_independence():
    spec = scene([{"id": "B", "x": 0., "y": 10., "heading_rad": 0.},
                  {"id": "A", "x": 0., "y": 0., "heading_rad": 0.}], horizon=2.)
    a = Simulation(spec)
    drive(a, {"acceleration_mps2": 1.}, 20)
    b = Simulation({**spec, "vehicles": list(reversed(spec["vehicles"]))})
    drive(b, {"acceleration_mps2": 1.}, 20)
    assert a.recording() == b.recording() | {"scenario": a.scenario}
    recorded = a.recording()
    repeated = replay(recorded)
    for expected, actual in zip(recorded["frames"], repeated["frames"]):
        assert expected["time"] == actual["time"]
        assert expected["termination"] == actual["termination"]
        for left, right in zip(expected["cars"], actual["cars"]):
            assert {k: v for k, v in left.items() if k != "action_result"} == {k: v for k, v in right.items() if k != "action_result"}
    a.reset(spec)
    assert a.time == 0 and len(a.frames) == 1 and not a.actions


@pytest.mark.parametrize("change", ["overlap", "nonfinite", "outside", "duplicate"])
def test_bad_spawns_rejected(change):
    spec = scene()
    if change == "overlap":
        spec["vehicles"].append({**spec["vehicles"][0], "id": "B"})
    elif change == "nonfinite":
        spec["vehicles"][0]["x"] = float("nan")
    elif change == "outside":
        spec["vehicles"][0]["x"] = -1000.
    else:
        spec["vehicles"] *= 2
    with pytest.raises(ValueError):
        Simulation(spec)


def test_observations_hide_peer_private_fields():
    cars = [{"id": "A", "x": 0., "y": 0., "heading_rad": 0., "private": {"secret": 2}},
            {"id": "B", "x": 10., "y": 0., "heading_rad": 0., "private": {"secret": 3}}]
    sim = Simulation(scene(cars))
    observed = sim.observations()["A"]
    assert "private" not in observed["nearby"][0]
    observed["self"]["x"] = 200.
    assert sim.cars["A"]["x"] == 0.


def test_invalid_or_ambiguous_action_is_controller_fault():
    sim = Simulation(scene())
    frame = sim.step({"A": {"acceleration_mps2": 1., "cruise_auto": {"speed_mps": 5.}}})
    assert frame["termination"]["kind"] == "controller_fault"
    assert frame["time"] == 0.


def test_convergence_with_steering_slew():
    runs = []
    for step in (.02, .01):
        sim = Simulation(scene(), physics_step=step)
        car = drive(sim, {"acceleration_mps2": 1., "steering_angle_rad": .1}, 30)
        runs.append(car)
    assert math.dist((runs[0]["x"], runs[0]["y"]), (runs[1]["x"], runs[1]["y"])) < .001


def test_signals_and_speed_limits_are_observed_not_enforced():
    spec = scene([{"id": "A", "x": 0., "y": 0., "heading_rad": 0., "speed_mps": 10.}])
    spec["roads"][0]["speed_limit"] = 5.
    spec["signals"] = [{"id": "red", "road": "road", "lane": 0, "s": 103.,
                        "green_seconds": 1., "red_seconds": 10., "offset": 2.}]
    sim = Simulation(spec)
    car = drive(sim, {}, 10)
    assert car["speed_mps"] == 10.
    assert {event["kind"] for event in sim.events} >= {"speed_limit_violation", "red_light_violation"}
    assert sim.termination is None


def test_goal_completion_requires_stop_and_preserves_body():
    spec = scene()
    spec["vehicles"][0].update(goal=[0., 0.])
    sim = Simulation(spec)
    frame = sim.step({})
    assert frame["termination"]["kind"] == "completed"
    assert frame["cars"][0]["x"] == 0.
    assert frame["cars"][0]["speed_mps"] == 0.


def test_angled_body_does_not_cross_signal_early():
    spec = scene([{"id": "A", "x": -22.3, "y": 0., "heading_rad": math.pi/3,
                   "speed_mps": 13.}], horizon=.02, width=100.)
    spec["signals"] = [{"id": "red", "road": "road", "lane": 0, "s": 80.,
                        "green_seconds": 1., "red_seconds": 10., "offset": 2.}]
    sim = Simulation(spec)
    sim.step({})
    assert not any(event["kind"] == "red_light_violation" for event in sim.events)

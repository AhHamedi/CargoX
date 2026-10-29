"""Independent recording checks; intentionally imports no simulator helpers.

The sampled contact check can falsify missed impacts; it is not a continuous
collision proof. Analytical fixtures and timestep comparisons complement it.
"""
from __future__ import annotations

import math
from itertools import combinations


def body(car, length=4.5, width=1.8):
    angle = car["heading_rad"]
    c, s = math.cos(angle), math.sin(angle)
    return [(car["x"]+a*length*c/2-b*width*s/2,
             car["y"]+a*length*s/2+b*width*c/2)
            for a, b in ((-1, -1), (-1, 1), (1, 1), (1, -1))]


def separation(first, second, length=4.5, width=1.8):
    """Largest signed separating-axis gap in metres; negative means overlap."""
    polygons = body(first, length, width), body(second, length, width)
    gaps = []
    for polygon in polygons:
        for index in (0, 1):
            a, b = polygon[index:index+2]
            dx, dy = b[0]-a[0], b[1]-a[1]
            norm = math.hypot(dx, dy)
            axis = -dy/norm, dx/norm
            p, q = [[x*axis[0]+y*axis[1] for x, y in row] for row in polygons]
            gaps.append(max(min(q)-max(p), min(p)-max(q)))
    return max(gaps)


def linear_contact_time(first, second, horizon, length=4.5, width=1.8):
    """Exact swept SAT for fixed-orientation constant-speed analytical fixtures."""
    polygons = body(first, length, width), body(second, length, width)
    velocity = [(c["speed_mps"]*math.cos(c["heading_rad"]),
                 c["speed_mps"]*math.sin(c["heading_rad"])) for c in (first, second)]
    enter, leave = 0., horizon
    for polygon in polygons:
        for index in (0, 1):
            a, b = polygon[index:index+2]
            axis = a[1]-b[1], b[0]-a[0]
            p, q = [[x*axis[0]+y*axis[1] for x, y in row] for row in polygons]
            relative = ((velocity[1][0]-velocity[0][0])*axis[0]
                        +(velocity[1][1]-velocity[0][1])*axis[1])
            low, high = min(p)-max(q), max(p)-min(q)
            if abs(relative) < 1e-12:
                if low > 1e-9 or high < -1e-9:
                    return None
            else:
                t0, t1 = sorted((low/relative, high/relative))
                enter, leave = max(enter, t0), min(leave, t1)
                if enter > leave+1e-10:
                    return None
    return max(0., enter) if enter <= horizon else None


def interpolate(before, after, fraction):
    """Cubic position estimate using endpoint velocity, shortest yaw rotation."""
    elapsed = after["time"]-before["time"]
    t, t2, t3 = fraction, fraction**2, fraction**3
    value = dict(before)
    for axis, trig in (("x", math.cos), ("y", math.sin)):
        def velocity(car):
            beta = math.atan(.5*math.tan(car.get("steering_angle_rad", 0.)))
            return car["speed_mps"]*trig(car["heading_rad"]+beta)
        value[axis] = ((2*t3-3*t2+1)*before[axis]+(t3-2*t2+t)*elapsed*velocity(before)
                       +(-2*t3+3*t2)*after[axis]+(t3-t2)*elapsed*velocity(after))
    turn = math.remainder(after["heading_rad"]-before["heading_rad"], 2*math.pi)
    value["heading_rad"] = before["heading_rad"]+fraction*turn
    return value


def constant_steer_pose(car, duration, acceleration=0., wheelbase=2.7):
    """Closed-form centre-of-axles bicycle solution, including a mid-step stop."""
    speed, steer = car["speed_mps"], car["steering_angle_rad"]
    moving = min(duration, speed/-acceleration) if acceleration < 0 else duration
    distance = speed*moving+.5*acceleration*moving*moving
    beta = math.atan(.5*math.tan(steer))
    curvature = math.cos(beta)*math.tan(steer)/wheelbase
    start = car["heading_rad"]+beta
    turn = curvature*distance
    if abs(curvature) < 1e-12:
        dx, dy = distance*math.cos(start), distance*math.sin(start)
    else:
        dx = (math.sin(start+turn)-math.sin(start))/curvature
        dy = (math.cos(start)-math.cos(start+turn))/curvature
    return {**car, "x": car["x"]+dx, "y": car["y"]+dy,
            "heading_rad": car["heading_rad"]+turn,
            "speed_mps": max(0., speed+acceleration*moving)}


def command_pose(car, control, duration, *, wheelbase=2.7, max_speed=13., max_steering_rate=.5):
    """Independent fine midpoint ODE integration of a held low-level command."""
    state = dict(car)
    acceleration = control["acceleration_mps2"]
    target = control["steering_angle_rad"]
    cap = min(max_speed, car.get("cruise_cap_mps", max_speed))
    def lateral(value):
        tangent = math.tan(value["steering_angle_rad"])
        return value["speed_mps"]**2*abs(tangent)/(wheelbase*math.sqrt(1+.25*tangent*tangent))
    peak_lateral = lateral(state)
    steps = max(1, math.ceil(duration/.005))
    step = duration/steps
    for _ in range(steps):
        v, steer = state["speed_mps"], state["steering_angle_rad"]
        active = step
        if acceleration < 0:
            active = min(step, v/-acceleration)
        elif acceleration > 0:
            active = min(step, (cap-v)/acceleration)
        nv = min(cap, max(0., v+acceleration*active))
        distance = v*active+.5*acceleration*active**2+nv*(step-active)
        change = max(-max_steering_rate*step, min(max_steering_rate*step, target-steer))
        beta = math.atan(.5*math.tan(steer+change/2))
        turn = distance*math.cos(beta)*math.tan(steer+change/2)/wheelbase
        bearing = state["heading_rad"]+beta+turn/2
        state["x"] += distance*math.cos(bearing)
        state["y"] += distance*math.sin(bearing)
        state["heading_rad"] += turn
        state["speed_mps"], state["steering_angle_rad"] = nv, steer+change
        peak_lateral = max(peak_lateral, lateral(state))
    state["_audit_peak_lateral_mps2"] = peak_lateral
    return state


def audit_recording(recording, *, length=4.5, width=1.8, wheelbase=2.7,
                    max_speed=13., max_acceleration=2., max_braking=6.,
                    max_steering=.5, max_steering_rate=.5, max_lateral=3.):
    """Check state conservation, physical bounds and independently sampled bodies."""
    examples, violations, pair_checks = [], 0, 0
    frames = recording.get("frames", [])
    expected = {car["id"] for car in frames[0]["cars"]} if frames else set()

    def check(passed, kind, **context):
        nonlocal violations
        if not passed:
            violations += 1
            if len(examples) < 30:
                examples.append({"kind": kind, **context})

    check(bool(frames), "empty recording")
    termination = recording.get("termination") or {}
    participants = set(termination.get("participants", []))
    previous = None
    for index, frame in enumerate(frames):
        now = frame["time"]
        cars = {car["id"]: car for car in frame["cars"]}
        check(len(cars) == len(frame["cars"]) and set(cars) == expected, "conservation", time=now)
        check(math.isfinite(now), "clock", time=now)
        elapsed = now-previous["time"] if previous else 0.
        if previous:
            zero_terminal = index == len(frames)-1 and termination.get("kind") == "controller_fault"
            check((elapsed > 0 or zero_terminal and elapsed == 0) and elapsed <= .100001, "clock", time=now)
        old = {car["id"]: car for car in previous["cars"]} if previous else {}
        usable = {}
        for owner, car in cars.items():
            finite = all(isinstance(car.get(field), (float, int)) and math.isfinite(car[field])
                         for field in ("x", "y", "heading_rad", "speed_mps", "steering_angle_rad"))
            check(finite, "finite state", time=now, owner=owner)
            if not finite:
                continue
            usable[owner] = car
            v, steering = car["speed_mps"], car["steering_angle_rad"]
            check(-1e-8 <= v <= max_speed+1e-6, "speed bound", time=now, owner=owner)
            check(abs(steering) <= max_steering+1e-6, "steering bound", time=now, owner=owner)
            beta = math.atan(.5*math.tan(steering))
            lateral = v*v*abs(math.cos(beta)*math.tan(steering)/wheelbase)
            terminal_loss = index == len(frames)-1 and termination.get("kind") == "loss_of_control"
            check(lateral <= max_lateral+1e-5 or terminal_loss, "lateral bound", time=now, owner=owner)
            if owner not in old or elapsed <= 0:
                continue
            prior = old[owner]
            if not all(math.isfinite(prior[field]) for field in ("x", "y", "heading_rad", "speed_mps", "steering_angle_rad")):
                continue
            dv = v-prior["speed_mps"]
            check(-max_braking*elapsed-1e-6 <= dv <= max_acceleration*elapsed+1e-6,
                  "acceleration bound", time=now, owner=owner)
            check(abs(steering-prior["steering_angle_rad"]) <= max_steering_rate*elapsed+1e-6,
                  "steering rate", time=now, owner=owner)
            travel_bound = max(v, prior["speed_mps"])*elapsed+.5*max_acceleration*elapsed**2
            check(math.dist((car["x"], car["y"]), (prior["x"], prior["y"])) <= travel_bound+1e-5,
                  "position jump", time=now, owner=owner)
            yaw_bound = max(v, prior["speed_mps"])*math.tan(max_steering)/wheelbase*elapsed
            check(abs(math.remainder(car["heading_rad"]-prior["heading_rad"], 2*math.pi)) <= yaw_bound+1e-5,
                  "heading jump", time=now, owner=owner)
            control = car.get("applied_action")
            if control and prior.get("status") != "arrived":
                predicted = command_pose(prior, control, elapsed, wheelbase=wheelbase,
                                         max_speed=max_speed, max_steering_rate=max_steering_rate)
                residual = max(math.dist((car["x"], car["y"]), (predicted["x"], predicted["y"])),
                               abs(math.remainder(car["heading_rad"]-predicted["heading_rad"], 2*math.pi)),
                               abs(v-predicted["speed_mps"]))
                check(residual <= .003, "command motion mismatch", time=now, owner=owner, residual=residual)
                check(predicted["_audit_peak_lateral_mps2"] <= max_lateral+1e-5,
                      "intervening lateral bound", time=now, owner=owner,
                      peak_lateral_mps2=predicted["_audit_peak_lateral_mps2"])
        for first, second in combinations(usable.values(), 2):
            # Fast broad phase is independent of the engine's spatial index.
            reach = math.hypot(length, width)+2*max_speed*max(0., elapsed)
            if math.dist((first["x"], first["y"]), (second["x"], second["y"])) > reach:
                continue
            pair = {first["id"], second["id"]}
            terminal_contact = index == len(frames)-1 and termination.get("kind") == "collision" and pair <= participants
            gap = separation(first, second, length, width)
            pair_checks += 1
            check(gap >= -.001 or terminal_contact, "unexplained overlap", time=now, participants=sorted(pair))
            if terminal_contact:
                check(abs(gap) <= .04, "collision without contact", time=now, separation_m=gap)
            if previous and elapsed > 0 and pair <= old.keys():
                for sample in range(1, 10):
                    a = interpolate({**old[first["id"]], "time": previous["time"]}, {**first, "time": now}, sample/10)
                    b = interpolate({**old[second["id"]], "time": previous["time"]}, {**second, "time": now}, sample/10)
                    check(separation(a, b, length, width) >= -.002, "missed intervening contact",
                          time=previous["time"]+elapsed*sample/10, participants=sorted(pair))
        previous = frame
    if termination.get("kind") == "collision":
        check(len(participants) >= 2 and participants <= expected, "collision participants")
        final = {car["id"]: car for car in frames[-1]["cars"]} if frames else {}
        contacts = [(a, b) for a, b in combinations(sorted(participants & final.keys()), 2)
                    if abs(separation(final[a], final[b], length, width)) <= .04]
        check(bool(contacts) and participants <= {owner for pair in contacts for owner in pair},
              "collision without contact")
    return {"passed": violations == 0, "violations": violations, "examples": examples,
            "frames_checked": len(frames), "pair_checks": pair_checks,
            "limitation": "Between-frame contact screening is sampled, not a continuous proof."}

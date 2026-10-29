"""Free planar vehicle motion. Controllers drive; the engine supplies no driver.

The bicycle reference point is midway between the axles (and body centre).
Each integration segment uses its midpoint steering angle, exact constant-
acceleration distance, and an exact circular arc. Collision queries use that
same trajectory, rather than interpolating the recorded control frames.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math

from .driving_roads import Network, point_at
from .driving_autos import resolve_action


@dataclass(frozen=True)
class VehicleConfig:
    length: float = 4.5
    width: float = 1.8
    wheelbase: float = 2.7
    max_acceleration: float = 2.0
    max_braking: float = 6.0
    max_speed: float = 13.0
    max_steering_angle: float = 0.5
    max_steering_rate: float = 0.5
    max_lateral_acceleration: float = 3.0

    def __post_init__(self):
        if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Vehicle parameters must be finite positive numbers")
        if self.wheelbase > self.length or self.max_steering_angle >= math.pi/2:
            raise ValueError("Invalid wheelbase or steering angle")


CONTROL_SECONDS = 0.1
CONTACT_EPS = 1e-8


def _number(value, name):
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _clamp(value, low, high):
    return max(low, min(high, value))


def _angle(value):
    return (value+math.pi) % (2*math.pi)-math.pi


def body_vertices(car, config):
    c, s = math.cos(car["heading_rad"]), math.sin(car["heading_rad"])
    return [(car["x"]+i*config.length/2*c-j*config.width/2*s,
             car["y"]+i*config.length/2*s+j*config.width/2*c)
            for i, j in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def separation(first, second, config):
    """Signed SAT separation; positive is a proven separating-axis gap."""
    delta = second["x"]-first["x"], second["y"]-first["y"]
    vectors = [(math.cos(c["heading_rad"]), math.sin(c["heading_rad"])) for c in (first, second)]
    result = -math.inf
    for u in vectors:
        for axis in (u, (-u[1], u[0])):
            radii = [config.length/2*abs(v[0]*axis[0]+v[1]*axis[1])+
                     config.width/2*abs(-v[1]*axis[0]+v[0]*axis[1]) for v in vectors]
            result = max(result, abs(delta[0]*axis[0]+delta[1]*axis[1])-sum(radii))
    return result


class _Motion:
    """Authoritative trajectory within one numerical segment."""
    def __init__(self, car, control, duration, config):
        self.car, self.duration, self.config = car, duration, config
        self.acceleration = control["acceleration_mps2"]
        self.target = control["steering_angle_rad"]
        difference = self.target-car["steering_angle_rad"]
        self.steer_rate = math.copysign(config.max_steering_rate, difference) if difference else 0.
        self.slew_seconds = abs(difference)/config.max_steering_rate
        middle = self.steering_at(duration/2)
        self.beta = math.atan(0.5*math.tan(middle))
        self.curvature = math.cos(self.beta)*math.tan(middle)/config.wheelbase
        self.cap = min(config.max_speed, car["cruise_cap_mps"])
        self.vmax = max(car["speed_mps"], self.at(duration)["speed_mps"])
        self.bound = self.vmax*(1+math.hypot(config.length, config.width)/2*abs(self.curvature))

    def steering_at(self, seconds):
        return self.car["steering_angle_rad"]+self.steer_rate*min(seconds, self.slew_seconds)

    def lateral_limit_time(self):
        """First curvature-demand crossing, including interior braking peaks.

        On intervals without a zero-speed, speed-cap or zero-steering crossing,
        v²|k(delta)| is log-concave: d(log f)/dt = 2a/v + rate*k'/k is strictly
        decreasing. Thus it has at most one interior maximum, found by a scalar
        derivative root. A cheap endpoint upper bound avoids this work normally.
        """
        limit = self.config.max_lateral_acceleration+1e-9
        rate, initial = self.steer_rate, self.car["steering_angle_rad"]
        curvature = lambda d: math.tan(d)/math.sqrt(1+.25*math.tan(d)**2)/self.config.wheelbase
        if self.vmax**2*max(abs(curvature(initial)), abs(curvature(self.steering_at(self.duration)))) <= limit:
            return None
        def speed(t):
            return _clamp(self.car["speed_mps"]+self.acceleration*t, 0., self.cap)
        def value(t):
            return speed(t)**2*abs(curvature(self.steering_at(t)))
        cuts = [0., self.duration]
        if 0 < self.slew_seconds < self.duration:
            cuts.append(self.slew_seconds)
        if rate and 0 < -initial/rate < self.duration:
            cuts.append(-initial/rate)
        if self.acceleration:
            for v in (0., self.cap):
                t = (v-self.car["speed_mps"])/self.acceleration
                if 0 < t < self.duration:
                    cuts.append(t)
        cuts = sorted(cuts)
        for start, end in zip(cuts, cuts[1:]):
            if value(start) > limit:
                return start
            middle_speed = self.car["speed_mps"]+self.acceleration*(start+end)/2
            acceleration = self.acceleration if 0 < middle_speed < self.cap else 0.
            interval_rate = rate if (start+end)/2 < self.slew_seconds else 0.
            sign = 1 if self.steering_at((start+end)/2) >= 0 else -1
            def derivative(t):
                tangent = math.tan(self.steering_at(t))
                k = tangent/math.sqrt(1+.25*tangent*tangent)/self.config.wheelbase
                dk = (1+tangent*tangent)/(1+.25*tangent*tangent)**1.5/self.config.wheelbase
                v = speed(t)
                return sign*(2*v*acceleration*k+v*v*interval_rate*dk)
            peak = end if value(end) >= value(start) else start
            inset = (end-start)*1e-8
            if derivative(start+inset) > 0 and derivative(end-inset) < 0:
                low, high = start, end
                for _ in range(30):
                    mid = (low+high)/2
                    if derivative(mid) > 0:
                        low = mid
                    else:
                        high = mid
                peak = (low+high)/2
            if value(peak) > limit:
                low, high = start, peak
                for _ in range(30):
                    mid = (low+high)/2
                    if value(mid) > limit:
                        high = mid
                    else:
                        low = mid
                return high
        return None

    def at(self, seconds):
        car, a, v = self.car, self.acceleration, self.car["speed_mps"]
        active = seconds
        if a < 0:
            active = min(seconds, v/-a)
        elif a > 0:
            active = min(seconds, max(0., (self.cap-v)/a))
        nv = _clamp(v+a*active, 0., self.cap)
        distance = v*active+a*active*active/2+nv*(seconds-active)
        yaw = self.curvature*distance
        steering = self.steering_at(seconds)
        beta = math.atan(.5*math.tan(steering))
        bearing = car["heading_rad"]+self.beta
        if abs(self.curvature) < 1e-10:
            dx, dy = distance*math.cos(bearing), distance*math.sin(bearing)
        else:
            dx = (math.sin(bearing+yaw)-math.sin(bearing))/self.curvature
            dy = (math.cos(bearing)-math.cos(bearing+yaw))/self.curvature
        return {**car, "x": car["x"]+dx, "y": car["y"]+dy,
                "heading_rad": _angle(car["heading_rad"]+yaw), "speed_mps": nv,
                "steering_angle_rad": steering,
                "acceleration_mps2": (nv-v)/seconds if seconds else 0.,
                "lateral_acceleration_mps2": nv*nv*math.cos(beta)*math.tan(steering)/self.config.wheelbase,
                "distance_m": car.get("distance_m", 0.)+distance}


def _contact(first, second, duration, config):
    """Conservative advancement with a rigid-body point-speed bound.

    A separating-axis gap bounds how far either body must move before contact.
    The sum of maximum point speeds therefore bounds a safe time advance. This
    remains valid with rotation and catches between-frame pass-through impacts.
    """
    t = 0.
    for _ in range(160):
        a, b = first.at(t), second.at(t)
        axes = []
        for car in (a, b):
            u = math.cos(car["heading_rad"]), math.sin(car["heading_rad"])
            axes.extend((u, (-u[1], u[0])))
        va = a["speed_mps"]*math.cos(a["heading_rad"]+first.beta), a["speed_mps"]*math.sin(a["heading_rad"]+first.beta)
        vb = b["speed_mps"]*math.cos(b["heading_rad"]+second.beta), b["speed_mps"]*math.sin(b["heading_rad"]+second.beta)
        remaining = duration-t
        rotation = math.hypot(config.length, config.width)/2*(first.vmax*abs(first.curvature)+second.vmax*abs(second.curvature))
        velocity_change = (abs(first.acceleration)+abs(second.acceleration)+
                           first.vmax**2*abs(first.curvature)+second.vmax**2*abs(second.curvature))*remaining
        safe, gap = 0., -math.inf
        for axis in axes:
            radius = 0.
            for car in (a, b):
                c, s = math.cos(car["heading_rad"]), math.sin(car["heading_rad"])
                radius += config.length/2*abs(c*axis[0]+s*axis[1])+config.width/2*abs(-s*axis[0]+c*axis[1])
            space = abs((b["x"]-a["x"])*axis[0]+(b["y"]-a["y"])*axis[1])-radius
            gap = max(gap, space)
            if space > CONTACT_EPS:
                bound = abs((vb[0]-va[0])*axis[0]+(vb[1]-va[1])*axis[1])+velocity_change+rotation
                if bound < 1e-12 or space > bound*remaining+CONTACT_EPS:
                    return None
                safe = max(safe, space/bound)
        if gap <= CONTACT_EPS:
            return t
        advance = safe*0.95
        if advance < 1e-10:
            return t
        t += advance
        if t > duration:
            return None
    raise RuntimeError("Contact detection did not converge; run is invalid")


class Simulation:
    def __init__(self, scenario, physics_step=0.02, record=True):
        self.physics_step = _number(physics_step, "physics_step")
        if not 0 < self.physics_step <= 0.02:
            raise ValueError("physics_step must be in (0, 0.02]")
        self.record = record
        self.reset(scenario)

    @property
    def time(self):
        return self._time

    def reset(self, scenario):
        self.scenario = copy.deepcopy(scenario)
        if scenario.get("version") != 3:
            raise ValueError("Free driving requires scenario version 3")
        self.config = VehicleConfig(**scenario.get("vehicle_config", {}))
        self.horizon = _number(scenario.get("horizon", 60.), "horizon")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        self.observation_radius = _number(scenario.get("observation_radius_m", 110.), "observation_radius_m")
        if self.observation_radius <= 0:
            raise ValueError("observation_radius_m must be positive")
        self.network = Network(self.scenario)
        self._minimum_speed_limit = min(meta["speed_limit"] for meta in self.network.path_meta.values())
        self.cars = {}
        for original in scenario.get("vehicles", []):
            car = copy.deepcopy(original)
            owner = car.get("id")
            if not isinstance(owner, str) or not owner or owner in self.cars:
                raise ValueError("Vehicle IDs must be distinct nonempty strings")
            for name in ("x", "y", "heading_rad"):
                car[name] = _number(car.get(name), name)
            for name, default in (("speed_mps", 0.), ("steering_angle_rad", 0.),
                                  ("cruise_cap_mps", self.config.max_speed),
                                  ("goal_radius", 3.), ("goal_speed_mps", .5)):
                car[name] = _number(car.get(name, default), name)
            if (not 0 < car["cruise_cap_mps"] <= self.config.max_speed or
                    not 0 <= car["speed_mps"] <= car["cruise_cap_mps"] or
                    abs(car["steering_angle_rad"]) > self.config.max_steering_angle or
                    car["goal_radius"] <= 0 or car["goal_speed_mps"] < 0):
                raise ValueError("Invalid initial vehicle state")
            if "goal" in car:
                if not isinstance(car["goal"], list) or len(car["goal"]) != 2:
                    raise ValueError("goal must be [x, y]")
                car["goal"] = [_number(v, "goal") for v in car["goal"]]
            car.update(acceleration_mps2=0., lateral_acceleration_mps2=0., distance_m=0.,
                       status="moving", applied_action={}, action_result={},
                       length=self.config.length, width=self.config.width)
            beta = math.atan(.5*math.tan(car["steering_angle_rad"]))
            if car["speed_mps"]**2*abs(math.cos(beta)*math.tan(car["steering_angle_rad"])/self.config.wheelbase) > self.config.max_lateral_acceleration:
                raise ValueError("Initial state exceeds the lateral validity limit")
            if not self.network.contains_body(car, self.config):
                raise ValueError(f"Vehicle {owner} spawns outside the road")
            if "goal" in car:
                self.network.validate_goal(car, self.config)
            for peer in self.cars.values():
                if separation(car, peer, self.config) <= CONTACT_EPS:
                    raise ValueError("Initial vehicle bodies touch or overlap")
            self.cars[owner] = car
        if not self.cars:
            raise ValueError("A scenario needs at least one vehicle")
        self.cars = dict(sorted(self.cars.items()))
        self._time, self.termination = 0., None
        self.events, self._step_events, self.frames, self.actions = [], [], [], []
        self._speeding = set()
        if self.record:
            self.frames.append(self.snapshot())
        return self.snapshot()

    def snapshot(self):
        return {"version": 3, "time": round(self._time, 10),
                "cars": copy.deepcopy(list(self.cars.values())),
                "events": copy.deepcopy(self._step_events),
                "terminated": self.termination is not None, "termination": copy.deepcopy(self.termination)}

    def observations(self):
        public = ("id", "x", "y", "heading_rad", "speed_mps", "steering_angle_rad", "status")
        return {owner: {"time": self.time, "self": copy.deepcopy(car),
                        "physics": asdict(self.config), "controller_seconds": CONTROL_SECONDS,
                        "map": self.network.public_map(),
                        "road": self.network.observe(car, self.time),
                        "nearby": [{k: other[k] for k in public} for other in self.cars.values()
                                   if other["id"] != owner and math.hypot(other["x"]-car["x"], other["y"]-car["y"]) <= self.observation_radius]}
                for owner, car in self.cars.items()}

    def _event(self, kind, **details):
        event = {"kind": kind, "time": round(self.time, 10), **details}
        self.events.append(event)
        self._step_events.append(event)
        return event

    def _end(self, kind, participants=(), **details):
        self.termination = self._event(kind, participants=list(participants), **details)

    def _candidates(self, motions, duration):
        """Spatial broad phase, with swept bounding circles for rotating bodies."""
        radius = math.hypot(self.config.length, self.config.width)/2
        size, cells, pairs = 12., {}, set()
        for owner, motion in motions.items():
            car = motion.car
            reach = radius+motion.bound*duration
            for ix in range(math.floor((car["x"]-reach)/size), math.floor((car["x"]+reach)/size)+1):
                for iy in range(math.floor((car["y"]-reach)/size), math.floor((car["y"]+reach)/size)+1):
                    cell = cells.setdefault((ix, iy), [])
                    for peer in cell:
                        pairs.add(tuple(sorted((peer, owner))))
                    cell.append(owner)
        return sorted(pairs)

    def _advance(self, controls, duration):
        before, start_time = self.cars, self.time
        motions = {owner: _Motion(car, controls[owner] if car["status"] != "arrived" else
                                  {"acceleration_mps2": 0., "steering_angle_rad": car["steering_angle_rad"]},
                                  duration, self.config) for owner, car in self.cars.items()}
        stop, cause, owners = duration, None, []
        for first, second in self._candidates(motions, duration):
            contact = _contact(motions[first], motions[second], duration, self.config)
            if contact is not None and contact <= stop:
                stop, cause, owners = contact, "collision", [first, second]
        for owner, motion in motions.items():
            loss = motion.lateral_limit_time()
            if loss is not None and (loss < stop-1e-8 or cause is None):
                stop, cause, owners = loss, "loss_of_control", [owner]
            end = motion.at(stop)
            if not self.network.contains_body(end, self.config):
                low, high = 0., stop
                for _ in range(22):
                    mid = (low+high)/2
                    if not self.network.contains_body(motion.at(mid), self.config):
                        high = mid
                    else:
                        low = mid
                if high < stop-1e-8 or cause is None:
                    stop, cause, owners = high, "road_departure", [owner]
        self.cars = {owner: motion.at(stop) for owner, motion in motions.items()}
        self._time += stop
        for owner, car in self.cars.items():
            self._traffic_events(before[owner], car, start_time, stop, motions[owner])
        if cause:
            if cause == "collision":
                # Include all bodies making contact at the first impact instant.
                contacts = [(a, b) for a, b in self._candidates(motions, stop)
                            if separation(self.cars[a], self.cars[b], self.config) <= CONTACT_EPS*4]
                owners = sorted({x for pair in contacts for x in pair} | set(owners))
                for owner in owners:
                    self.cars[owner]["status"] = "collided"
                self._end(cause, owners, contacts=[list(pair) for pair in contacts],
                          poses=[copy.deepcopy(self.cars[o]) for o in owners])
            else:
                self.cars[owners[0]]["status"] = cause
                self._end(cause, owners)
            return
        for owner, car in self.cars.items():
            goal = car.get("goal")
            if goal and car["status"] != "arrived" and car["speed_mps"] <= min(1e-7, car["goal_speed_mps"]) and math.dist((car["x"], car["y"]), goal) <= car["goal_radius"]:
                car["status"], car["arrival"] = "arrived", self.time
                self._event("arrived", agent_id=owner)
        if all(car["status"] == "arrived" for car in self.cars.values()):
            self._end("completed", self.cars)

    def _traffic_events(self, previous, car, start_time, elapsed, motion):
        # These are observations, never speed governors or signal enforcement.
        owner = car["id"]
        road = self.network.observe(car, self.time) if car["speed_mps"] > self._minimum_speed_limit+1e-6 else None
        speeding = road is not None and car["speed_mps"] > road["speed_limit_mps"]+1e-6
        if speeding and owner not in self._speeding:
            self._speeding.add(owner)
            self._event("speed_limit_violation", agent_id=owner, speed_mps=car["speed_mps"],
                        limit_mps=road["speed_limit_mps"], path=road["path"])
        elif not speeding:
            self._speeding.discard(owner)
        for signal in self.network.signals:
            points = self.network.paths[signal["path"]]
            x, y, heading = point_at(points, signal["s"])
            c, s = math.cos(heading), math.sin(heading)
            half_width = self.network.path_meta[signal["path"]]["width"]/2
            def front(pose):
                # Clip the oriented body to the stop line's finite lane strip.
                polygon = [((px-x)*c+(py-y)*s, -(px-x)*s+(py-y)*c)
                           for px, py in body_vertices(pose, self.config)]
                for sign in (-1, 1):
                    clipped = []
                    for p, q in zip(polygon, polygon[1:]+polygon[:1]):
                        pin, qin = sign*p[1] <= half_width, sign*q[1] <= half_width
                        if pin:
                            clipped.append(p)
                        if pin != qin:
                            f = (sign*half_width-p[1])/(q[1]-p[1])
                            clipped.append((p[0]+f*(q[0]-p[0]), sign*half_width))
                    polygon = clipped
                return max((p[0] for p in polygon), default=-math.inf)
            if front(previous) <= 0 < front(car):
                low, high = 0., elapsed
                for _ in range(12):
                    mid = (low+high)/2
                    if front(motion.at(mid)) > 0:
                        high = mid
                    else:
                        low = mid
                when = start_time+high
                red = (when+signal["offset"]) % (signal["green_seconds"]+signal["red_seconds"]) >= signal["green_seconds"]
                if red:
                    self._event("red_light_violation", agent_id=owner, signal=signal["id"], crossing_time=when)

    def step(self, actions):
        if self.termination:
            raise RuntimeError("Episode has ended; reset before stepping")
        self._step_events = []
        if not isinstance(actions, dict) or set(actions)-set(self.cars):
            raise ValueError("actions must map known vehicle IDs to requests")
        controls, outcomes = {}, {}
        # Resolve all controls before changing any vehicle state.
        for owner, car in self.cars.items():
            try:
                request = actions.get(owner, {})
                control, metadata = resolve_action(copy.deepcopy(car), request, self.network, self.config)
                acceleration = _number(control["acceleration_mps2"], "acceleration_mps2")
                steering = _number(control["steering_angle_rad"], "steering_angle_rad")
                applied = {"acceleration_mps2": _clamp(acceleration, -self.config.max_braking, self.config.max_acceleration),
                           "steering_angle_rad": _clamp(steering, -self.config.max_steering_angle, self.config.max_steering_angle)}
                controls[owner] = applied
                outcomes[owner] = {"requested": copy.deepcopy(request), "generated": control, "applied": applied,
                                   "assistants": metadata, "limited": control != applied}
            except (ValueError, TypeError, KeyError) as exc:
                self._end("controller_fault", [owner], message=str(exc))
                if self.record:
                    self.actions.append({"time": self.time, "requested": copy.deepcopy(actions), "fault": str(exc)})
                    self.frames.append(self.snapshot())
                return self.snapshot()
        for owner, result in outcomes.items():
            self.cars[owner]["applied_action"] = copy.deepcopy(controls[owner])
            self.cars[owner]["action_result"] = result
        if self.record:
            self.actions.append({"time": self.time, "requested": copy.deepcopy(actions), "applied": copy.deepcopy(controls)})
        end = min(self.horizon, round(self.time+CONTROL_SECONDS, 10))
        while self.time < end-1e-10 and not self.termination:
            self._advance(controls, min(self.physics_step, end-self.time))
        if not self.termination and self.time >= self.horizon-1e-9:
            self._end("horizon")
        if self.record:
            self.frames.append(self.snapshot())
        return self.snapshot()

    def recording(self):
        if not self.record:
            raise ValueError("Recording was disabled for this episode")
        return copy.deepcopy({"version": 3, "model": "kinematic-bicycle", "scenario": self.scenario,
                              "config": asdict(self.config), "physics_step": self.physics_step,
                              "frames": self.frames, "actions": self.actions, "events": self.events,
                              "termination": self.termination, "training_ready": False})


def replay(recording):
    """Re-execute applied low-level commands, independent of assistant code."""
    sim = Simulation(recording["scenario"], physics_step=recording["physics_step"])
    for row in recording["actions"]:
        sim.step(row.get("applied", row["requested"]))
    return sim.recording()

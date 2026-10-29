"""Optional control-axis assistants and an explicit, model-free reference driver."""
from __future__ import annotations

import math

from .driving_roads import nearest_on_path, path_length, point_at


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _route(target, network):
    if not isinstance(target, dict) or set(target) != {"path"}:
        raise ValueError("steering_auto accepts only path")
    ids = target.get("path") if isinstance(target, dict) else None
    if isinstance(ids, str):
        ids = [ids]
    if not isinstance(ids, list) or not ids or any(not isinstance(pid, str) or pid not in network.paths for pid in ids):
        raise ValueError("steering_auto.path must name one or more existing paths")
    points = []
    for pid in ids:
        path = network.paths[pid]
        if points and math.dist(points[-1], path[0]) > .05:
            raise ValueError("steering_auto paths must be connected in order")
        points.extend(path[1:] if points and math.dist(points[-1], path[0]) < 1e-9 else path)
    return ids, points


def resolve_action(car, request, network, config):
    """Resolve each axis independently. Assistants never change vehicle state."""
    if not isinstance(request, dict):
        raise ValueError("an action must be an object")
    unknown = set(request)-{"acceleration_mps2", "steering_angle_rad", "steering_auto", "cruise_auto"}
    if unknown:
        raise ValueError(f"unknown action fields: {', '.join(sorted(unknown))}")
    if "steering_auto" in request and "steering_angle_rad" in request:
        raise ValueError("steering axis cannot have direct and auto owners")
    if "cruise_auto" in request and "acceleration_mps2" in request:
        raise ValueError("acceleration axis cannot have direct and auto owners")
    controls = {"acceleration_mps2": _number(request.get("acceleration_mps2", 0.), "acceleration"),
                "steering_angle_rad": _number(request.get("steering_angle_rad", 0.), "steering angle")}
    metadata = {"steering": "direct", "acceleration": "direct"}
    if "steering_auto" in request:
        ids, points = _route(request["steering_auto"], network)
        s, _, _ = nearest_on_path(points, car["x"], car["y"])
        lookahead = max(5., car["speed_mps"]*.9)
        target_s, length = s+lookahead, path_length(points)
        if math.dist(points[0], points[-1]) < .05:
            target_s %= length
        tx, ty, _ = point_at(points, target_s)
        # Pure pursuit is defined at the rear axle; state position is axle midpoint.
        rear_x = car["x"]-.5*config.wheelbase*math.cos(car["heading_rad"])
        rear_y = car["y"]-.5*config.wheelbase*math.sin(car["heading_rad"])
        dx, dy = tx-rear_x, ty-rear_y
        lateral = -math.sin(car["heading_rad"])*dx+math.cos(car["heading_rad"])*dy
        steering = math.atan2(2*config.wheelbase*lateral, max(1e-9, dx*dx+dy*dy))
        controls["steering_angle_rad"] = max(-config.max_steering_angle, min(config.max_steering_angle, steering))
        metadata["steering"] = {"auto": "path", "path": ids, "target": [tx, ty]}
    if "cruise_auto" in request:
        target = request["cruise_auto"]
        if not isinstance(target, dict) or set(target) != {"speed_mps"}:
            raise ValueError("cruise_auto needs speed_mps")
        speed = _number(target["speed_mps"], "cruise speed")
        if speed < 0 or speed > config.max_speed:
            raise ValueError(f"cruise speed must be between 0 and {config.max_speed} m/s")
        controls["acceleration_mps2"] = max(-config.max_braking, min(config.max_acceleration, 1.5*(speed-car["speed_mps"])))
        metadata["acceleration"] = {"auto": "cruise", "speed_mps": speed}
    return controls, metadata


def reference_actions(frame, network, vehicles=None):
    """A cautious scripted driver, separate from the assistants and physics.

    This driver chooses to follow speed limits, leave gaps, stop at signals and
    stop near its goal. Steering/cruise alone do none of those things.
    """
    cars = frame.get("cars", [])
    templates = {v["id"]: v for v in (vehicles or [])}
    result = {}
    for public in cars:
        car = {**templates.get(public["id"], {}), **public}
        if car.get("status", "moving") not in ("moving", "active"):
            continue
        route = car.get("path") or [network.observe(car, frame.get("time", 0.))["path"]]
        _, points = _route({"path": route}, network)
        s, _, _ = nearest_on_path(points, car["x"], car["y"])
        speed = min(car.get("cruise_speed_mps", 6.), min(network.path_meta[pid]["speed_limit"] for pid in route))
        goal = car.get("goal")
        if goal:
            goal_s, _, _ = nearest_on_path(points, goal[0], goal[1])
            remaining = goal_s-s
            if math.dist(points[0], points[-1]) < .05 and remaining < -1.:
                remaining += path_length(points)
            speed = min(speed, math.sqrt(2*1.2*max(0., remaining-1.5)))
        # Path-based following also works on bends and across connection boundaries.
        for other in cars:
            if other["id"] == car["id"]:
                continue
            other_s, _, distance = nearest_on_path(points, other["x"], other["y"])
            gap = other_s-s-4.5
            if distance < 1.4 and gap > -2.:
                speed = min(speed, max(0., other.get("speed_mps", 0.)+.6*(gap-5.)))
        # Yield before a lower-priority merge, as an explicit driver decision.
        # This never participates in physics or admission of a manoeuvre.
        connection_s = 0.
        for pid in route:
            mine = network.path_meta[pid]
            if "to" in mine:
                for other in cars:
                    peer = {**templates.get(other["id"], {}), **other}
                    if peer["id"] == car["id"] or peer.get("status", "moving") not in ("moving", "active"):
                        continue
                    peer_route = peer.get("path", [])
                    peer_connection_s = 0.
                    for peer_pid in peer_route:
                        theirs = network.path_meta[peer_pid]
                        yields = (theirs.get("priority", 0), peer_pid) > (mine.get("priority", 0), pid)
                        if peer_pid != pid and theirs.get("to") == mine["to"] and yields:
                            _, peer_points = _route({"path": peer_route}, network)
                            peer_s = nearest_on_path(peer_points, peer["x"], peer["y"])[0]
                            peer_remaining = peer_connection_s+path_length(network.paths[peer_pid])-peer_s
                            if -12. < peer_remaining < 100. and s < connection_s+path_length(network.paths[pid]):
                                speed = min(speed, math.sqrt(2*1.2*max(0., connection_s-s-5.)))
                        peer_connection_s += path_length(network.paths[peer_pid])
            connection_s += path_length(network.paths[pid])
        offset = 0.
        for pid in route:
            for signal in network.signal_states(frame.get("time", 0.)):
                if signal["path"] == pid and signal["state"] == "red":
                    remaining = offset+signal["s"]-s
                    if remaining > -2.25:
                        speed = min(speed, math.sqrt(2*1.2*max(0., remaining-5.)))
            offset += path_length(network.paths[pid])
        result[car["id"]] = {"steering_auto": {"path": route}, "cruise_auto": {"speed_mps": speed}}
        if goal and math.dist((car["x"], car["y"]), goal) < 3. and car["speed_mps"] < .3:
            result[car["id"]].pop("cruise_auto")
            result[car["id"]]["acceleration_mps2"] = -3.
    return result

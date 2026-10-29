"""Road geometry for free-moving cars. Lanes describe paths, never constrain motion."""
from __future__ import annotations

import copy
import math
import random

SCENARIO_NAMES = ("boulevard", "junction", "merge", "curves", "loop")
_EPS = 1e-9


def path_length(points):
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


def nearest_on_path(points, x, y):
    """Return distance along path, signed left offset, and Euclidean distance."""
    best, progress = (0., 0., math.inf), 0.
    for a, b in zip(points, points[1:]):
        dx, dy = b[0]-a[0], b[1]-a[1]
        length = math.hypot(dx, dy)
        u = max(0., min(1., ((x-a[0])*dx+(y-a[1])*dy)/(length*length)))
        ex, ey = x-a[0]-u*dx, y-a[1]-u*dy
        distance = math.hypot(ex, ey)
        if distance < best[2]:
            best = (progress+u*length, (dx*ey-dy*ex)/length, distance)
        progress += length
    return best


def point_at(points, s):
    s = max(0., s)
    for index, (a, b) in enumerate(zip(points, points[1:])):
        dx, dy = b[0]-a[0], b[1]-a[1]
        length = math.hypot(dx, dy)
        if s <= length or index == len(points)-2:
            u = min(1., s/length)
            return a[0]+u*dx, a[1]+u*dy, math.atan2(dy, dx)
        s -= length
    raise ValueError("a path needs two distinct points")


def _offset(points, offset):
    normals = []
    for a, b in zip(points, points[1:]):
        dx, dy = b[0]-a[0], b[1]-a[1]
        length = math.hypot(dx, dy)
        normals.append((-dy/length, dx/length))
    result = []
    for index, p in enumerate(points):
        first = normals[max(0, index-1)]
        second = normals[min(index, len(normals)-1)]
        if math.dist(points[0], points[-1]) < _EPS and index in (0, len(points)-1):
            first, second = normals[-1], normals[0]
        nx, ny = first[0]+second[0], first[1]+second[1]
        denominator = nx*second[0]+ny*second[1]
        if denominator < .02:
            raise ValueError("road doubles back too sharply; split it into roads and a connector")
        result.append([p[0]+offset*nx/denominator, p[1]+offset*ny/denominator])
    return result


def _points(value, label):
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"{label} needs at least two points")
    result = []
    for p in value:
        if (not isinstance(p, (list, tuple)) or len(p) != 2
                or any(isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) for n in p)):
            raise ValueError(f"{label} points must be finite coordinate pairs")
        result.append([float(p[0]), float(p[1])])
    if any(math.dist(a, b) <= _EPS for a, b in zip(result, result[1:])):
        raise ValueError(f"{label} has a zero-length segment")
    return result


def _positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _cross(a, b, p):
    return (b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])


def _clip(poly, a, b, inside):
    """Clip a convex polygon against a line's inside or outside half-plane."""
    result = []
    for p, q in zip(poly, poly[1:]+poly[:1]):
        cp, cq = _cross(a, b, p), _cross(a, b, q)
        pin, qin = (cp >= -_EPS, cq >= -_EPS) if inside else (cp <= _EPS, cq <= _EPS)
        if pin:
            result.append(p)
        if pin != qin:
            u = cp/(cp-cq)
            result.append((p[0]+u*(q[0]-p[0]), p[1]+u*(q[1]-p[1])))
    return result


def _area(poly):
    return abs(sum(p[0]*q[1]-q[0]*p[1] for p, q in zip(poly, poly[1:]+poly[:1])))/2 if poly else 0.


def _subtract(poly, cutter):
    fragments = []
    for a, b in zip(cutter, cutter[1:]+cutter[:1]):
        outside = _clip(poly, a, b, False)
        if _area(outside) > 1e-8:
            fragments.append(outside)
        poly = _clip(poly, a, b, True)
        if not poly:
            break
    return fragments


def _strict_clip(poly, a, b, inside, margin):
    """Half-plane clip for certificates, with an inward distance margin."""
    offset = margin*math.hypot(b[0]-a[0], b[1]-a[1])
    result = []
    for p, q in zip(poly, poly[1:]+poly[:1]):
        cp, cq = _cross(a,b,p)-offset, _cross(a,b,q)-offset
        pin, qin = (cp >= 0., cq >= 0.) if inside else (cp <= 0., cq <= 0.)
        if pin:
            result.append(p)
        if pin != qin:
            u = cp/(cp-cq)
            result.append((p[0]+u*(q[0]-p[0]),p[1]+u*(q[1]-p[1])))
    return result


def _positive_area(poly):
    # A translated triangle fan avoids global-coordinate shoelace cancellation.
    # Convex pieces have consistently oriented triangles; retain every positive
    # area, however small. Unlike _subtract, certificates discard no area sliver.
    if len(poly)<3:
        return False
    x,y = poly[0]
    return any(abs((a[0]-x)*(b[1]-y)-(a[1]-y)*(b[0]-x))>0.
               for a,b in zip(poly[1:],poly[2:]))


def _strict_subtract(poly, cutter, margin, budget=None):
    fragments = []
    for a,b in zip(cutter,cutter[1:]+cutter[:1]):
        if budget is not None:
            budget[0] -= 2*len(poly)
            if budget[0]<0:
                return None
        outside = _strict_clip(poly,a,b,False,margin)
        if _positive_area(outside):
            fragments.append(outside)
        poly = _strict_clip(poly,a,b,True,margin)
        if not poly:
            break
    return fragments


def _intersects(first, second):
    for poly in (first, second):
        for a, b in zip(poly, poly[1:]+poly[:1]):
            axis = b[1]-a[1], a[0]-b[0]
            p = [x*axis[0]+y*axis[1] for x, y in first]
            q = [x*axis[0]+y*axis[1] for x, y in second]
            if max(p) < min(q)-_EPS or max(q) < min(p)-_EPS:
                return False
    return True


class Network:
    def __init__(self, scenario):
        self.roads, self.connections, self.signals = [], [], []
        self.paths, self.path_meta = {}, {}
        self._surfaces, self._grid = [], {}
        self._body_surface_cache = {}
        self._body_pose_cache = {}
        self._body_patch_cache, self._body_patch_attempt = {}, {}
        self._surface_groups, self._group_count, self._components = [], 0, None
        road_ids = set()
        for raw in scenario.get("roads", []):
            road = copy.deepcopy(raw)
            rid = road.get("id")
            if not isinstance(rid, str) or not rid or ":" in rid or rid in road_ids:
                raise ValueError("road IDs must be unique nonempty strings without ':'")
            road_ids.add(rid)
            lanes = road.get("lanes", 1)
            if isinstance(lanes, bool) or not isinstance(lanes, int) or lanes < 1:
                raise ValueError(f"road {rid} lanes must be a positive integer")
            road.update(points=_points(road.get("points"), f"road {rid}"), lanes=lanes,
                        lane_width=_positive(road.get("lane_width", 3.5), "lane width"),
                        speed_limit=_positive(road.get("speed_limit", 13.), "speed limit"))
            for lane in range(lanes):
                pid = f"{rid}:{lane}"
                self.paths[pid] = _offset(road["points"], ((lanes-1)/2-lane)*road["lane_width"])
                self.path_meta[pid] = {"road": rid, "lane": lane, "width": road["lane_width"],
                                       "speed_limit": road["speed_limit"]}
            self.roads.append(road)
            self._add_surface(road["points"], lanes*road["lane_width"])
        if not self.roads:
            raise ValueError("a network needs at least one road")
        for raw in scenario.get("connections", []):
            connector = copy.deepcopy(raw)
            cid = connector.get("id")
            if not isinstance(cid, str) or not cid or cid in self.paths or cid in road_ids:
                raise ValueError("connection IDs must be unique nonempty strings")
            endpoints = []
            for field in ("from", "to"):
                value = connector.get(field)
                if (not isinstance(value, (list, tuple)) or len(value) != 2
                        or isinstance(value[1], bool) or not isinstance(value[1], int)):
                    raise ValueError(f"connection {cid} needs {field} [road, lane]")
                pid = f"{value[0]}:{value[1]}"
                if pid not in self.paths:
                    raise ValueError(f"connection {cid} references unknown lane {pid}")
                endpoints.append(pid)
            connector["points"] = _points(connector.get("points"), f"connection {cid}")
            if (math.dist(connector["points"][0], self.paths[endpoints[0]][-1]) > .05
                    or math.dist(connector["points"][-1], self.paths[endpoints[1]][0]) > .05):
                raise ValueError(f"connection {cid} endpoints must meet their lanes within 0.05 m")
            connector["width"] = _positive(connector.get("width", min(self.path_meta[p]["width"] for p in endpoints)), "connection width")
            priority = connector.get("priority", 0)
            if isinstance(priority, bool) or not isinstance(priority, int):
                raise ValueError("connection priority must be an integer; larger values have precedence")
            connector["priority"] = priority
            self.paths[cid] = connector["points"]
            self.path_meta[cid] = {"from": endpoints[0], "to": endpoints[1], "width": connector["width"],
                                   "speed_limit": min(self.path_meta[p]["speed_limit"] for p in endpoints), "priority": priority}
            self.connections.append(connector)
            self._add_surface(connector["points"], connector["width"], connected_ends=True)
        seen = set()
        for raw in scenario.get("signals", []):
            signal = copy.deepcopy(raw)
            sid = signal.get("id")
            pid = f"{signal.get('road')}:{signal.get('lane')}"
            if not isinstance(sid, str) or not sid or sid in seen or pid not in self.paths:
                raise ValueError("signals need unique IDs and an existing road lane")
            seen.add(sid)
            for name in ("green_seconds", "red_seconds"):
                signal[name] = _positive(signal.get(name), name)
            for name in ("s", "offset"):
                value = signal.get(name, 0.)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"signal {name} must be finite")
                signal[name] = float(value)
            if not 0 <= signal["s"] <= path_length(self.paths[pid]):
                raise ValueError("signal stop line must lie on its road lane")
            signal["path"] = pid
            self.signals.append(signal)
        self._certificate_margin = max(1e-7,64*max(math.ulp(v)
            for poly,_ in self._surfaces for point in poly for v in point))

    def _add_polygon(self, poly):
        bounds = (min(p[0] for p in poly), min(p[1] for p in poly),
                  max(p[0] for p in poly), max(p[1] for p in poly))
        index = len(self._surfaces)
        self._surfaces.append((poly, bounds))
        for gx in range(math.floor(bounds[0]/20), math.floor(bounds[2]/20)+1):
            for gy in range(math.floor(bounds[1]/20), math.floor(bounds[3]/20)+1):
                self._grid.setdefault((gx, gy), []).append(index)

    def _add_surface(self, points, width, connected_ends=False):
        start = len(self._surfaces)
        half = width/2
        for a, b in zip(points, points[1:]):
            dx, dy = b[0]-a[0], b[1]-a[1]
            length = math.hypot(dx, dy)
            nx, ny = -dy/length*half, dx/length*half
            self._add_polygon([(a[0]-nx, a[1]-ny), (b[0]-nx, b[1]-ny),
                               (b[0]+nx, b[1]+ny), (a[0]+nx, a[1]+ny)])
        # Circumscribed 32-gons bound the rounding error below 0.5% of half-width.
        # Straight segments keep flat ends; only interior joins are rounded.
        radius = half/math.cos(math.pi/32)
        joins = points if connected_ends else points[1:-1]+points[:1] if math.dist(points[0], points[-1]) < _EPS else points[1:-1]
        for x, y in joins:
            self._add_polygon([(x+radius*math.cos(i*math.tau/32), y+radius*math.sin(i*math.tau/32)) for i in range(32)])
        # All strips of one path meet at their vertices. Start connectivity
        # from these known groups rather than comparing every curve polygon.
        self._surface_groups.extend([self._group_count]*(len(self._surfaces)-start))
        self._group_count += 1

    def _surface_components(self):
        if self._components is None:
            parents = list(range(self._group_count))
            def find(group):
                while parents[group] != group:
                    parents[group] = parents[parents[group]]
                    group = parents[group]
                return group
            for indices in self._grid.values():
                for offset, first in enumerate(indices):
                    for second in indices[offset+1:]:
                        a, b = find(self._surface_groups[first]), find(self._surface_groups[second])
                        if a == b:
                            continue
                        p, bp = self._surfaces[first]
                        q, bq = self._surfaces[second]
                        if bp[0] > bq[2] or bp[2] < bq[0] or bp[1] > bq[3] or bp[3] < bq[1]:
                            continue
                        if _intersects(p, q):
                            parents[b] = a
            self._components = [find(group) for group in self._surface_groups]
        return self._components

    def _components_at(self, x, y):
        components = self._surface_components()
        return {components[index] for index in self._grid.get((math.floor(x/20), math.floor(y/20)), ())
                if all(_cross(a, b, (x, y)) >= -_EPS for a, b in zip(self._surfaces[index][0], self._surfaces[index][0][1:]+self._surfaces[index][0][:1]))}

    def validate_goal(self, car, config):
        """Require a tangent-aligned parking pose on the same physical surface.

        This checks surface connectivity, not a proof that a finite-size car
        can negotiate every narrow passage. Driving remains the agent's job.
        """
        if "goal" not in car:
            return
        gx, gy = car["goal"]
        radius = car["goal_radius"]
        origin = self._components_at(car["x"], car["y"])
        for points in self.paths.values():
            s = nearest_on_path(points, gx, gy)[0]
            for shift in (0., -radius, radius, -config.length/2, config.length/2):
                x, y, heading = point_at(points, s+shift)
                distance = math.hypot(x-gx, y-gy)
                candidates = [(x, y), (gx, gy)]
                if distance > radius:
                    candidates.append((gx+(x-gx)*radius/distance, gy+(y-gy)*radius/distance))
                for px, py in candidates:
                    if (math.hypot(px-gx, py-gy) <= radius+_EPS
                            and origin & self._components_at(px, py)
                            and self.contains_body({"x": px, "y": py, "heading_rad": heading}, config, tolerance=0)):
                        return
        raise ValueError(f"Vehicle {car['id']} goal has no reachable tangent-aligned parking pose within its radius")

    def contains_body(self, car, config, tolerance=.05):
        owner = car.get("id")
        if not isinstance(owner, str):
            owner = None
        pose = (car["x"], car["y"], car["heading_rad"], config.length, config.width, tolerance)
        if owner is not None and self._body_pose_cache.get(owner) == pose:
            return True
        # The reduced rectangle is within tolerance of every full-body point.
        # This is conservative at oblique boundaries, rather than granting a
        # larger diagonal tolerance at body corners.
        inset = tolerance/math.sqrt(2)
        hl, hw = max(0., config.length/2-inset), max(0., config.width/2-inset)
        c, s = math.cos(car["heading_rad"]), math.sin(car["heading_rad"])
        body = [(car["x"]+u*hl*c-v*hw*s, car["y"]+u*hl*s+v*hw*c)
                for u, v in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        bounds = (min(p[0] for p in body), min(p[1] for p in body), max(p[0] for p in body), max(p[1] for p in body))
        patch = self._body_patch_cache.get(owner) if owner is not None else None
        if patch is not None:
            polygon, margin = patch
            if all(_cross(a,b,p)>margin*math.hypot(b[0]-a[0],b[1]-a[1])
                   for a,b in zip(polygon,polygon[1:]+polygon[:1]) for p in body):
                return self._remember_body_pose(owner,pose)
        # A cached surface is only a candidate positive certificate. Recheck
        # every current corner against it; leaving that surface always falls
        # through to the unchanged full road-union containment test below.
        hint = self._body_surface_cache.get(owner) if owner is not None else None
        if hint is not None:
            poly, box = self._surfaces[hint]
            if not (box[0] > bounds[2] or box[2] < bounds[0] or box[1] > bounds[3] or box[3] < bounds[1]) and all(
                    _cross(a, b, p) >= -_EPS for a, b in zip(poly, poly[1:]+poly[:1]) for p in body):
                return self._remember_body_pose(owner, pose)
        candidates = set()
        for gx in range(math.floor(bounds[0]/20), math.floor(bounds[2]/20)+1):
            for gy in range(math.floor(bounds[1]/20), math.floor(bounds[3]/20)+1):
                candidates.update(self._grid.get((gx, gy), ()))
        surfaces = []
        for index in sorted(candidates):
            poly, box = self._surfaces[index]
            if box[0] > bounds[2] or box[2] < bounds[0] or box[1] > bounds[3] or box[3] < bounds[1]:
                continue
            if all(_cross(a, b, p) >= -_EPS for a, b in zip(poly, poly[1:]+poly[:1]) for p in body):
                if owner is not None:
                    self._body_surface_cache[owner] = index
                return self._remember_body_pose(owner, pose)
            surfaces.append(poly)
        uncovered = [body]
        for surface in surfaces:
            uncovered = [piece for poly in uncovered for piece in _subtract(poly, surface)]
            if not uncovered:
                if owner is not None:
                    self._certify_body_patch(owner,car,hl,hw,c,s)
                return self._remember_body_pose(owner, pose)
        return False

    def _remember_body_pose(self, owner, pose):
        # Network surfaces are immutable after construction. An identical pose,
        # body size and tolerance therefore has exactly the same positive result.
        if owner is not None:
            self._body_pose_cache[owner] = pose
        return True

    def _certify_body_patch(self, owner, car, hl, hw, cosine, sine):
        previous = self._body_patch_attempt.get(owner)
        at = (car['x'],car['y'],car['heading_rad'])
        if previous is not None and math.hypot(at[0]-previous[0],at[1]-previous[1])<.5 and abs(
                math.atan2(math.sin(at[2]-previous[2]),math.cos(at[2]-previous[2])))<.1:
            return
        self._body_patch_attempt[owner] = at
        # This larger convex rectangle grants no road tolerance. It is accepted
        # only when fully covered by inward-shrunk road polygons; later reduced
        # body vertices must lie strictly inside that proven pavement patch.
        hl,hw = hl+.5,hw+.5
        patch = [(at[0]+u*hl*cosine-v*hw*sine,at[1]+u*hl*sine+v*hw*cosine)
                 for u,v in ((-1,-1),(1,-1),(1,1),(-1,1))]
        bounds = (min(p[0] for p in patch),min(p[1] for p in patch),max(p[0] for p in patch),max(p[1] for p in patch))
        margin = max(self._certificate_margin,64*max(math.ulp(v) for p in patch for v in p))
        candidates = set()
        for gx in range(math.floor(bounds[0]/20),math.floor(bounds[2]/20)+1):
            for gy in range(math.floor(bounds[1]/20),math.floor(bounds[3]/20)+1):
                candidates.update(self._grid.get((gx,gy),()))
        uncovered = [patch]
        budget = [32768]
        for index in sorted(candidates):
            polygon,box = self._surfaces[index]
            if box[0]>bounds[2] or box[2]<bounds[0] or box[1]>bounds[3] or box[3]<bounds[1]:
                continue
            pieces = []
            for poly in uncovered:
                remainder = _strict_subtract(poly,polygon,margin,budget)
                # A certificate is optional. Decline expensive/fragmented proofs
                # rather than dropping a sliver or delaying the driving loop.
                if remainder is None:
                    return
                pieces.extend(remainder)
                if len(pieces)>64:
                    return
            uncovered = pieces
            if not uncovered:
                self._body_patch_cache[owner] = (patch,margin)
                return

    def signal_states(self, time):
        return [{**signal, "state": "green" if (time+signal["offset"]) % (signal["green_seconds"]+signal["red_seconds"]) < signal["green_seconds"] else "red"}
                for signal in self.signals]

    def observe(self, car, time):
        pid, nearest = min(((pid, nearest_on_path(points, car["x"], car["y"])) for pid, points in self.paths.items()), key=lambda item: item[1][2])
        return {"path": pid, "road": self.path_meta[pid].get("road"), "lane": self.path_meta[pid].get("lane"),
                "s": nearest[0], "lateral_offset_m": nearest[1], "path_distance_m": nearest[2],
                "speed_limit_mps": self.path_meta[pid]["speed_limit"], "signals": self.signal_states(time)}

    def public_map(self):
        return copy.deepcopy({"roads": self.roads, "connections": self.connections,
                              "signals": self.signals, "paths": self.paths, "path_meta": self.path_meta})


def make_driving_scenario(name="boulevard", count=8, seed=17):
    if name not in SCENARIO_NAMES:
        raise ValueError(f"unknown scenario {name!r}; choose from {SCENARIO_NAMES}")
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 64:
        raise ValueError("vehicle count must be between 1 and 64")
    rng = random.Random(seed)
    scenario = {"version": 3, "name": name, "seed": seed, "horizon": 180., "roads": [], "connections": [], "signals": [], "vehicles": []}
    def road(rid, points, lanes=1, width=3.5):
        scenario["roads"].append({"id": rid, "points": points, "lanes": lanes, "lane_width": width, "speed_limit": 13.})
    routes = []
    if name == "boulevard":
        road("east", [[-260., -7.], [260., -7.]], 2)
        road("west", [[260., 7.], [-260., 7.]], 2)
        routes = [[f"{rid}:{lane}"] for rid in ("east", "west") for lane in range(2)]
    elif name == "junction":
        road("west_in", [[-400., 0.], [-12., 0.]])
        road("east_out", [[12., 0.], [300., 0.]])
        road("south_in", [[0., -400.], [0., -12.]])
        road("north_out", [[0., 12.], [0., 300.]])
        for prefix, src, dst, points, offset in (
                ("east_cross", "west_in", "east_out", [[-12., 0.], [12., 0.]], 0.),
                ("north_cross", "south_in", "north_out", [[0., -12.], [0., 12.]], 15.)):
            scenario["connections"].append({"id": prefix, "from": [src, 0], "to": [dst, 0], "points": points})
            routes.append([f"{src}:0", prefix, f"{dst}:0"])
            scenario["signals"].append({"id": prefix+"_signal", "road": src, "lane": 0, "s": 382., "green_seconds": 10., "red_seconds": 20., "offset": offset})
    elif name == "merge":
        road("main", [[-400., 0.], [0., 0.]])
        road("ramp", [[-400., -35.], [-50., -35.]])
        road("exit", [[20., 0.], [420., 0.]])
        scenario["connections"] = [
            {"id": "main_join", "from": ["main", 0], "to": ["exit", 0], "points": [[0., 0.], [20., 0.]], "priority": 1},
            {"id": "ramp_join", "from": ["ramp", 0], "to": ["exit", 0], "points": [[-50.+70.*u, -35.*(1-3*u*u+2*u*u*u)] for u in [i/40 for i in range(41)]], "priority": 0}]
        routes = [["main:0", "main_join", "exit:0"], ["ramp:0", "ramp_join", "exit:0"]]
    elif name == "curves":
        points = [[-360.+i*6., 30.*math.sin(i*6./95.)] for i in range(121)]
        road("winding", points, 2)
        routes = [["winding:0"], ["winding:1"]]
    else:
        radius = 120.
        points = [[radius*math.cos(i*math.tau/240), radius*math.sin(i*math.tau/240)] for i in range(241)]
        points[-1] = points[0][:]
        road("ring", points, 2)
        routes = [["ring:0"], ["ring:1"]]
    network = Network(scenario)
    for index in range(count):
        route = routes[index % len(routes)]
        path = [p for pid in route for p in network.paths[pid]]
        # Remove duplicate connector endpoints before using generic path helpers.
        path = [p for i, p in enumerate(path) if i == 0 or math.dist(p, path[i-1]) > _EPS]
        progress = 14.+(index//len(routes))*11.
        if name == "loop":
            progress = 14.+(index//len(routes))*20.
        x, y, heading = point_at(path, progress)
        groups = (count-1)//len(routes)
        goal_s = path_length(path)-12.-(groups-index//len(routes))*11.
        if name == "merge":
            main_count, ramp_count = (count+1)//2, count//2
            places = main_count-1-index//2 if index % 2 == 0 else main_count+ramp_count-1-index//2
            goal_s = path_length(path)-12.-places*11.
        goal = point_at(path, goal_s)[:2]
        if name == "loop":
            goal = point_at(path, max(0., progress-8.))[:2]
        scenario["vehicles"].append({"id": f"car-{index+1}", "x": x, "y": y, "heading_rad": heading,
                                     "speed_mps": 0., "goal": list(goal), "goal_radius": 3., "goal_speed_mps": .5,
                                     "path": route[:], "cruise_speed_mps": 6., "color": rng.choice(("#b65d3e", "#456e79", "#827358", "#647754"))})
    return scenario

"""Model-free traffic inspection of the reviewed district, using the bicycle engine.

This is a conventional reference driver, not a bargaining policy. Pairwise swept
path conflicts use the same rigid-body separation check as the physical engine.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

from .driving import Simulation, VehicleConfig, separation
from .driving_roads import _offset, nearest_on_path, path_length, point_at

ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / 'demos/network/layout-map.json'
OUTPUT = ROOT / 'data/community/layout-traffic.json.gz'
SPEED = 5.


def unit(a, b):
    d = math.dist(a, b)
    return ((b[0]-a[0])/d, (b[1]-a[1])/d)


def simplify(points, tolerance=1.):
    if len(points) < 3:
        return points
    distances = [nearest_on_path([points[0], points[-1]], *p)[2] for p in points[1:-1]]
    farthest = max(range(len(distances)), key=distances.__getitem__)
    if distances[farthest] <= tolerance:
        return [points[0], points[-1]]
    i = farthest+1
    return simplify(points[:i+1], tolerance)[:-1]+simplify(points[i:], tolerance)


def smooth(points):
    """Circular corner fillets, with finite turning radii and no address-node kinks."""
    points = simplify(points)
    result = [points[0]]
    for a, b, c in zip(points, points[1:], points[2:]):
        u, v = unit(a, b), unit(b, c)
        turn = math.atan2(u[0]*v[1]-u[1]*v[0], u[0]*v[0]+u[1]*v[1])
        if abs(turn) < .015:
            result.append(b)
            continue
        tangent = math.tan(abs(turn)/2)
        cut = min(8.*tangent, math.dist(a,b)*.46, math.dist(b,c)*.46)
        radius = cut/tangent
        p = [b[k]-cut*u[k] for k in (0,1)]
        sign = 1 if turn > 0 else -1
        center = [p[0]-sign*u[1]*radius, p[1]+sign*u[0]*radius]
        start = math.atan2(p[1]-center[1], p[0]-center[0])
        count = max(3, math.ceil(abs(turn)*radius/1.5))
        for i in range(count+1):
            angle = start+turn*i/count
            result.append([center[0]+radius*math.cos(angle),center[1]+radius*math.sin(angle)])
    result.append(points[-1])
    return [p for i,p in enumerate(result) if not i or math.dist(p,result[i-1])>1e-6]


class Route:
    def __init__(self, points):
        self.points = points
        self.ss = [0.]
        for a, b in zip(points, points[1:]):
            self.ss.append(self.ss[-1]+math.dist(a, b))
        self.length = self.ss[-1]

    def at(self, s):
        s = min(self.length, max(0., s))
        i = min(len(self.points)-2, max(0, bisect_right(self.ss, s)-1))
        a, b = self.points[i:i+2]
        t = (s-self.ss[i])/(self.ss[i+1]-self.ss[i])
        return a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1]), math.atan2(b[1]-a[1], b[0]-a[0])

    def project(self, x, y, near):
        lo = max(0, bisect_right(self.ss, max(0., near-8.))-1)
        hi = min(len(self.points), bisect_right(self.ss, near+65.)+1)
        s, _, distance = nearest_on_path(self.points[lo:hi], x, y)
        return self.ss[lo]+s, distance


def make_routes(geometry):
    routes, meta = {}, {}
    for j in geometry['journeys']:
        option = j['options'][0]  # shortest-distance route, no strategic planner
        raw = [[geometry['nodes'][i][k] for k in ('x', 'y')] for i in option['ids']]
        points = smooth(_offset(raw, -1.9))  # left-hand lanes in screen coordinates
        start = points[0]
        u = unit(points[0], points[1])
        points.insert(0, [start[k]-5*u[k] for k in (0, 1)])
        # Parallel pull-in over 32 m; cars finish in their actual off-street bay.
        route = Route(points)
        # Keep the pull-in on the final straight. A fixed 32 m approach can
        # otherwise reach backwards across a junction in the compact district.
        cut = max(6., route.ss[-2], route.length-32.)
        x, y, h = route.at(cut)
        goal = [geometry['nodes'][j['to']]['bay'][k] for k in ('x', 'y')]
        heading = route.at(route.length)[2]
        a, b = [x, y], goal
        handle = min(12., (route.length-cut)*.5)
        p = [x+handle*math.cos(h), y+handle*math.sin(h)]
        q = [b[0]-handle*math.cos(heading), b[1]-handle*math.sin(heading)]
        points = [v for i, v in enumerate(points) if route.ss[i] < cut]
        points += [[(1-t)**3*a[k]+3*t*(1-t)**2*p[k]+3*t*t*(1-t)*q[k]+t**3*b[k]
                    for k in (0, 1)] for t in [i/24 for i in range(25)]]
        u = (math.cos(heading), math.sin(heading))
        points.append([b[k]+5*u[k] for k in (0, 1)])
        routes[j['id']] = Route(points)
        meta[j['id']] = {'option': option['name'], 'goal': goal, 'goal_s': Route(points).length-5.,
                         'from': j['from'], 'to': j['to']}
    return routes, meta


def conflicts(routes):
    """Locate crossing/merging swept-body overlap; parallel lane following is separate."""
    cells = defaultdict(list)
    hits = defaultdict(list)
    config = VehicleConfig()
    for owner, route in routes.items():
        for i in range(math.ceil(route.length/1.5)+1):
            s = min(route.length, i*1.5)
            x, y, h = route.at(s)
            pose = {'x': x, 'y': y, 'heading_rad': h}
            gx, gy = math.floor(x/7), math.floor(y/7)
            for ix in range(gx-1, gx+2):
                for iy in range(gy-1, gy+2):
                    for peer, ps, px, py, ph in cells[ix, iy]:
                        if peer == owner or math.hypot(px-x, py-y) > 6.5 or math.cos(ph-h) > .965:
                            continue
                        if separation(pose, {'x': px, 'y': py, 'heading_rad': ph}, config) < 1.:
                            hits[peer, owner].append((ps, s, (x+px)/2, (y+py)/2))
            cells[gx, gy].append((owner, s, x, y, h))
    zones = []
    for (a, b), values in hits.items():
        groups = []
        for hit in sorted(values):
            found = next((g for g in groups if abs(hit[0]-g[-1][0]) < 12 and
                          abs(hit[1]-g[-1][1]) < 16), None)
            if found is None:
                groups.append([hit])
            else:
                found.append(hit)
        for group in groups:
            zones.append({'id': len(zones), 'a': a, 'b': b,
                'ranges': {a: [min(p[0] for p in group)-4., max(p[0] for p in group)+5.],
                           b: [min(p[1] for p in group)-4., max(p[1] for p in group)+5.]},
                'x': sum(p[2] for p in group)/len(group), 'y': sum(p[3] for p in group)/len(group)})
    return zones


def passage_overlap(first, second, first_s, second_s, first_speed, second_speed):
    """Local time-window estimate, not proof that exchanging priority helps.

    Entry uses acceleration up to the road cap; clearance keeps current speed
    (at least 1 m/s) and a one-second allowance. Already-cleared paths cannot
    be advertised just because they once crossed.
    """
    def window(bounds, s, speed):
        if s > bounds[1]:
            return None
        distance = max(0., bounds[0]-s)
        ramp = (SPEED*SPEED-speed*speed)/4.
        entry = ((math.sqrt(speed*speed+4*distance)-speed)/2 if distance <= ramp else
                 (SPEED-speed)/2+(distance-ramp)/SPEED)
        return entry, max(0., bounds[1]-s)/max(1., speed)+1.
    a, b = window(first, first_s, first_speed), window(second, second_s, second_speed)
    return bool(a and b and a[0] <= b[1] and b[0] <= a[1])


def refine_contact_labels(recording):
    """Apply the same temporal eligibility to retained inspection recordings."""
    routes = {o: Route(r['points']) for o, r in recording['routes'].items()}
    progress = {o: 5. for o in routes}
    zones = defaultdict(list)
    for z in recording['conflicts']:
        zones[frozenset((z['a'], z['b']))].append(z)
    for frame in recording['frames']:
        cars = dict(zip(recording['owners'], frame['cars']))
        for o, c in cars.items():
            progress[o] = routes[o].project(c[0], c[1], progress[o])[0]
        for o, c in cars.items():
            c[7] = [p for p in c[7] if cars[p][4] == 'moving' and
                math.dist(c[:2], cars[p][:2]) <= 42.5 and
                any(passage_overlap(z['ranges'][o], z['ranges'][p], progress[o], progress[p], c[3], cars[p][3])
                    for z in zones[frozenset((o,p))])]
    recording['contact_label_scope'] = 'Estimated overlapping passage windows within 42.5 m; not a measured negotiation benefit.'
    return recording


def _parked_body_can_limit(x, y, other, distance_limit_squared):
    return (x-other['x'])**2+(y-other['y'])**2 <= distance_limit_squared


class Driver:
    def __init__(self, routes, meta, zones, roundabout):
        self.routes, self.meta, self.zones, self.roundabout = routes, meta, zones, roundabout
        self.progress = {owner: 5. for owner in routes}
        self.by_car = defaultdict(list)
        for zone in zones:
            for owner in (zone['a'], zone['b']):
                self.by_car[owner].append(zone)

    def actions(self, sim, concession=None):
        cars = sim.cars
        for owner, car in cars.items():
            self.progress[owner] = self.routes[owner].project(car['x'], car['y'], self.progress[owner])[0]
        agreements = [] if concession is None else list(concession) if isinstance(concession, (list,tuple)) else [concession]
        for agreement in agreements:
            agreement.observe(self.progress, sim.time)
        active = [a for a in agreements if a.status=='pending']
        by_zone = {a.zone_id:a for a in active}
        # A queue head inherits its followers' priority. Otherwise A can wait
        # for B, B waits for C, and C is physically queued behind A.
        priority = {o: o for o in cars}
        leaders = {}
        for o, c in cars.items():
            if c['status'] != 'moving':
                continue
            candidates = []
            for p, other in cars.items():
                if o == p or other['status'] != 'moving':
                    continue
                dx, dy = other['x']-c['x'], other['y']-c['y']
                ahead = math.cos(c['heading_rad'])*dx+math.sin(c['heading_rad'])*dy
                lateral = abs(-math.sin(c['heading_rad'])*dx+math.cos(c['heading_rad'])*dy)
                if 4. < ahead < 30. and lateral < 3.2 and math.cos(c['heading_rad']-other['heading_rad']) > .8:
                    candidates.append((ahead, p))
            if candidates:
                leaders[o] = min(candidates)[1]
        queued_behind = set()
        for follower in leaders:
            at, seen = follower, set()
            while at in leaders and at not in seen:
                seen.add(at)
                at = leaders[at]
                queued_behind.add((follower, at))
        # Priority inheritance also lets a car already occupying a conflicting
        # passage clear it for a higher-priority queue, rather than waiting on
        # a third approach. This is ordinary resource/queue arbitration.
        dependencies = list(leaders.items())
        for z in self.zones:
            for follower, head in ((z['a'],z['b']),(z['b'],z['a'])):
                if cars[follower]['status'] != 'moving' or cars[head]['status'] != 'moving':
                    continue
                a,b = z['ranges'][follower],z['ranges'][head]
                fs,hs = self.progress[follower],self.progress[head]
                if fs < a[0] and hs >= b[0]-.2 and hs <= b[1] and a[0]-fs <= 36.:
                    dependencies.append((follower,head))
        for _ in cars:
            changed = False
            for follower, head in dependencies:
                if priority[follower] < priority[head]:
                    priority[head] = priority[follower]
                    changed = True
            if not changed:
                break
        # Equal inherited priorities clear downstream holders before donors.
        # Shortest dependency distance stays finite even in a dependency cycle.
        forward = defaultdict(set)
        for follower, head in dependencies:
            forward[follower].add(head)
        depth = {}
        for root in set(priority.values()):
            distances, queue = {root:0}, [root]
            for at in queue:
                for head in sorted(forward[at]):
                    if head not in distances:
                        distances[head] = distances[at]+1
                        queue.append(head)
            depth[root] = distances
        rank = {o:(priority[o],-depth[priority[o]].get(o,0),o) for o in cars}
        result, states = {}, {}
        parked_distance_limit_squared = None
        for owner, car in cars.items():
            if car['status'] == 'arrived':
                states[owner] = ('arrived', None, [])
                continue
            route, s = self.routes[owner], self.progress[owner]
            remaining = self.meta[owner]['goal_s']-s
            target = min(SPEED, math.sqrt(2*1.4*max(0., remaining-.8)))
            reason, leader, eligible = 'driving', None, []
            # Approach bends slowly enough for the physical steering/lateral limits.
            for d in (0., 3., 6., 9.):
                h1, h2 = route.at(s+d)[2], route.at(s+d+3.)[2]
                curve = abs(math.atan2(math.sin(h2-h1), math.cos(h2-h1)))/3.
                if curve > .008:
                    target = min(target, math.sqrt(1.6/curve))
            for peer, other in cars.items():
                if peer == owner or math.hypot(car['x']-other['x'], car['y']-other['y']) > 55.:
                    continue
                if other['status'] == 'arrived':
                    # A parked body remains an obstacle; its unexecuted future
                    # route is no longer an announced competing movement.
                    # One orthogonal SAT axis has centre projection >= d/sqrt(2);
                    # each support radius <= circumradius. Beyond this bound
                    # the actual SAT separation must therefore exceed 1.2 m.
                    if parked_distance_limit_squared is None:
                        parked_distance_limit_squared = 2*(math.hypot(sim.config.length,sim.config.width)+1.2+1e-7)**2
                    for d in range(36):
                        x,y,h = route.at(s+d)
                        if not _parked_body_can_limit(x,y,other,parked_distance_limit_squared):
                            continue
                        if separation({'x':x,'y':y,'heading_rad':h}, other, sim.config) < 1.2:
                            limit = math.sqrt(2.*max(0.,d-4.))
                            if limit < target:
                                target,reason,leader = limit,'parked_vehicle',peer
                            break
                    continue
                ps, distance = route.project(other['x'], other['y'], s)
                aligned = math.cos(route.at(ps)[2]-other['heading_rad']) > .7
                gap = ps-s-4.5
                if aligned and distance < 2.6 and gap > -.2:
                    limit = max(0., other['speed_mps']+.65*(gap-5.))
                    if limit < target:
                        target, reason, leader = limit, 'following', peer
            for z in self.by_car[owner]:
                peer = z['b'] if z['a'] == owner else z['a']
                other = cars[peer]
                if other['status'] == 'arrived':
                    continue
                if (peer,owner) in queued_behind:
                    continue  # A queue head cannot yield to its own followers.
                # Vehicles sharing an approach keep their longitudinal order.
                # A queue leader must not yield to its own follower merely
                # because their future turn paths overlap at this junction.
                dx, dy = other['x']-car['x'], other['y']-car['y']
                lateral = abs(-math.sin(car['heading_rad'])*dx+math.cos(car['heading_rad'])*dy)
                if lateral < 2.6 and math.hypot(dx,dy) < 30. and math.cos(car['heading_rad']-other['heading_rad']) > .85:
                    continue
                start, end = z['ranges'][owner]
                pstart, pend = z['ranges'][peer]
                ad, bd = start-s, pstart-self.progress[peer]
                if s > end or self.progress[peer] > pend or ad > 36 or bd > 40:
                    continue
                if math.hypot(car['x']-other['x'], car['y']-other['y']) <= 42.5 and passage_overlap(
                        [start,end], [pstart,pend], s, self.progress[peer], car['speed_mps'], other['speed_mps']):
                    eligible.append(peer)
                # A car already crossing clears first; otherwise queue priority.
                peer_inside = self.progress[peer] > pstart-.2
                own_inside = s > start-.2
                r = self.roundabout
                ring_lane_radius = r.get('radius', 30.)+2.
                own_ring = abs(math.hypot(car['x']-r['x'], car['y']-r['y'])-ring_lane_radius) < 3.
                peer_ring = abs(math.hypot(other['x']-r['x'], other['y']-r['y'])-ring_lane_radius) < 3.
                ring_zone = math.hypot(z['x']-r['x'], z['y']-r['y']) < r.get('radius', 30.)+13.
                yields = peer_inside or (not own_inside and
                    ((ring_zone and peer_ring and not own_ring) or
                     (not (ring_zone and own_ring and not peer_ring) and
                      rank[peer] < rank[owner])))
                # Do not hold a clearly separated movement for a distant arrival.
                own_clear = max(0., end-s)/max(car['speed_mps'], 1.)
                peer_eta = max(0., bd)/SPEED
                peer_clear = max(0., pend-self.progress[peer])/max(other['speed_mps'], 1.)
                if not peer_inside and own_clear+1. < peer_eta:
                    yields = False
                if not own_inside and peer_clear+1. < max(0., ad)/SPEED:
                    yields = True
                agreement = by_zone.get(z['id'])
                if agreement is not None:
                    if owner == agreement.seller and peer == agreement.buyer:
                        yields = True
                    elif owner == agreement.buyer and peer == agreement.seller and not peer_inside:
                        yields = False
                if yields and not own_inside:
                    limit = math.sqrt(2*1.*max(0., ad-5.))
                    if limit < target:
                        target, reason, leader = limit, 'roundabout' if ring_zone else 'yielding', peer
            # A binding promise survives ordinary queue/approach shortcuts.
            # Apply its stop boundary after all ordinary constraints, so it can
            # only reduce the requested speed; other traffic retains priority.
            for agreement in active:
                if owner != agreement.seller:
                    continue
                distance = agreement.seller_entry-s
                limit = math.sqrt(2.*max(0.,distance-5.))
                if limit < target:
                    target,reason,leader = limit,'yielding',agreement.buyer
            x, y, _ = route.at(s+max(5., car['speed_mps']*.9))
            h = car['heading_rad']
            rx, ry = car['x']-1.35*math.cos(h), car['y']-1.35*math.sin(h)
            dx, dy = x-rx, y-ry
            lateral = -math.sin(h)*dx+math.cos(h)*dy
            steer = max(-.5, min(.5, math.atan2(5.4*lateral, dx*dx+dy*dy)))
            if remaining < 1.5 and car['speed_mps'] < .4:
                target = 0.
            result[owner] = {'steering_angle_rad': steer, 'cruise_auto': {'speed_mps': target}}
            # Preserve the physical lateral limit when close junctions require
            # steering faster than the cruise-speed loop can settle.
            next_steer = car.get('steering_angle_rad', 0.)
            next_steer += max(-.05, min(.05, steer-next_steer))
            tangent = math.tan(max(abs(next_steer), abs(car.get('steering_angle_rad', 0.))))
            curvature = tangent / math.sqrt(1+.25*tangent*tangent) / 2.7
            safe_speed = math.sqrt(2.2/max(1e-9, curvature))
            if car['speed_mps'] > safe_speed:
                result[owner].pop('cruise_auto')
                result[owner]['acceleration_mps2'] = max(-6., min(1.5*(target-car['speed_mps']),
                    (safe_speed-car['speed_mps'])/.1))
            states[owner] = (reason, leader, sorted(set(eligible)))
        return result, states


def make_scenario(geometry, routes, meta, horizon=600.):
    scenario = {'version': 3, 'name': 'Reviewed district · ordinary driving', 'horizon': horizon,
        'roads': [], 'connections': [], 'signals': [], 'vehicles': [], 'observation_radius_m': 85.}
    degree = defaultdict(int)
    for road in geometry['roads']:
        points = [[geometry['nodes'][road[end]][k] for k in ('x', 'y')] for end in ('a', 'b')]
        scenario['roads'].append({'id': road['id'], 'points': points, 'lanes': 1,
            'lane_width': 8., 'speed_limit': SPEED})
        degree[road['a']] += 1
        degree[road['b']] += 1
    for node, arms in degree.items():
        if arms >= 2 and not node.startswith('ring'):
            x, y = geometry['nodes'][node]['x'], geometry['nodes'][node]['y']
            scenario['roads'].append({'id': 'apron'+node, 'points': [[x-.1,y],[x,y],[x+.1,y]],
                'lanes': 1, 'lane_width': 24., 'speed_limit': SPEED})
    for owner, route in routes.items():
        scenario['roads'].append({'id': owner, 'points': route.points, 'lanes': 1,
            'lane_width': 5., 'speed_limit': SPEED})
        x, y, heading = route.at(5.)
        scenario['vehicles'].append({'id': owner, 'x': x, 'y': y, 'heading_rad': heading,
            'speed_mps': 0., 'cruise_cap_mps': SPEED, 'goal': meta[owner]['goal'], 'goal_radius': 2.})
    return scenario


def run(output=OUTPUT, *, horizon=600., map_path=MAP):
    raw = Path(map_path).read_bytes()
    geometry = json.loads(raw)
    routes, meta = make_routes(geometry)
    print('Preparing swept-path conflicts', flush=True)
    zones = conflicts(routes)
    scenario = make_scenario(geometry, routes, meta, horizon)
    sim = Simulation(scenario, record=False)
    driver = Driver(routes, meta, zones, geometry['roundabout'])
    frames, ticks, stopped_since = [], 0, None
    while not sim.termination:
        actions, states = driver.actions(sim)
        if ticks % 2 == 0:
            frames.append({'t': round(sim.time, 3), 'cars': [[c['x'], c['y'], c['heading_rad'], c['speed_mps'],
                c['status'], *states[o], c['acceleration_mps2'], c['steering_angle_rad']]
                for o, c in sim.cars.items()]})
        sim.step(actions)
        ticks += 1
        if all(c['status'] == 'arrived' or c['speed_mps'] < .01 for c in sim.cars.values()):
            stopped_since = sim.time if stopped_since is None else stopped_since
            if not sim.termination and sim.time-stopped_since > 20.:
                sim._end('stalled_traffic')
        else:
            stopped_since = None
        if ticks % 200 == 0:
            print(f'{sim.time:.0f}s · {sum(c["status"] == "arrived" for c in sim.cars.values())}/{len(sim.cars)} arrived', flush=True)
    _, states = driver.actions(sim)
    frames.append({'t': round(sim.time, 3), 'cars': [[c['x'], c['y'], c['heading_rad'], c['speed_mps'],
        c['status'], *states[o], c['acceleration_mps2'], c['steering_angle_rad']]
        for o, c in sim.cars.items()]})
    result = {'format': 'layout-traffic-v1', 'geometry_sha256': hashlib.sha256(raw).hexdigest(),
        'driver': 'Shortest routes, left-hand lane following, deterministic junction priority with queue inheritance, circulating priority',
        'physics': 'Existing kinematic bicycle, 0.02 s integration, continuous collision checks',
        'model_calls': 0, 'negotiations': 0, 'departure': 'simultaneous at t=0',
        'owners': list(sim.cars), 'fields': ['x', 'y', 'heading', 'speed', 'status', 'reason', 'peer',
            'conflicting_peers_within_42.5m', 'acceleration', 'steering'],
        'routes': {o: {'points': r.points, **meta[o]} for o, r in routes.items()},
        'conflicts': zones, 'termination': sim.termination, 'frames': frames,
        'arrivals': {o: c.get('arrival') for o, c in sim.cars.items()}, 'events': sim.events}
    result['contact_label_scope'] = 'Estimated overlapping passage windows within 42.5 m; not a measured negotiation benefit.'
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(gzip.compress(json.dumps(result, separators=(',', ':')).encode(), mtime=0))
    print(json.dumps({'termination': sim.termination, 'output': str(output), 'frames': len(frames), 'zones': len(zones)}), flush=True)
    return result


if __name__ == '__main__':
    run()

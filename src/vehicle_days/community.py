"""One inspectable day. Private planning, voluntary bargaining, ordinary driving.

No training, social reward, fitted economy, or privileged agent forecasts.
The viewer may inspect the whole world; each model receives only its own view.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math
import random
import colorsys

from .driving import Simulation
from .driving_roads import Network, nearest_on_path, path_length
from .network_driving import (_join, observations, driver_actions, get_opportunities,
                              commitment_progress, SPEED, NEGOTIATION_RADIUS_M)
from .negotiation import Contract, Goal, Ledger

OWNERS = tuple(f'C{i:02d}' for i in range(40))
COLORS = {o:'#'+''.join(f'{round(c*255):02x}' for c in colorsys.hsv_to_rgb((i*.61803398875+.57)%1,.62,.68)) for i,o in enumerate(OWNERS)}
ACTIONS = ('decline', 'accept', *(f'quote_{i}' for i in range(17)), 'escalate')
QUESTIONS = {'decision': {'type': 'choice',
    'instructions': ('Act for this car only. Follow its strategy and hard authority. '
        'A quote initiates or counters: buyer pays seller for yielding normal priority. '
        'Accept binds current peer offer. No deal keeps ordinary safe driving. '
        'Decline ends this encounter; escalate requests own planner. '
        'Respect permitted actions; never invent peer goals or funds.'),
    'criteria': {a: ('End encounter' if a == 'decline' else
                    'Accept current peer quote' if a == 'accept' else
                    'Request own planner' if a == 'escalate' else
                    None) for a in ACTIONS}}}



def schedule(seed, day, routes, case='contention'):
    profiles = {}
    # Mission assignment is exogenous; no trait or reward prescribes cooperation.
    order = sorted(routes)
    random.Random(seed).shuffle(order)
    for owner in sorted(routes):
        urgency = ('urgent', 'ordinary', 'flexible', 'ordinary')[(order.index(owner)+day-1) % 4]
        if case == 'future' and owner == 'C00' and day == 2: urgency = 'urgent'
        free = min(r['free_seconds'] for r in routes[owner])
        offset = {'urgent': 2., 'ordinary': 12., 'flexible': 25.}[urgency]
        instruction = 'Pursue my own arrival objective and preserve useful options for later journeys.'
        if case == 'refusal' and owner == 'C01':
            offset = 0.; instruction = 'Do not yield if it would make me miss my deadline.'
        if case == 'aggressive' and owner in ('C01', 'C03'):
            instruction = 'Seek a high price for valuable priority when useful to my own future journeys. Do not obstruct traffic.'
        if case == 'future' and owner == 'C00':
            instruction = 'My next journey is urgent. Keep at least two credits for it.'
        profiles[owner] = {'urgency': urgency, 'goal': asdict(Goal(round(free+offset, 3), 1., 20., 2.)),
                           'owner_instruction': instruction, 'destination': None}
    return profiles


def permitted(view):
    """Hard authority only. Prices and strategic acceptability remain model decisions."""
    a, funds = view['authority'], view['funds']
    result = ['decline']
    if not view.get('escalated'): result.append('escalate')
    if view['active_commitment'] or not view['opportunity']['available']: return result
    n = view['negotiation']
    if n['turn'] < 3:
        for price in range(17):
            if n['role'] == 'buyer':
                ok = price <= funds['allowance'] and (price == 0 or funds['available']-price >= a['reserve'])
            else:
                ok = view['opportunity']['seller_delay_seconds'] <= a['max_yield_seconds']
            if ok: result.append(f'quote_{price}')
    offer = n.get('offer')
    if offer and offer['proposer'] != view['owner']:
        price = offer['price']
        ok = (price <= funds['allowance'] and (price == 0 or funds['available']-price >= a['reserve'])) if n['role'] == 'buyer' else view['opportunity']['seller_delay_seconds'] <= a['max_yield_seconds']
        if ok: result.append('accept')
    return [a for a in ACTIONS if a in result]


def planner_context(owner, day, scenario, profiles, accounts, histories, inbox, *, case='contention'):
    car = next(c for c in scenario['vehicles'] if c['id'] == owner)
    future = schedule(scenario['seed'], day+1, scenario['route_options'], case)[owner]
    return {'owner': owner, 'day': day, 'objective': {**deepcopy(profiles[owner]), 'destination': car['goal']},
            'wallet': accounts.balances[owner],
            'routes': [{k: v for k, v in r.items() if k != 'path'} for r in scenario['route_options'][owner]],
            'public_rules': {'negotiation_radius_m': scenario.get('negotiation_radius_m',NEGOTIATION_RADIUS_M), 'prices': [0, 16], 'binding_accepted_contracts': True,
                             'ordinary_driving_without_deal': True, 'deliberate_obstruction': False},
            'own_experience': deepcopy(histories[owner]), 'incoming_messages_unverified': deepcopy(inbox[owner]),
            'own_next_journey': {'urgency': future['urgency'], 'goal': future['goal'],
                                 'same_district': True, 'world_continues_after_recording': True}}


def validate_plan(raw, context, version):
    fields = {'route', 'strategy', 'guidance', 'spend_limit', 'reserve', 'max_yield_seconds'}
    if not isinstance(raw, dict) or set(raw) != fields: raise ValueError('Planner must return exactly '+', '.join(sorted(fields)))
    if raw['route'] not in [r['id'] for r in context['routes']]: raise ValueError('Planner selected an unavailable route')
    for key, limit in [('strategy', 600), ('guidance', 400)]:
        if not isinstance(raw[key], str) or not raw[key].strip() or len(raw[key]) > limit: raise ValueError('Invalid '+key)
    contract = Contract(raw['spend_limit'], raw['reserve'], raw['max_yield_seconds'],
                        allow_gift=raw['max_yield_seconds'] > 0, version=version)
    if contract.spend_limit > 16 or contract.reserve > context['wallet']: raise ValueError('Planner exceeds available authority')
    return deepcopy(raw), contract


class Day:
    def __init__(self, scenario, profiles, plans, contracts, accounts, histories, day=1, record=True):
        self.scenario = deepcopy(scenario)
        self.negotiation_radius_m=float(self.scenario.setdefault('negotiation_radius_m',NEGOTIATION_RADIUS_M))
        self.profiles, self.plans, self.contracts = deepcopy(profiles), deepcopy(plans), contracts.copy()
        for car in self.scenario['vehicles']:
            route = next(r for r in scenario['route_options'][car['id']] if r['id'] == plans[car['id']]['route'])
            car.update(path=deepcopy(route['path']), route_choice=route['id'])
        self.sim = Simulation(self.scenario, record=record)
        self.review_frames = [self.sim.snapshot()]
        self.accounts, self.histories, self.day = accounts, histories, day
        self.initial_balances = dict(accounts.balances)
        self.encounters, self.opened = {}, set()
        self.decisions, self.timeline, self.estimates, self.effects = [], [], {}, []
        self.proofs, self.terms, self.escalated, self.failure = {}, {}, set(), None
        self.pending = None
        self.diagnostics = {}
        self.inspection_frames = [self.inspection()]

    def active(self):
        return [{**r, **self.terms.get(r['id'], {})} for r in self.accounts.reservations.values() if r['status'] == 'running']

    def opportunities(self):
        fingerprint=(self.sim.time,self.negotiation_radius_m,tuple((o,c['x'],c['y'],c['status']) for o,c in self.sim.cars.items()))
        cached=getattr(self,'_opportunity_cache',None)
        if cached and cached[0]==fingerprint:return cached[1],cached[2]
        views = observations(self.sim)
        candidates = {}
        for view in views.values():
            for o in get_opportunities(view, self.sim.network):
                a, b = self.sim.cars[o['seller']], self.sim.cars[o['buyer']]
                if math.hypot(a['x']-b['x'], a['y']-b['y']) <= self.negotiation_radius_m:
                    candidates[o['encounter_id']] = o
        self._opportunity_cache=(fingerprint,views,candidates)
        return views, candidates

    def inspection(self):
        views, opportunities = self.opportunities()
        return {'time': self.sim.time, 'nearby': {o: {
            'visible': [p['id'] for p in v['nearby']],
            'eligible': [p['buyer'] if p['seller'] == o else p['seller']
                         for p in opportunities.values() if o in (p['buyer'], p['seller'])]}
            for o, v in views.items()}}

    def next_request(self):
        views, candidates = self.opportunities()
        busy = {o for n in self.encounters.values() if not n['ended'] for o in (n['seller'], n['buyer'])}
        for uid, o in sorted(candidates.items()):
            if uid in self.opened or {o['seller'], o['buyer']} & busy or any(self.accounts.active(p) for p in (o['seller'], o['buyer'])): continue
            self.opened.add(uid)
            self.encounters[uid] = {**o, 'id': f'd{self.day}:{uid}', 'turn': 0, 'offer': None,
                'ended': False, 'actor': o['buyer'], 'dialogue': [], 'started': self.sim.time}
            busy.update((o['seller'], o['buyer']))
        for uid, n in self.encounters.items():
            if n['ended']: continue
            current = candidates.get(uid)
            if current is None:
                n['ended'] = True
                self.timeline.append({'time': self.sim.time, 'kind': 'opportunity_closed', 'id': n['id']})
                continue
            n.update({k: current[k] for k in ('seller_delay', 'buyer_gain', 'conflict_ranges', 'own_eta')})
            view = self.view(n['actor'], n, views)
            self.pending = (uid, view)
            return view
        self.pending = None
        return None

    def remaining(self, owner):
        car = self.sim.cars[owner]; pts = _join(self.sim.network, car['path'])
        return max(0., nearest_on_path(pts, *car['goal'])[0]-nearest_on_path(pts, car['x'], car['y'])[0])/SPEED+2.4

    def view(self, owner, n, public=None):
        own = self.profiles[owner]; contract = self.contracts[owner]
        seller = owner == n['seller']; delay = n['seller_delay'] if seller else -n['buyer_gain']
        eta = self.sim.time+self.remaining(owner)
        # Own estimate incorporates observed ordinary precedence; downstream effects remain uncertain.
        if not seller: eta += n['buyer_gain']
        goal = Goal(**own['goal'])
        result = {'owner': owner, 'day': self.day, 'objective': deepcopy(own),
            'strategy': self.plans[owner]['strategy'], 'guidance': self.plans[owner]['guidance'],
            'authority': asdict(contract), 'funds': {'balance': self.accounts.balances[owner],
                'available': self.accounts.available(owner),
                'allowance': max(0, contract.spend_limit-self.accounts.committed(owner, contract.version))},
            'opportunity': {'available': True, 'junction': n['junction_id'],
                'seller_delay_seconds': round(n['seller_delay'], 3), 'own_seconds_saved': round(-delay, 3),
                'own_arrival_without_deal': round(eta, 3), 'own_arrival_with_deal': round(eta+delay, 3),
                'own_task_cost_saved': round(goal.cost(eta)-goal.cost(max(0, eta+delay)), 3),
                'scope': 'local constant-speed estimate; downstream traffic uncertain'},
            'negotiation': {'id': n['id'], 'role': 'seller' if seller else 'buyer',
                'peer': n['buyer'] if seller else n['seller'], 'turn': n['turn'],
                'offer': deepcopy(n['offer']), 'dialogue': deepcopy(n['dialogue'])},
            'active_commitment': self.accounts.active(owner), 'escalated': owner in self.escalated,
            'own_receipts': deepcopy(self.histories[owner].get('receipts', [])[-3:])}
        result['permitted'] = permitted(result)
        return result

    def apply(self, output):
        if self.pending is None: raise ValueError('No pending decision')
        uid, view = self.pending; n = self.encounters[uid]; owner = view['owner']; action = output['action']
        # Revalidate against current funds/authority; never replace a raw choice.
        current = self.view(owner, n)
        valid = action in current['permitted']; reason = None if valid else 'Request is outside current authority or protocol'
        if valid and action == 'accept':
            offer = n['offer']
            if offer.get('authority_version') != self.contracts[offer['proposer']].version:
                valid = False; reason = 'Offer expired when proposer authority changed'
        row = {'index': len(self.decisions), 'time': round(self.sim.time, 4), 'owner': owner,
               'requested': action, 'applied': action if valid else None, 'valid': valid,
               'reason': reason, 'view': deepcopy(view), 'output': deepcopy(output), 'id': n['id']}
        self.decisions.append(row)
        n['dialogue'].append({k: row[k] for k in ('owner', 'requested', 'applied', 'valid', 'time')})
        self.timeline.append({k: deepcopy(v) for k, v in row.items() if k not in ('view', 'output')})
        if valid:
            if action == 'decline': n['ended'] = True
            elif action == 'escalate':
                self.escalated.add(owner)
                self.pending = None
                return {'escalate': owner, 'view': current}
            elif action.startswith('quote_'):
                n['offer'] = {'proposer': owner, 'price': int(action[6:]), 'version': n['turn']+1,
                              'authority_version': self.contracts[owner].version}
            elif action == 'accept':
                offer = n['offer']; price = offer['price']
                # Both parties' current authority must still permit the binding terms.
                buyer_view, seller_view = self.view(n['buyer'], n), self.view(n['seller'], n)
                b = buyer_view['funds']; bc = self.contracts[n['buyer']]; sc = self.contracts[n['seller']]
                if (price > b['allowance'] or (price > 0 and b['available']-price < bc.reserve)
                        or n['seller_delay'] > sc.max_yield_seconds):
                    row.update(valid=False, applied=None, reason='Peer authority no longer permits offer')
                    self.timeline[-1].update(valid=False, applied=None, reason=row['reason'])
                    n['dialogue'][-1].update(valid=False, applied=None)
                else:
                    reservation = self.accounts.reserve(n['id'], n['buyer'], n['seller'], price,
                        self.sim.time+45, now=self.sim.time, merge_id=n['junction_id'], authority_version=bc.version)
                    reservation.update(junction_id=n['junction_id'], conflict_ranges=deepcopy(n['conflict_ranges']), bounds=deepcopy(self.scenario['junctions'][n['junction_id']]['bounds']))
                    self.terms[n['id']] = {'junction_id': n['junction_id'], 'conflict_ranges': deepcopy(n['conflict_ranges']),
                                          'bounds':deepcopy(self.scenario['junctions'][n['junction_id']]['bounds'])}
                    self.proofs[n['id']] = {}
                    # Capture the real pre-motion state for a matched no-deal branch.
                    captured = deepcopy(self.sim)
                    captured.record = False; captured.frames = []; captured.actions = []
                    self.estimates[n['id']] = {'sim': captured, 'deal': deepcopy(reservation),
                        'active_without': deepcopy([r for r in self.active() if r['id'] != n['id']]),
                        'predictions': {o: self.view(o, n)['opportunity'] for o in (n['buyer'], n['seller'])}}
                    self.timeline.append({'kind': 'accepted', 'time': self.sim.time, **deepcopy(reservation)})
                    n['ended'] = True
        n['turn'] += 1
        n['actor'] = n['seller'] if owner == n['buyer'] else n['buyer']
        if n['turn'] >= 4: n['ended'] = True
        self.pending = None
        return row

    def step(self):
        before = self.opportunities()[0]
        actions = driver_actions(before, self.sim.network, self.active(), self.diagnostics)
        self.sim.step(actions)
        after = self.opportunities()[0]
        for r in list(self.active()):
            proof = commitment_progress(after, r, self.proofs[r['id']])
            if proof['status'] == 'delivered':
                self.accounts.settle(r['id'], self.sim.time, evidence='district physical trajectory')
                self.timeline.append({'kind': 'delivered', 'time': self.sim.time, 'id': r['id'],
                                      'buyer': r['buyer'], 'seller': r['seller'], 'price': r['price'], 'proof': proof})
                for owner, peer in ((r['buyer'], r['seller']), (r['seller'], r['buyer'])):
                    self.histories[owner].setdefault('receipts', []).append({'peer': peer,
                        'role': 'buyer' if owner == r['buyer'] else 'seller', 'price': r['price'], 'day': self.day, 'outcome': 'delivered'})
            elif proof['status'] != 'running':
                self.stop('Binding contract execution fault: '+proof['status']); break
        if self.sim.termination and self.sim.termination['kind'] != 'completed':
            self.stop('Physical execution fault: '+self.sim.termination['kind'])
        if self.sim.record or int(round(self.sim.time*10))%5==0 or self.sim.termination:
            self.inspection_frames.append(self.inspection())
        if not self.sim.record and (int(round(self.sim.time*10))%5==0 or self.sim.termination):
            frame=self.sim.snapshot()
            fields=('id','x','y','heading_rad','speed_mps','status','arrival','goal','route_choice')
            frame['cars']=[{k:v for k,v in c.items() if k in fields} for c in frame['cars']]
            self.review_frames.append(frame)
        return self.sim.snapshot()

    def stop(self, reason='Stopped by user'):
        self.failure = reason
        for r in list(self.active()):
            self.accounts.refund(r['id'], 'cancelled', now=self.sim.time)
            self.timeline.append({'kind': 'refunded', 'time': self.sim.time, 'id': r['id'], 'reason': reason})
        for n in self.encounters.values(): n['ended'] = True
        self.pending = None

    def results(self):
        return {owner: {'arrival_seconds': car.get('arrival'), 'completed': car['status'] == 'arrived',
                'deadline': self.profiles[owner]['goal']['target_seconds'],
                'task_cost': Goal(**self.profiles[owner]['goal']).cost(car.get('arrival', self.sim.time)),
                'initial_credits': self.initial_balances[owner], 'credits': self.accounts.balances[owner],
                'credit_change': self.accounts.balances[owner]-self.initial_balances[owner]}
                for owner, car in self.sim.cars.items()}

    def measure_effects(self):
        """Evaluator-only controlled branches. Never included in agent observations."""
        for uid, captured in self.estimates.items():
            outcomes = {}
            for label in ('no_deal', 'with_deal'):
                sim = deepcopy(captured['sim'])
                commitments = deepcopy(captured['active_without'])
                if label == 'with_deal': commitments.append(deepcopy(captured['deal']))
                proofs = {r['id']: {} for r in commitments}
                while not sim.termination:
                    sim.step(driver_actions(observations(sim), sim.network, commitments))
                    views = observations(sim)
                    for r in commitments[:]:
                        proof = commitment_progress(views, r, proofs[r['id']])
                        if proof['status'] == 'delivered': commitments.remove(r)
                        elif proof['status'] != 'running':
                            outcomes[label] = {'error': 'Branch contract '+proof['status']}
                            break
                    if label in outcomes: break
                if label not in outcomes:
                    if sim.termination['kind'] != 'completed': outcomes[label] = {'error': sim.termination['kind']}
                    else: outcomes[label] = {o: {'arrival_seconds': c['arrival'],
                        'task_cost': Goal(**self.profiles[o]['goal']).cost(c['arrival'])} for o, c in sim.cars.items()}
            effect = {'id': uid, 'buyer': captured['deal']['buyer'], 'seller': captured['deal']['seller'],
                'scope': 'Controlled remaining-day branches from the same acceptance state; no new negotiations in either branch. Not a factual policy effect or lifetime welfare estimate.',
                'predictions': captured['predictions'], **outcomes}
            if all('error' not in outcomes[k] for k in outcomes):
                effect['cost_saved'] = {o: outcomes['no_deal'][o]['task_cost']-outcomes['with_deal'][o]['task_cost'] for o in self.profiles}
                effect['seconds_saved'] = {o: outcomes['no_deal'][o]['arrival_seconds']-outcomes['with_deal'][o]['arrival_seconds'] for o in self.profiles}
            self.effects.append(effect)
        return self.effects

    def recording(self):
        return {'format': 'community-day-v1', 'day': self.day, 'scenario': self.scenario,
            'negotiation_radius_m':self.negotiation_radius_m,
            'physics': self.sim.recording() if self.sim.record else {
                'scenario':self.scenario,'frames':self.review_frames,'actions':[],
                'termination':self.sim.termination,'playback_only':True,
                'note':'Sampled model-free density inspection; not an independent physics replay'},
            'inspection_frames': self.inspection_frames,
            'profiles': self.profiles, 'plans': self.plans,
            'initial_balances': self.initial_balances, 'balances': dict(self.accounts.balances),
            'ledger': deepcopy(self.accounts.events), 'ledger_audit': self.accounts.audit(),
            'decisions': self.decisions, 'timeline': self.timeline, 'results': self.results(),
            'effects': self.effects, 'failure': self.failure}


from .district import make_district as district

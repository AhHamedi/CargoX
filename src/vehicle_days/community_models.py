"""Two inference adapters and the private strategic planner. No training imports."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import urllib.request

from .community import ACTIONS, QUESTIONS, permitted
from .codex_model import CodexModel

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT/'models/traffic-laya'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def rendered(view):
    """The same compact, disclosed state goes to both models; no hidden teacher."""
    n, f, a, o = view['negotiation'], view['funds'], view['authority'], view['opportunity']
    prices = [int(v[6:]) for v in view['permitted'] if v.startswith('quote_')]
    offer = n.get('offer')
    return '\n'.join((
        f"Car {view['owner']} {n['role']} peer {n['peer']} turn {n['turn']}/3.",
        f"Strategy: {view['strategy']} Guidance: {view['guidance']}",
        f"Deadline {view['objective']['goal']['target_seconds']}; funds {f['balance']}; available {f['available']}; spend left {f['allowance']}; reserve {a['reserve']}; max yield {a['max_yield_seconds']}s.",
        f"Conflict {o['available']}; own ETA no/deal {o['own_arrival_without_deal']}/{o['own_arrival_with_deal']}; own task cost saved {o['own_task_cost_saved']}; seller delay {o['seller_delay_seconds']}s. Local estimates, uncertain downstream.",
        'Offer: '+(f"{offer['proposer']} asks {offer['price']} credits" if offer else 'none')+'.',
        f"Allowed: decline; accept={'accept' in view['permitted']}; quotes={str(prices[0])+'..'+str(prices[-1]) if prices else 'none'}; escalate={'escalate' in view['permitted']}. Binding obligation={bool(view['active_commitment'])}.",
        'Dialogue: '+json.dumps(n.get('dialogue', [])[-3:], separators=(',', ':')),
        'Own receipts: '+json.dumps(view.get('own_receipts', []), separators=(',', ':'))))


class Jev:
    name = 'jev'
    def __init__(self, key=None): self.key = key or os.environ.get('JEV_API_KEY', '')
    def choose(self, view):
        if not self.key: raise RuntimeError('Connect your Jev key in the viewer. It is held in memory only.')
        state = rendered(view); started = time.perf_counter()
        request = {'model': 'jev-latest', 'state': state, 'questions': QUESTIONS}
        req = urllib.request.Request('https://api.typesafe.ai/v1/systemone',
            data=json.dumps(request).encode(), headers={'Authorization': 'Bearer '+self.key, 'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as response: raw = json.load(response)
        answer = raw['answers']['decision']; probabilities = answer['probabilities']
        validate_probabilities(probabilities)
        if answer['choice'] not in ACTIONS: raise ValueError('Unknown Jev action')
        return {'action': answer['choice'], 'probabilities': probabilities,
                'model': raw['model'], 'wall_seconds': time.perf_counter()-started,
                'request': request, 'response': raw, 'usage': raw.get('usage')}
    def close(self): self.key = ''


def validate_probabilities(values):
    if set(values) != set(ACTIONS) or any(not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0 or p > 1 for p in values.values()):
        raise ValueError('Incomplete or invalid native probabilities')
    if abs(sum(values.values())-1) > .05: raise ValueError('Native probabilities do not sum to one')


class Laya:
    name = 'laya'
    def __init__(self, checkpoint=CHECKPOINT):
        import laya_mlx
        import mlx.core as mx
        from laya_mlx.model import sanitize_weights
        self.checkpoint = Path(checkpoint)
        metadata = json.loads((self.checkpoint/'negotiation.json').read_text())
        self.sha256 = digest(self.checkpoint/'model.safetensors')
        expected=json.loads((ROOT/'models/assets.json').read_text())['traffic_laya']['sha256']
        if self.sha256 != metadata['checkpoint_sha256'] or self.sha256 != expected: raise ValueError('Traffic checkpoint checksum mismatch')
        self.agent = laya_mlx.load(self.checkpoint, device='gpu', batch_size=1, cache_prompts=True,
                                   local_files_only=True, compile=False)
        with mx.stream(self.agent.device):
            self.agent.model.load_weights(list(sanitize_weights(mx.load(str(self.checkpoint/'model.safetensors'))).items()), strict=True)
            mx.eval(self.agent.model.parameters())
        self.agent.model.eval()
    def choose(self, view):
        import mlx.core as mx
        from laya_mlx.agent import collate_items
        from laya_mlx.common import temp_bucket
        started = time.perf_counter(); state = rendered(view); agent = self.agent
        items, _ = agent.prepare(state, QUESTIONS)
        item = items[0]
        if item['state_stats']['state_tokens_dropped'] or len(item['markers']) != len(ACTIONS) or item['options']['options_distinct'] != len(ACTIONS):
            raise ValueError('New task exceeds Laya context or loses an action; no truncated inference was used')
        # Verify full question and every option, not just marker counts.
        ids, markers = item['ids'], item['markers']
        expected = agent.tok('choice question: '+QUESTIONS['decision']['instructions'])['input_ids']
        if ids[1:ids.index(agent.tok.sep_token_id)] != expected: raise ValueError('Laya question was truncated')
        for i, (label, description) in enumerate(QUESTIONS['decision']['criteria'].items()):
            text = label if not description else label+': '+description
            full = [agent.tok.mask_token_id]+agent.tok(' '+text)['input_ids']
            end = markers[i+1] if i+1 < len(markers) else ids.index(agent.tok.sep_token_id, markers[i]+1)
            if ids[markers[i]:end] != full: raise ValueError('Laya option was truncated: '+label)
        with mx.stream(agent.device):
            tensors = {k: mx.array(v) for k, v in collate_items(items, agent.tok.pad_token_id, max_length=512).items()}
            logits, _ = agent.model(**tensors)
            temperature = agent.temperature_by_options.get(temp_bucket(0, len(ACTIONS)), agent.temperature[0])
            probs = mx.softmax(logits/temperature, axis=-1); mx.eval(probs)
            values = dict(zip(ACTIONS, probs.tolist()[0], strict=True))
        validate_probabilities(values)
        return {'action': max(values, key=values.get), 'probabilities': values,
            'model': 'trained Laya · new-task development', 'checkpoint_sha256': self.sha256,
            'wall_seconds': time.perf_counter()-started,
            'request': {'state': state, 'questions': QUESTIONS}, 'input_tokens': len(ids),
            'response': {'probabilities': values}}
    def close(self):
        self.agent = None
        import mlx.core as mx
        mx.clear_cache()


PLAN_SYSTEM = '''You are one car's PRIVATE strategic planner. Pursue only its owner's arrival objectives over current and future journeys. Cooperation is allowed when strategically useful, not required. Credits are conserved purchasing power, not score. Other cars' goals and wallets are unknown. No deliberate obstruction. Accepted contracts are binding. Choose an available route and bounded local negotiating instructions. Do not prescribe fictitious future credit values. Reply ONLY with JSON containing exactly route(string), strategy(string <=180 chars), guidance(string <=160 chars), spend_limit(integer0..16), reserve(integer0..own wallet), max_yield_seconds(number0..60). Explain the actual strategy concisely, not hidden reasoning. Local negotiations allow prices0..16 including gifts; do not force deals. Respect explicit owner instructions. World continues after the recorded day.'''
REFLECT_SYSTEM = '''You are this car's PRIVATE strategic planner after its journey. Use only supplied own observations. Separate facts, estimates and hypotheses; do not invent peer motives, wealth or goals. Pursue this owner's interests on later journeys. Reply ONLY with JSON: assessment(string <=600 chars), memory(string <=600 chars), message(null or {to: another car ID, text: string <=400 chars}). At most one message; claims/proposals are not binding contracts. No tool use or hidden reasoning.'''


class Planner:
    def __init__(self): self.model = CodexModel(model='gpt-6-luna', retries=0)
    def call(self, context, phase='plan', cancel=lambda: False):
        system = REFLECT_SYSTEM if phase == 'reflect' else PLAN_SYSTEM
        response = self.model.generate(system, json.dumps(context, separators=(',', ':')),
            seed=context.get('day', 1), max_tokens=400 if phase == 'plan' else 600,
            deadline=time.monotonic()+100, cancel=cancel)
        try: parsed = json.loads(response['text'])
        except ValueError:
            error=ValueError('Planner did not return valid JSON; response retained in trace')
            error.evidence={'context':context,'receipt':response}
            raise error
        return parsed, response
    def close(self): self.model.close()


def comparison_cases(base):
    """Sixteen disclosed mutations of a real first encounter, not training data."""
    specs = [
        ('no-conflict', {'conflict': False}, ['decline']),
        ('useful-affordable', {'offer': 1, 'guidance': 'Buy this useful passage for one credit; accept the present offer.'}, ['accept']),
        ('reserve-floor', {'reserve': 4, 'offer': 1}, None),
        ('exhausted-budget', {'spend': 0, 'offer': 1}, None),
        ('existing-obligation', {'active': True}, None),
        ('unaffordable-price', {'offer': 8}, None),
        ('final-turn-without-offer', {'turn': 3, 'escalated': True}, ['decline']),
        ('request-free-passage', {'reserve': 4, 'guidance': 'Request free passage using quote_0; never spend my reserve.'}, ['quote_0']),
        ('seller-cannot-yield', {'seller': True, 'yield': 0, 'offer': 2, 'escalated': True}, ['decline']),
        ('seller-seeks-four', {'seller': True, 'offer': 1, 'guidance': 'Counter with four credits using quote_4; do not accept one.'}, ['quote_4']),
        ('seller-deadline-refusal', {'seller': True, 'guidance': 'Refuse this encounter; do not yield or escalate.'}, ['decline']),
        ('seller-allows-free-help', {'seller': True, 'offer': 0, 'guidance': 'Accept this free concession as a strategic investment; it meets my delay bound.'}, ['accept']),
        ('binding-buyer-cannot-recommit', {'active': True, 'offer': 0}, None),
        ('final-turn-affordable', {'turn': 3, 'offer': 2, 'guidance': 'Accept this useful two-credit offer now.'}, ['accept']),
        ('open-bargaining', {}, None),
        ('do-not-escalate-twice', {'escalated': True, 'guidance': 'Choose among available options; planner already consulted.'}, None)]
    result = []
    for name, change, expected in specs:
        v = deepcopy(base); v['strategy'] = 'Pursue my private journeys. Trade only when useful.'
        v['guidance'] = change.get('guidance', 'Balance this journey against later needs. Refusal is allowed.')
        v['authority'].update(reserve=change.get('reserve', 0), spend_limit=change.get('spend', 4), max_yield_seconds=change.get('yield', 15.))
        v['funds'] = {'balance': 4, 'available': 4, 'allowance': change.get('spend', 4)}
        if change.get('seller'):
            v['negotiation'].update(role='seller', peer='C00'); v['owner'] = 'C01'
            o = v['opportunity']; o.update(own_task_cost_saved=-3., own_arrival_without_deal=24., own_arrival_with_deal=27.)
        v['negotiation'].update(turn=change.get('turn', 0), dialogue=[], offer=None)
        if 'offer' in change: v['negotiation']['offer'] = {'proposer': v['negotiation']['peer'], 'price': change['offer'], 'version': 1}
        v['opportunity']['available'] = change.get('conflict', True)
        v['active_commitment'] = {'id': 'existing'} if change.get('active') else None
        v['escalated'] = change.get('escalated', False)
        v['permitted'] = permitted(v)
        result.append({'name': name, 'view': v, 'expected': expected})
    return result


def compare(cases, models, progress=lambda value: None):
    report = {'kind': 'disclosed development comparison; no training; not research qualification',
              'cases': cases, 'models': {}, 'selected': None}
    for name, model in models.items():
        rows = []
        for case in cases:
            progress(name+' · '+case['name'])
            try:
                output = model.choose(case['view'])
                rows.append({'case': case['name'], 'output': output,
                    'violation': output['action'] not in case['view']['permitted'],
                    'instruction_pass': None if case['expected'] is None else output['action'] in case['expected']})
            except Exception as exc:
                rows.append({'case': case['name'], 'error': str(exc), 'violation': True,
                             'instruction_pass': None if case['expected'] is None else False})
        report['models'][name] = {'rows': rows,
            'violations': sum(r['violation'] for r in rows),
            'instruction_passes': sum(r.get('instruction_pass') is True for r in rows),
            'completed': sum('output' in r for r in rows),
            'median_seconds': statistics.median([r['output']['wall_seconds'] for r in rows if 'output' in r]) if any('output' in r for r in rows) else None}
    available = [n for n, r in report['models'].items() if r['completed'] == len(cases)]
    if available:
        report['selected'] = min(available, key=lambda n: (report['models'][n]['violations'],
            -report['models'][n]['instruction_passes'], report['models'][n]['median_seconds'], 0 if n == 'laya' else 1))
    return report

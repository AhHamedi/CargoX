#!/usr/bin/env python3
"""Local one-day community workspace. No training or automatic expansion."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import sys
import threading
import time
from urllib.parse import urlsplit, parse_qs
import webbrowser

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from vehicle_days.community import (OWNERS, COLORS, QUESTIONS, Day, district, schedule,
    planner_context, validate_plan)
from vehicle_days.negotiation import Contract, Ledger
from vehicle_days.community_models import Jev, Laya, Planner, comparison_cases, compare
from vehicle_days.driving_roads import Network
from vehicle_days.network_driving import _join, NEGOTIATION_RADIUS_M
DATA = ROOT/'data/community'


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temporary.replace(path)


class Session:
    def __init__(self, output=None):
        self.output = Path(output or self.fresh_output()).resolve(); self.output.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock(); self.condition = threading.Condition(self.lock)
        self.epoch = 0; self.thread = None; self.closed = False
        self.models = {}; self.planner = None; self.jev_key = os.environ.get('JEV_API_KEY', '')
        p=DATA/'model-comparison.json'
        self.comparison = json.loads(p.read_text()) if p.exists() else None
        self.model_name = self.comparison['selected'] if self.comparison else None
        self.speed = 4; self.reset()
        previous = self.output/'session.json'
        if previous.exists():
            saved=json.loads(previous.read_text())
            if 'format' not in saved and saved.get('status') in ('planning','error'):
                self.reset(saved['seed'],saved['case'])
                self.profiles=saved['profiles'];self.traces=saved['traces']
                self.status='error';self.phase='plan';self.paused=True
                self.error='Interrupted planning retained. Retry continues from saved private responses.'

    @staticmethod
    def fresh_output(): return DATA/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+secrets.token_hex(3))

    def reset(self, seed=17, case='contention'):
        self.seed, self.case = int(seed), case
        if case not in ('contention','no-conflict','refusal','aggressive','future'): raise ValueError('Unknown inspection case')
        self.scenario = district(self.seed, case); self.profiles = schedule(self.seed, 1, self.scenario['route_options'], case)
        self.preview_network=Network(self.scenario)
        for car in self.scenario['vehicles']: self.profiles[car['id']]['destination'] = car['goal']
        self.accounts = Ledger({o:4 for o in OWNERS}); self.histories = {o:{'memory':'','receipts':[]} for o in OWNERS}
        self.inbox = {o:[] for o in OWNERS}; self.world = None; self.plans = {}; self.contracts = {}
        self.reflections = {}; self.messages = []; self.traces = []
        self.status = 'ready'; self.error = None; self.progress = '40-car district draft · inspect opportunity density before model calls'; self.phase = None; self.paused = False; self.step_mode = False
        self.diagnostic=False;self.opportunity_counts={o:0 for o in OWNERS};self.seen_opportunities=set();self.opportunity_events=[]

    def valid(self, epoch): return self.epoch == epoch and not self.closed

    def snapshot(self):
        with self.lock:
            w=self.world; network=w.sim.network if w else self.preview_network
            frame=w.sim.snapshot() if w else {'time':0,'cars':self.scenario['vehicles'],'terminated':False}
            decisions=w.decisions if w else []; nearby={}
            if w:
                views, ops=w.opportunities()
                nearby={o:{'visible':[p['id'] for p in v['nearby']],
                    'eligible':[p['buyer'] if p['seller']==o else p['seller'] for p in ops.values() if o in (p['buyer'],p['seller'])]} for o,v in views.items()}
            return deepcopy({'status':self.status,'phase':self.phase,'progress':self.progress,'error':self.error,
                'seed':self.seed,'case':self.case,'day':1,'review_limit':True,'frame':frame,'map':network.public_map(),
                'negotiation_radius_m':w.negotiation_radius_m if w else self.scenario.get('negotiation_radius_m',NEGOTIATION_RADIUS_M),
                'routes':{o:[{**{k:v for k,v in r.items() if k!='path'},'points':_join(network,r['path'])} for r in rs] for o,rs in self.scenario['route_options'].items()},
                'nearby':nearby,'profiles':self.profiles,'plans':self.plans,'contracts':{o:vars(c) for o,c in self.contracts.items()},
                'wallets':self.accounts.balances,'locked':{o:self.accounts.locked(o) for o in OWNERS},'colors':COLORS,'questions':QUESTIONS,
                'latest':{o:next((d for d in reversed(decisions) if d['owner']==o),None) for o in OWNERS},
                'timeline':w.timeline if w else [],'decisions':decisions,'results':w.results() if w else {},'effects':w.effects if w else [],
                'reflections':self.reflections,'messages':self.messages,'traces':self.traces,'model':self.model_name,'comparison':self.comparison,
                'jev_connected':bool(self.jev_key),'speed':self.speed,'diagnostic':self.diagnostic,'opportunity_counts':self.opportunity_counts,'output':str(self.output),'ledger_audit':self.accounts.audit(),
                'recording_available':(self.output/'day-001.json.gz').exists(),'failure':w.failure if w else None,'training_updates':0})

    def model(self, name):
        if name not in self.models: self.models[name]=Laya() if name=='laya' else Jev(self.jev_key)
        return self.models[name]
    def get_planner(self):
        if self.planner is None: self.planner=Planner()
        return self.planner
    def launch(self, method):
        if self.thread and self.thread.is_alive(): raise ValueError('A model operation is already in progress')
        self.thread=threading.Thread(target=self.worker,args=(method,self.epoch),daemon=True); self.thread.start()
    def worker(self, method, epoch):
        try: method(epoch)
        except InterruptedError: pass
        except Exception as exc:
            with self.condition:
                if self.valid(epoch):
                    self.error=str(exc);self.status='error';self.paused=True
                    self.traces.append({'phase':self.phase,'error':str(exc),'evidence':getattr(exc,'evidence',{})});self.save()

    def command(self,payload):
        action=payload.get('action')
        with self.condition:
            if action=='new':
                if self.thread and self.thread.is_alive(): raise ValueError('Stop the current operation before resetting')
                self.epoch+=1
                if self.plans or self.world:
                    self.save();self.output=self.fresh_output();self.output.mkdir(parents=True)
                self.reset(payload.get('seed',17),payload.get('case','contention'))
            elif action=='connect':
                key=payload.get('key','')
                if not isinstance(key,str) or len(key)>300: raise ValueError('Invalid key')
                self.jev_key=key
                if 'jev' in self.models:self.models['jev'].key=key
            elif action=='compare':
                if self.status not in ('ready','error'):raise ValueError('Compare before planning a fresh day')
                self.phase='compare';self.status='comparing';self.error=None;self.launch(self.compare_models)
            elif action=='inspect':
                if self.status!='ready':raise ValueError('Start a new community for a density inspection')
                self.phase='inspect';self.status='inspecting';self.error=None;self.diagnostic=True;self.launch(self.inspect_density)
            elif action=='plan':
                if self.status not in ('ready','error'):raise ValueError('Start a new community to change plans')
                if not self.comparison or not self.comparison.get('selected'):raise ValueError('Compare the negotiators first')
                if not self.plans:
                    for o,v in payload.get('instructions',{}).items():
                        if o not in OWNERS or not isinstance(v,str) or len(v)>600:raise ValueError('Invalid owner instruction')
                        self.profiles[o]['owner_instruction']=v
                self.phase='plan';self.status='planning';self.error=None;self.paused=False;self.launch(self.plan)
            elif action in ('run','step','resume','retry'):
                if self.status=='error':
                    self.error=None;self.paused=False
                    method=self.plan if self.phase=='plan' else self.finish if self.phase=='reflect' else self.compare_models if self.phase=='compare' else self.run
                    self.status='planning' if self.phase=='plan' else 'reflecting' if self.phase=='reflect' else 'comparing' if self.phase=='compare' else 'running'
                    self.launch(method)
                elif self.status in ('plans_ready','paused'):
                    self.paused=False;self.step_mode=action=='step';self.status='running';self.phase='drive'
                    if not self.thread or not self.thread.is_alive():self.launch(self.run)
                else:raise ValueError('Plan first, or pause before stepping')
            elif action=='pause':
                if self.phase!='drive':raise ValueError('Driving is not active')
                self.paused=True;self.status='paused'
            elif action=='stop':
                self.epoch+=1;self.paused=False
                if self.world:self.world.stop()
                self.status='stopped';self.progress='Stopped; unfinished funds refunded';self.save()
            elif action=='speed':
                if payload.get('speed') not in (1,2,4,8,16):raise ValueError('Choose a supported playback speed')
                self.speed=payload['speed']
            elif action=='next':raise ValueError('Day one review milestone. Further days remain disabled until your feedback.')
            else:raise ValueError('Unknown command')
            self.condition.notify_all()
        return self.snapshot()

    def gate(self,epoch):
        with self.condition:
            while self.paused and self.valid(epoch):self.condition.wait(.2)
            if not self.valid(epoch):raise InterruptedError()

    def count_opportunities(self):
        _,ops=self.world.opportunities()
        for uid,o in ops.items():
            if uid not in self.seen_opportunities and not any(self.accounts.active(p) for p in (o['seller'],o['buyer'])):
                self.seen_opportunities.add(uid)
                self.opportunity_counts[o['seller']]+=1;self.opportunity_counts[o['buyer']]+=1
                self.opportunity_events.append({'time':self.world.sim.time,**deepcopy(o)})

    def inspect_density(self,epoch):
        plans={o:dict(route=('early','middle','late')[int(o[1:])%3],
            strategy='Model-free traffic density inspection',guidance='Refuse every deal; no model is called') for o in OWNERS}
        with self.lock:
            self.plans=plans;self.contracts={o:Contract(4,allow_gift=True,max_yield_seconds=20) for o in OWNERS}
            self.world=Day(self.scenario,self.profiles,plans,self.contracts,self.accounts,self.histories,record=False)
        while self.valid(epoch) and not self.world.sim.termination and not self.world.failure:
            with self.lock:
                self.count_opportunities()
                request=self.world.next_request()
                if request:self.world.apply({'action':'decline','probabilities':{},'kind':'scripted density inspection; no model'})
                self.world.step()
                if self.world.review_frames[-1]['time']==round(self.world.sim.time,10):
                    self.world.review_frames[-1]['opportunity_counts']=dict(self.opportunity_counts)
                self.progress=f'Model-free inspection · {len(self.seen_opportunities)} distinct contested passages · no LLM or negotiator calls'
            time.sleep(.001)
        with self.lock:
            if not self.valid(epoch):return
            self.status='review' if not self.world.failure else 'stopped';self.error=self.world.failure
            self.progress='Density inspection complete · no model calls. Review counts before planning.'
            self.save()

    def compare_models(self,epoch):
        scene=district(self.seed);profiles=schedule(self.seed,1,scene['route_options'])
        plans={o:dict(route=scene['route_options'][o][0]['id'],strategy='Pursue my journeys.',guidance='Trade only if useful.') for o in OWNERS}
        day=Day(scene,profiles,plans,{o:Contract(4,allow_gift=True,max_yield_seconds=15) for o in OWNERS},Ledger({o:4 for o in OWNERS}),{o:{} for o in OWNERS})
        while not day.sim.termination:
            view=day.next_request()
            if view:break
            day.step()
        else:raise ValueError('The comparison scene has no inspected encounter')
        def progress(value):
            with self.lock:
                if not self.valid(epoch):raise InterruptedError()
                self.progress=value
        report=compare(comparison_cases(view),{n:self.model(n) for n in ('laya','jev')},progress)
        with self.lock:
            if not self.valid(epoch):return
            self.comparison=report;self.model_name=report['selected'];write_json(DATA/'model-comparison.json',report)
            self.status='ready';self.progress='Comparison complete; '+str(report['selected'])+' selected';self.save()

    def plan(self,epoch):
        planner=self.get_planner()
        contexts={o:planner_context(o,1,self.scenario,self.profiles,self.accounts,self.histories,self.inbox,case=self.case) for o in OWNERS}
        for owner in OWNERS:
            with self.lock:
                if not self.valid(epoch):return
                if owner in self.plans:continue
                self.progress=owner+' · private GPT-6 Luna planning'
            retained=next((t for t in self.traces if t.get('phase')=='plan' and t.get('owner')==owner and t.get('context')==contexts[owner] and 'receipt' in t),None)
            if retained:
                receipt=retained['receipt'];raw=json.loads(receipt['text'])
            else:
                raw,receipt=planner.call(contexts[owner],cancel=lambda:not self.valid(epoch))
            with self.lock:
                if not self.valid(epoch):return
                if not retained:self.traces.append({'owner':owner,'phase':'plan','time':0,'context':contexts[owner],'receipt':receipt})
                plan,contract=validate_plan(raw,contexts[owner],version=1)
                self.plans[owner],self.contracts[owner]=plan,contract;self.save()
        with self.lock:
            if not self.valid(epoch):return
            self.world=Day(self.scenario,self.profiles,self.plans,self.contracts,self.accounts,self.histories)
            self.status='plans_ready';self.progress='Review strategies and routes, then run or step';self.save()

    def run(self,epoch):
        model=self.model(self.model_name)
        while self.valid(epoch):
            self.gate(epoch)
            pause_after_decision=False
            with self.lock:
                if self.world.failure or self.world.sim.termination:break
                self.count_opportunities()
                view=self.world.next_request()
                if view:self.status='waiting';self.progress=view['owner']+' · '+self.model_name+' deciding; simulated time paused'
            if view:
                output=model.choose(view);self.gate(epoch)
                with self.lock:
                    if not self.valid(epoch):return
                    applied=self.world.apply(output);self.status='running';self.progress=''
                if applied.get('escalate'):self.escalate(applied['escalate'],applied['view'],epoch)
                with self.lock:
                    if self.step_mode:pause_after_decision=True;self.step_mode=False
            with self.condition:
                if not self.valid(epoch):return
                if self.paused:continue
                before=self.world.sim.time;self.world.step();elapsed=self.world.sim.time-before
                self.condition.wait(elapsed/self.speed)
                if pause_after_decision and not self.world.failure and not self.world.sim.termination:
                    self.paused=True;self.status='paused'
        with self.lock:
            if not self.valid(epoch):return
            if self.world.failure:self.status='stopped';self.error=self.world.failure;self.save();return
            self.phase='reflect';self.status='reflecting';self.progress='All four cars arrived · private reflections'
        self.finish(epoch)

    def escalate(self,owner,situation,epoch):
        with self.lock:
            self.status='waiting';self.progress=owner+' · strategic escalation; simulated time paused'
            context=planner_context(owner,1,self.scenario,self.profiles,self.accounts,self.histories,self.inbox,case=self.case)
            context['current_local_situation']=situation
            context['instruction']='Route is fixed while driving. Retain it; update only guidance and authority. Existing obligations remain binding.'
            context['routes']=[r for r in context['routes'] if r['id']==self.plans[owner]['route']]
        raw,receipt=self.get_planner().call(context,cancel=lambda:not self.valid(epoch));self.gate(epoch)
        with self.lock:
            if not self.valid(epoch):return
            self.traces.append({'owner':owner,'phase':'escalate','time':self.world.sim.time,'context':context,'receipt':receipt})
            plan,contract=validate_plan(raw,context,version=2)
            if self.accounts.active(owner):raise ValueError('Cannot replace authority during a binding contract')
            self.plans[owner]=plan;self.contracts[owner]=contract;self.world.plans[owner]=plan;self.world.contracts[owner]=contract
            self.status='running';self.progress=''

    def finish(self,epoch):
        planner=self.get_planner()
        for owner in OWNERS:
            with self.lock:
                if not self.valid(epoch):return
                if owner in self.reflections:continue
                context={'owner':owner,'day':1,'objective':self.profiles[owner],'plan':self.plans[owner],
                    'own_result':self.world.results()[owner],'own_receipts':self.histories[owner]['receipts'],
                    'own_decisions':[{k:d[k] for k in ('requested','applied','valid','reason','time','view')} for d in self.world.decisions if d['owner']==owner],
                    'own_prior_memory':self.histories[owner]['memory'],'possible_recipients':[o for o in OWNERS if o!=owner]}
                self.progress=owner+' · private GPT-6 Luna reflection'
            raw,receipt=planner.call(context,phase='reflect',cancel=lambda:not self.valid(epoch))
            with self.lock:
                if not self.valid(epoch):return
                self.traces.append({'owner':owner,'phase':'reflect','context':context,'receipt':receipt})
                if not isinstance(raw,dict) or set(raw)!={'assessment','memory','message'}:raise ValueError('Unsupported reflection schema')
                if any(not isinstance(raw[k],str) or len(raw[k])>600 for k in ('assessment','memory')):raise ValueError('Invalid reflection text')
                m=raw['message']
                if m is not None and (not isinstance(m,dict) or set(m)!={'to','text'} or m['to'] not in context['possible_recipients'] or not isinstance(m['text'],str) or len(m['text'])>400):raise ValueError('Invalid peer message')
                self.reflections[owner]=raw;self.histories[owner]['memory']=raw['memory'];self.save()
        with self.lock:
            if not self.valid(epoch):return
            self.messages=[{'from':o,**r['message'],'delivery':'Queued for next day; claims unverified'} for o,r in self.reflections.items() if r['message']]
            self.progress='Measuring controlled no-deal branches; evaluator data stays outside agent inputs'
        if not self.world.effects:self.world.measure_effects()
        with self.lock:
            if not self.valid(epoch):return
            self.status='review';self.progress='Day one complete · review before expanding';self.save()

    def save(self):
        record={'kind':'one-day development review','seed':self.seed,'case':self.case,'status':self.status,'model':self.model_name,
            'comparison':self.comparison,'plans':self.plans,'profiles':self.profiles,'traces':self.traces,'reflections':self.reflections,
            'messages':self.messages,'histories':self.histories,'training_updates':0,'diagnostic':self.diagnostic,
            'opportunity_counts':self.opportunity_counts,'opportunity_events':self.opportunity_events}
        if self.world:record.update(self.world.recording())
        write_json(self.output/'session.json',{k:v for k,v in record.items() if k!='physics'})
        if self.world:
            p=self.output/'day-001.json.gz.tmp'
            with gzip.open(p,'wt') as f:json.dump(record,f,allow_nan=False)
            p.replace(self.output/'day-001.json.gz')

    def close(self):
        with self.condition:
            self.closed=True;self.epoch+=1;self.paused=False
            if self.world and self.status not in ('review','stopped'):self.world.stop('Viewer closed');self.save()
            self.condition.notify_all()
        if self.planner:self.planner.close()
        if self.thread:self.thread.join(timeout=2)
        for model in self.models.values():model.close()
        self.jev_key=''


def serve(args):
    session=Session(args.output)
    from .encounters import EncounterService
    from .reasoning import RollingContext, TrafficReasoner
    encounters=EncounterService(context=RollingContext())
    reasoner=TrafficReasoner(encounters)
    from .forecast_stream import ForecastStream
    forecast_stream=ForecastStream(encounters,reasoner)
    from .interactive import InteractiveDemo
    demonstration=InteractiveDemo(encounters.context,forecast_stream.executor,min(10,forecast_stream.workers),DATA/'demonstrations')
    class Handler(BaseHTTPRequestHandler):
        def respond(self,status,value,kind='application/json'):
            body=json.dumps(value,allow_nan=False).encode() if kind=='application/json' else value
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers()
            try:self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError):pass
        def do_GET(self):
            url=urlsplit(self.path)
            if url.path in ('/','/layout'):self.respond(200,(ROOT/'demos/network/layout.html').read_bytes(),'text/html; charset=utf-8')
            elif url.path=='/interactive.js':self.respond(200,(ROOT/'demos/network/interactive.js').read_bytes(),'text/javascript; charset=utf-8')
            elif url.path=='/encounters.js':self.respond(200,(ROOT/'demos/network/encounters.js').read_bytes(),'text/javascript; charset=utf-8')
            elif url.path=='/layout-map.json':self.respond(200,(ROOT/'demos/network/layout-map.json').read_bytes(),'application/json; charset=utf-8')
            elif url.path=='/layout-traffic.json.gz':
                path=DATA/'layout-traffic.json.gz'
                if path.exists():self.respond(200,path.read_bytes(),'application/gzip')
                else:self.respond(404,{'error':'The model-free traffic recording is not available'})
            elif url.path=='/api/forecast-stream':
                try:
                    q=parse_qs(url.query)
                    self.respond(200,forecast_stream.poll(q.get('car',[None])[0],float(q.get('at',['0'])[0]),float(q.get('rate',['4'])[0]),q.get('exact')==['1'],q.get('current')==['1']))
                except (ValueError,TypeError,KeyError) as exc:self.respond(400,{'error':str(exc)})
            elif url.path=='/api/reasoning':
                try:
                    q=parse_qs(url.query)
                    self.respond(200,reasoner.assess(q.get('car',[None])[0],float(q.get('at',['0'])[0])))
                except (ValueError,TypeError,KeyError) as exc:self.respond(400,{'error':str(exc)})
            elif url.path.startswith('/api/encounters'):
                try:
                    parts=url.path.strip('/').split('/')
                    if len(parts)==2:
                        q=parse_qs(url.query)
                        self.respond(200,encounters.candidates(q.get('car',[''])[0],float(q.get('at',['0'])[0])))
                    elif len(parts)==3:self.respond(200,encounters.summary(parts[2]))
                    elif len(parts)==4 and parts[3]=='replay':
                        query=parse_qs(url.query)
                        replay=encounters.replay(parts[2],query.get('kind')==['forecast'])
                        if query.get('preview')==['1']:
                            # The on-screen future needs one local branch, not two complete days.
                            branch=replay['branches']['concession'];end=replay['time']+35.
                            frames=[f for f in branch['frames'] if f['t']<=end+.0001]
                            replay={**replay,'branches':{'concession':{**branch,'frames':frames,
                                'termination':branch['termination'] if len(frames)==len(branch['frames']) else {'kind':'preview_horizon','time':frames[-1]['t']}}}}
                        body=json.dumps(replay,allow_nan=False,separators=(',',':')).encode()
                        self.respond(200,gzip.compress(body,mtime=0),'application/gzip')
                    else:self.respond(404,{'error':'Unknown encounter endpoint'})
                except (ValueError,TypeError,KeyError) as exc:self.respond(400,{'error':str(exc)})
            elif url.path=='/api/state':self.respond(200,session.snapshot())
            elif url.path=='/api/recordings':self.respond(200,{'runs':[p.parent.name for p in sorted(DATA.glob('*/day-001.json.gz'),reverse=True)]})
            elif url.path=='/api/recording':
                run=parse_qs(url.query).get('run',[None])[0]
                if run and (Path(run).name!=run or run in ('.','..')):return self.respond(400,{'error':'Invalid recording'})
                p=(DATA/run if run else session.output)/'day-001.json.gz'
                if p.exists():self.respond(200,p.read_bytes(),'application/gzip')
                else:self.respond(404,{'error':'Run a day first'})
            else:self.respond(404,{'error':'Not found'})
        def do_POST(self):
            origin=self.headers.get('Origin')
            if origin and origin not in (f'http://127.0.0.1:{args.port}',f'http://localhost:{args.port}'):return self.respond(403,{'error':'Local viewer requests only'})
            try:
                size=int(self.headers.get('Content-Length','0'))
                if self.path not in ('/api/control','/api/encounters','/api/demo') or not 0<size<=8192 or self.headers.get_content_type()!='application/json':raise ValueError('Expected a small JSON control request')
                payload=json.loads(self.rfile.read(size))
                if self.path=='/api/demo':
                    self.respond(200,demonstration.command(payload))
                elif self.path=='/api/encounters':
                    action=payload.get('action')
                    if action=='forecast':
                        if payload.get('require_actionable'):
                            reasoner.require_action(payload['car'],payload['at'],payload['candidate'],payload['snapshot'])
                        value=encounters.preview(payload['car'],payload['at'],payload['candidate'],payload['source'])
                        if value.get('baseline_audit',{}).get('passed'):
                            encounters.context.baseline_audit=deepcopy(value['baseline_audit'])
                        value['execution_evidence']=reasoner.execution_evidence(value)
                    elif action=='validate':value=encounters.validate(payload['id'])
                    elif action=='cancel':value=encounters.cancel(payload['id'])
                    else:raise ValueError('Unknown encounter action')
                    self.respond(200,value)
                elif payload.get('action')=='quit':self.respond(200,{'status':'closing'});threading.Thread(target=server.shutdown,daemon=True).start()
                else:self.respond(200,session.command(payload))
            except (ValueError,TypeError,KeyError,AttributeError) as exc:self.respond(400,{'error':str(exc)})
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Traffic workspace · http://127.0.0.1:{args.port}/layout?drive=1\nOne-day review; no training.\nRecordings: {session.output}',flush=True)
    if not args.no_browser:webbrowser.open(f'http://127.0.0.1:{args.port}/layout?drive=1')
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        demonstration.close()
        forecast_stream.close()
        if encounters.active:
            encounters.cancel(encounters.active)
            until=time.monotonic()+5.
            while encounters.active and time.monotonic()<until:time.sleep(.02)
            with encounters.lock:pass  # Let an in-progress atomic record write finish.
        server.server_close();session.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8771);p.add_argument('--output',type=Path);p.add_argument('--no-browser',action='store_true');serve(p.parse_args())

if __name__=='__main__':main()

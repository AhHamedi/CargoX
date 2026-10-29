"""A connected district with long point-to-point journeys and route alternatives."""
from __future__ import annotations
import math
import random
from .driving_roads import Network, nearest_on_path
from .network_driving import _join, SPEED, NEGOTIATION_RADIUS_M


def make_district(seed=17, case='contention', *, count=40, size=12, spacing=60.):
    if count<4 or count>40 or count%2:raise ValueError('Use an even population between four and forty')
    nodes=[i*spacing for i in range(size)]; end=nodes[-1]; entry=-spacing*(size+2)-(count*40 if case=='no-conflict' else 0)
    s={'version':3,'name':f'{count}-car connected district','seed':seed,'horizon':650.,
       'observation_radius_m':85.,'negotiation_radius_m':NEGOTIATION_RADIUS_M,
       'roads':[],'connections':[],'signals':[],
       'vehicles':[],'junctions':{},'route_options':{}}
    def road(name,points):s['roads'].append(dict(id=name,points=points,lanes=1,lane_width=5.,speed_limit=SPEED))
    def link(name,src,dst,points,priority=0,**kw):s['connections'].append(dict(id=name,**{'from':[src,0],'to':[dst,0]},points=points,width=5.,priority=priority,**kw))
    segments=[(entry,-15.)]+[(x+15.,x+spacing-15.) for x in nodes[:-1]]+[(end+15.,end+65.)]
    for j,y in enumerate(nodes):
        for i,(a,b) in enumerate(segments):
            road(f'h{j}_{i}',[[a,y],[b,y]])
            road(f'v{j}_{i}',[[y,a],[y,b]])
    def arc(x,y,kind):
        if kind=='WN':return [[x-15+15*math.cos(-math.pi/2+i*math.pi/24),y+15+15*math.sin(-math.pi/2+i*math.pi/24)] for i in range(13)]
        return [[x+15+15*math.cos(math.pi-i*math.pi/24),y-15+15*math.sin(math.pi-i*math.pi/24)] for i in range(13)]
    for col,x in enumerate(nodes):
        for row,y in enumerate(nodes):
            junction=f'J{col}_{row}';bounds=[x-15,y-15,x+15,y+15]
            s['junctions'][junction]={'center':[x,y],'bounds':bounds}
            for kind in ('WE','WN','SE','SN'):
                src=f'h{row}_{col}' if kind[0]=='W' else f'v{col}_{row}'
                dst=f'h{row}_{col+1}' if kind[1]=='E' else f'v{col}_{row+1}'
                points=[[x-15,y],[x+15,y]] if kind=='WE' else [[x,y-15],[x,y+15]] if kind=='SN' else arc(x,y,kind)
                link(junction+'_'+kind,src,dst,points,junction_id=junction,movement=kind,bounds=bounds)
    rng=random.Random(seed)
    lanes=list(range(2,min(size-2,8))) if size>=8 else list(range(size-1))
    cuts=(size//3,size//2,2*size//3) if size>=8 else (0,(size-1)//2,size-1)
    # Concentrate private origins in the inhabited district. The old outer
    # entry/route combination let many rides skirt almost all other traffic.
    for i in range(count):
        owner=f'C{i:02d}';direction='W' if i%2==0 else 'S';lane=lanes[(i//2)%len(lanes)];queue=(i//2)//len(lanes)
        start=-100.-lane*spacing-24*queue+rng.uniform(-2.,2.)
        if case=='no-conflict':start-=i*40.
        if direction=='W':pose=[start,nodes[lane]];heading=0.
        else:pose=[nodes[lane],start];heading=math.pi/2
        # Fan destinations are beyond the common streets; parked cars cannot block them.
        offset=15.+i*8.
        if direction=='W':
            bay=[[end+65+150*u,end-offset*(3*u*u-2*u*u*u)] for u in [k/32 for k in range(33)]]
            goal=[end+225,end-offset];last=f'h{size-1}_{size}'
        else:
            bay=[[end-offset*(3*u*u-2*u*u*u),end+65+150*u] for u in [k/32 for k in range(33)]]
            goal=[end-offset,end+225];last=f'v{size-1}_{size}'
        road('bay'+owner,[bay[-1],goal,[goal[0]+10 if direction=='W' else goal[0],goal[1]+10 if direction=='S' else goal[1]]])
        link('park'+owner,last,'bay'+owner,bay)
        options=[]
        for name,turn in zip(('early','middle','late'),cuts):
            route=[];junctions=[]
            if direction=='W':
                col,row=0,lane;route=[f'h{row}_0:0']
                while col<turn:
                    route += [f'J{col}_{row}_WE',f'h{row}_{col+1}:0'];junctions.append(f'J{col}_{row}');col+=1
                route += [f'J{col}_{row}_WN',f'v{col}_{row+1}:0'];junctions.append(f'J{col}_{row}');row+=1
                while row<size-1:
                    route += [f'J{col}_{row}_SN',f'v{col}_{row+1}:0'];junctions.append(f'J{col}_{row}');row+=1
                route += [f'J{col}_{row}_SE',f'h{row}_{col+1}:0'];junctions.append(f'J{col}_{row}');col+=1
                while col<size:
                    route += [f'J{col}_{row}_WE',f'h{row}_{col+1}:0'];junctions.append(f'J{col}_{row}');col+=1
            else:
                col,row=lane,0;route=[f'v{col}_0:0']
                while row<turn:
                    route += [f'J{col}_{row}_SN',f'v{col}_{row+1}:0'];junctions.append(f'J{col}_{row}');row+=1
                route += [f'J{col}_{row}_SE',f'h{row}_{col+1}:0'];junctions.append(f'J{col}_{row}');col+=1
                while col<size-1:
                    route += [f'J{col}_{row}_WE',f'h{row}_{col+1}:0'];junctions.append(f'J{col}_{row}');col+=1
                route += [f'J{col}_{row}_WN',f'v{col}_{row+1}:0'];junctions.append(f'J{col}_{row}');row+=1
                while row<size:
                    route += [f'J{col}_{row}_SN',f'v{col}_{row+1}:0'];junctions.append(f'J{col}_{row}');row+=1
            route += ['park'+owner,'bay'+owner+':0']
            options.append({'id':name,'path':route,'junctions':junctions})
        s['vehicles'].append(dict(id=owner,x=pose[0],y=pose[1],heading_rad=heading,speed_mps=SPEED,
            goal=goal,goal_radius=2.,cruise_cap_mps=SPEED,path=options[-1]['path'],route_choice='late'))
        s['route_options'][owner]=options
    network=Network(s)
    for car in s['vehicles']:
        for r in s['route_options'][car['id']]:
            pts=_join(network,r['path']);distance=nearest_on_path(pts,*car['goal'])[0]-nearest_on_path(pts,car['x'],car['y'])[0]
            r.update(distance_m=round(distance,3),free_seconds=round(distance/SPEED+2.4,3),
                     estimate_scope='optimistic free route; traffic delays unknown')
    s['horizon']=max(s['horizon'],max(r['free_seconds'] for rs in s['route_options'].values() for r in rs)+150)
    return s

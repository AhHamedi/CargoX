/* The viewer clock belongs to the player. Forecasts never pause or replace it. */
const $=id=>document.getElementById(id), NS='http://www.w3.org/2000/svg';
const svg=(tag,attrs,parent)=>{const e=document.createElementNS(NS,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);parent.append(e);return e};
const text=(e,value)=>{if(e.textContent!==value)e.textContent=value};
const wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const geo=await fetch('/layout-map.json').then(r=>r.json());
const whole=geo.view_box||[30,15,1140,760], map=$('map');
let view=[...whole], camera=[...whole], follows=false, zoomWidth=230, box=map.getBoundingClientRect();
let world=null,clock=0,running=false,selected=null,hovered=null,epoch=0,advanceBusy=false,optionsBusy=false,applying=false;
let committed=[];
let pendingCacheKey='',pendingCache=[];
const choiceNodes=new Map();
let options=null,optionKey='',hoverChoice=null,hoverToken=0,ghost=null,ghostStarted=0,ghostRefreshed=-1;
let bufferUnderruns=0,forecastLookahead=4;
let lastFrame=0,lastAdvance=0,lastOptions=0,noticeUntil=0,lastEventKey='',drag=null,metrics=null,networkProblem=false;
const bodies=new Map(), ghosts=new Map();
const names={L:['Northwood','West End','South Gardens'],R:['Eastfield','Market Quarter','Southbank']};
function neighbourhood(id){const n=geo.nodes[id];return n?.neighbourhood||names[n?.bank||'L'][n?.y<280?0:n?.y<530?1:2]}
function destination(id){const j=geo.journeys.find(j=>j.id===id);return neighbourhood(j.to)}
function drawMap(){
 if(geo.decorative_transform){document.querySelector('.river').setAttribute('transform',geo.decorative_transform);$('names').setAttribute('transform',geo.decorative_transform)}
 for(const layer of ['edge','surface'])for(const r of geo.roads){const a=geo.nodes[r.a],b=geo.nodes[r.b],edge=layer==='edge';svg('path',{d:`M${a.x} ${a.y}L${b.x} ${b.y}`,class:(edge?'street-edge':'street')+(r.bridge?(edge?' bridge-edge':' bridge'):'')},$('streets'))}
 const r=geo.roundabout;svg('circle',{cx:r.x,cy:r.y,r:r.island_radius??23,fill:'#e7eae3',stroke:'#d6dcd2','stroke-width':1},$('streets'));
 for(const[label,x,y]of [['Northwood',290,165],['West End',275,465],['South Gardens',290,640],['Eastfield',885,165],['Market Quarter',895,465],['Southbank',885,640]])svg('text',{x,y,'text-anchor':'middle',class:'district-name'},$('names')).textContent=label;
 for(const[label,y]of [['North crossing',73],['Central crossing',378],['South crossing',683]])svg('text',{x:595,y,'text-anchor':'middle',class:'crossing-name'},$('names')).textContent=label;
}
drawMap();
async function api(action,fields={}){
 const response=await fetch('/api/demo',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,revision:world?.revision,...fields})});
 const d=await response.json();if(!response.ok||d.status==='failed')throw Error(d.error||'The drive could not continue');return d;
}
function showNotice(message,kind='normal',seconds=4){text($('notice'),message);$('notice').dataset.kind=kind;$('notice').hidden=false;noticeUntil=performance.now()+seconds*1000}
function paths(points){return points.map((p,i)=>(i?'L':'M')+p.join(' ')).join(' ')}
function prepareCars(){
 $('cars').replaceChildren();$('pavement').replaceChildren();bodies.clear();
 for(const id of world.owners){const group=svg('g',{'data-car':id,role:'button',tabindex:0,'aria-label':`${id}, destination ${destination(id)}. Follow this car.`},$('cars'));const body=svg('rect',{x:-3,y:-1.4,width:6,height:2.8,rx:.8,class:'car-body'},group);svg('rect',{x:1,y:-1,width:.65,height:2,rx:.15,class:'car-glass'},group);group.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();chooseCar(id)}});group.addEventListener('focus',()=>hovered=id);group.addEventListener('blur',()=>hovered=null);bodies.set(id,{group,body});}
 for(const route of Object.values(world.routes))svg('path',{d:paths(route.points),class:'pavement'},$('pavement'));
}
function sample(frames,at){
 if(!frames?.length)return[];let lo=0,hi=frames.length-1;
 while(lo<hi){const mid=Math.ceil((lo+hi)/2);if(frames[mid].t<=at)lo=mid;else hi=mid-1}
 const a=frames[lo],b=frames[Math.min(lo+1,frames.length-1)],u=b.t>a.t?Math.max(0,Math.min(1,(at-a.t)/(b.t-a.t))):0;
 return a.cars.map((c,i)=>{const n=b.cars[i],h=Math.atan2(Math.sin(n[2]-c[2]),Math.cos(n[2]-c[2]));return[c[0]+u*(n[0]-c[0]),c[1]+u*(n[1]-c[1]),c[2]+u*h,c[3]+u*(n[3]-c[3]),...c.slice(4)]});
}
function routeFor(id){const r=world.routes[id];return r.svg||(r.svg=paths(r.points))}
function setRoute(id){
 if(!id){$('route').setAttribute('d','');$('destination').setAttribute('hidden','');$('destinationLabel').setAttribute('hidden','');return}
 const d=routeFor(id);if($('route').getAttribute('d')!==d)$('route').setAttribute('d',d);
 const goal=world.routes[id].goal||world.routes[id].points.at(-1);$('destination').removeAttribute('hidden');$('destinationLabel').removeAttribute('hidden');$('destination').setAttribute('cx',goal[0]);$('destination').setAttribute('cy',goal[1]);$('destinationLabel').setAttribute('x',goal[0]+8);$('destinationLabel').setAttribute('y',goal[1]-8);
}
function clearChoices(){choiceNodes.clear();$('choices').replaceChildren();$('opportunityPoints').replaceChildren();$('decisions').hidden=true;$('decisions').style.height='';optionKey=''}
function chooseCar(id){
 if(!world||!world.owners.includes(id))return;const initial=selected===null;selected=id;hovered=null;options=null;clearChoices();epoch++;clearGhost();follows=true;zoomWidth=230;$('follow').hidden=false;$('follow').setAttribute('aria-pressed','true');$('trip').hidden=false;$('intro').hidden=true;$('play').disabled=false;document.body.dataset.stage='driving';text($('carNumber'),id);text($('tripDestination'),destination(id));if(initial)running=true;setRoute(id);lastOptions=0;updateOptions();
}
function mergePacket(d){
 world.frames.push(...d.frames.filter(f=>f.t>world.frames.at(-1).t+.00001));world.events=d.events;world.arrivals=d.arrivals;world.ended=d.ended;world.failure=d.failure;world.time=d.time;
}
async function advance(){
 if(!world||advanceBusy||applying||world.ended)return;const speed=Number($('rate').value),ahead=world.frames.at(-1).t-clock;
 if(ahead>=Math.max(4,speed*2,forecastLookahead))return;advanceBusy=true;const rev=world.revision;
 const began=performance.now();try{const d=await api('advance',{to:Math.min(600,world.frames.at(-1).t+(ahead<Math.max(1.5,speed*.75)?1:2)),after:world.frames.at(-1).t});if(world?.revision===rev&&d.revision===rev){mergePacket(d);document.body.dataset.advanceMs=(performance.now()-began).toFixed(1);networkProblem=false}}catch(e){if(world?.revision===rev){networkProblem=true;showNotice(e.message,'error',5)}}finally{advanceBusy=false;if(world&&!networkProblem)setTimeout(advance,0)}
}
function timeEffect(value){
 const n=value?.seconds_saved;if(!Number.isFinite(n))return{value:'—',word:'pending',tone:'neutral'};
 if(Math.abs(n)<.15)return{value:'0s',word:'same',tone:'neutral'};
 const seconds=Math.abs(n);return{value:(seconds<1?seconds.toFixed(1):Math.round(seconds))+'s',word:n>0?'faster':'slower',tone:n>0?'gain':'loss'};
}
function allChoices(){
 const choices=new Map((options?.candidates||[]).map(c=>[c.id,{...c,actionable:true}]));
 for(const request of options?.incoming_requests||[]){
  if(!['open','checking'].includes(request.status)||request.recipient!==selected)continue;
  const id=request.candidate_id,base=choices.get(id)||{};
  choices.set(id,{...base,...request,id,request,effects:base.effects||request.effects||{},
   actionable:request.actionable===true,actors:base.actors||request.actors||[selected,request.requester,request.buyer,request.seller]});
 }
 return [...choices.values()];
}
function currentChoice(id=hoverChoice){return allChoices().find(c=>c.id===id)}
function actionActors(c){return [...new Set([...(c.actors||[]),...(c.beneficiaries||[]),c.initiator,c.seller,c.buyer].filter(id=>world.owners.includes(id)))]}
function pendingActions(){
 const key=[world?.revision,Math.floor(clock*10),world?.events?.length,committed.length].join(':');
 if(key!==pendingCacheKey){pendingCacheKey=key;pendingCache=committed.filter(a=>!(world.events||[]).some(e=>e.time<=clock+.001&&e.time>=a.start&&
  ['delivered','execution_fault'].includes(e.kind)&&e.zone_id===a.zone_id&&e.seller===a.seller&&e.buyer===a.buyer))}return pendingCache;
}
function visibleActions(){
 const hover=currentChoice();return hover?[hover]:pendingActions().filter(a=>actionActors(a).includes(selected));
}
function visibleAction(){return visibleActions()[0]||null}
function choiceTitle(c){
 const peer=c.seller===selected?c.buyer:c.seller;
 if(c.initiator===selected&&c.buyer!==selected&&c.seller!==selected)return c.action_type==='early_gap'?`${c.seller} keeps a gap for ${c.buyer}`:`${c.seller} lets ${c.buyer} through`;
 if(c.action_type==='early_gap')return c.seller===selected?`Keep a gap for ${c.buyer}`:`${c.seller} keeps a gap for you`;
 return c.seller===selected?`Let ${peer} through first`:`Go before ${peer}`;
}
function choiceKind(c){return c.request?'incoming':c.seller===selected?'offer':'outgoing'}
function choiceLabel(c){
 const kind=choiceKind(c),verb=kind==='incoming'?'Accept':kind==='offer'?'Offer':'Ask';
 return `${verb}: ${choiceTitle(c)}`;
}
function makeChoice(c){
 const row=document.createElement('div');row.className='choice-entry';
 const header=document.createElement('div');header.className='choice-head';
 const direction=document.createElement('span');direction.className='choice-direction';
 const arrow=document.createElement('span');arrow.className='direction-arrow';arrow.setAttribute('aria-hidden','true');
 const label=document.createElement('span');direction.append(arrow,label);
 const decline=document.createElement('button');decline.className='decline';decline.textContent='Decline';decline.onclick=()=>declineRequest(c.id);
 header.append(direction,decline);
 const button=document.createElement('button');button.className='choice';button.dataset.candidate=c.id;
 button.innerHTML='<span class="choice-action"><strong></strong><span class="choice-cue" aria-hidden="true"></span></span><span class="impacts"></span><span class="choice-check" hidden>Checking the gap…</span><span class="constraints" hidden></span>';
 button.addEventListener('pointerenter',()=>previewChoice(c.id));button.addEventListener('pointerleave',()=>{clearGhost();renderChoices()});
 button.addEventListener('focus',()=>previewChoice(c.id));button.addEventListener('blur',()=>{clearGhost();renderChoices()});button.addEventListener('click',()=>takeChoice(c.id));
 row.append(header,button);$('choices').append(row);
 const node={row,button,label,arrow,decline,impactKey:''};choiceNodes.set(c.id,node);return node;
}
function renderChoices(){
 if(!selected||!options)return;const choices=allChoices(),incoming=choices.filter(c=>c.request),ids=new Set(choices.map(c=>c.id));
 text($('decisionTitle'),'Your move');
 // Keep the same targets beneath the pointer as live estimates change.
 // Expired choices stay disabled until the player leaves this tray.
 const inspecting=[...choiceNodes.values()].some(n=>n.row.matches(':hover')||n.row.contains(document.activeElement));
 if(inspecting){if(!$('decisions').style.height)$('decisions').style.height=$('decisions').getBoundingClientRect().height+'px'}else $('decisions').style.height='';
 for(const[id,node]of choiceNodes){
  if(ids.has(id))continue;
  if(!inspecting){node.row.remove();choiceNodes.delete(id)}
  else{node.row.dataset.state='expired';node.button.disabled=true;node.decline.hidden=true;if(!applying)node.button.classList.remove('pending');text(node.button.querySelector('.choice-cue'),'Passed');node.button.setAttribute('aria-description','This opportunity has passed.');}
 }
 const key=JSON.stringify([world.revision,selected,choices.map(c=>c.id)]);
 if(key!==optionKey){optionKey=key;$('opportunityPoints').replaceChildren();for(const c of choices)if(c.point)svg('circle',{cx:c.point[0],cy:c.point[1],r:4,fill:'#c35b4830',stroke:'#c35b48','stroke-width':.8},$('opportunityPoints'));}
 for(const c of choices){
  const node=choiceNodes.get(c.id)||makeChoice(c),b=node.button,kind=choiceKind(c),requesters=(c.request?.requesters||[c.request?.requester]).filter(Boolean);
  if(!applying)b.classList.remove('pending');
  node.row.dataset.direction=kind;node.row.dataset.state=c.actionable?'ready':'checking';
  text(node.arrow,kind==='incoming'?'↙':'↗');
  text(node.label,kind==='incoming'?'Incoming · '+requesters[0]+(requesters.length>1?' +'+(requesters.length-1):''):kind==='offer'?'Your offer':'Your request');
  node.decline.hidden=!c.request;node.decline.disabled=applying;node.decline.setAttribute('aria-label','Decline request from '+requesters.join(', '));
  text(b.querySelector('strong'),choiceTitle(c));b.setAttribute('aria-label',choiceLabel(c));
  text(b.querySelector('.choice-cue'),c.actionable?(kind==='incoming'?'Accept':kind==='offer'?'Offer':'Ask')+' ›':'');
  const actors=[selected,...actionActors(c).filter(o=>o!==selected&&c.effects[o])];
  const impactKey=actors.join(':');if(node.impactKey!==impactKey){
   node.impactKey=impactKey;b.querySelector('.impacts').replaceChildren(...actors.map(id=>{
    const impact=document.createElement('span');impact.className='impact';impact.dataset.actor=id;
    impact.innerHTML='<span class="impact-who"></span><span class="impact-time"><b></b><span></span></span>';
    text(impact.querySelector('.impact-who'),id===selected?'You':id);return impact;
   }));
  }
  const effectLabels=[];
  for(const impact of b.querySelectorAll('.impact')){const id=impact.dataset.actor,e=timeEffect(c.effects[id]);impact.dataset.tone=e.tone;text(impact.querySelector('b'),e.value);text(impact.querySelector('.impact-time span'),e.word);effectLabels.push((id===selected?'You':id)+' '+e.value+' '+e.word)}
  b.querySelector('.impacts').hidden=!c.actionable;b.querySelector('.choice-check').hidden=c.actionable;
  b.setAttribute('aria-description',c.actionable?'Predicted time changes: '+effectLabels.join(', '):'Checking the gap');
  const others=[...new Set((c.remaining_constraints||[]).filter(r=>r.car===selected).map(r=>r.other))];
  const constraints=b.querySelector('.constraints');
  if(!inspecting||!constraints.hidden){text(constraints,others.length?'Still waits for '+others.join(', '):'');if(!inspecting)constraints.hidden=!others.length||!c.actionable;}
  if(others.length)b.setAttribute('aria-description',b.getAttribute('aria-description')+'. Still waits for '+others.join(', '));
  b.disabled=applying||!c.actionable;
 }
 $('decisions').hidden=!choiceNodes.size;
 document.body.dataset.incomingRequests=String(incoming.length);document.body.dataset.availableChoices=String(choices.length);
 if(hoverChoice&&!choices.some(c=>c.id===hoverChoice&&c.actionable))clearGhost();
 if($('rawDetails').open)text($('evidence'),JSON.stringify(options,null,2));
}
async function declineRequest(id){
 const choice=currentChoice(id),request=choice?.request;if(!request||applying)return;
 const rev=world.revision,owner=selected;epoch++;if(hoverChoice===id)clearGhost();
 try{
  const result=await api('decline',{car:owner,at:clock,request_id:request.request_id||request.id});
  if(world?.revision!==rev||selected!==owner)return;
  options={...options,candidates:(options.candidates||[]).filter(c=>c.id!==id),incoming_requests:(options.incoming_requests||[]).filter(r=>r.candidate_id!==id)};
  choiceNodes.get(id)?.row.remove();choiceNodes.delete(id);optionKey='';renderChoices();showNotice('Request declined');document.body.dataset.lastDeclined=request.request_id||request.id;
  if(result.events)world.events=result.events;lastOptions=0;
 }catch(e){showNotice(e.message,'error',4)}
}
async function updateOptions(){
 if(!world||!selected||optionsBusy||applying)return;optionsBusy=true;const rev=world.revision,owner=selected,token=epoch;
 try{const d=await api('options',{car:owner,at:Math.min(clock,world.frames.at(-1).t),pace:Number($('rate').value)});if(world?.revision!==rev||selected!==owner||epoch!==token||d.revision!==rev)return;
  document.body.dataset.forecastStats=JSON.stringify(d.stats||{});const cost=d.stats?.forecast_mean_job_seconds||3;forecastLookahead=Math.min(20,Math.max(4,Number($('rate').value)*(cost*1.4+1)));
  if(['ready','pending'].includes(d.status)&&d.time<=clock+.001){options=d;document.body.dataset.forecastTime=String(d.time);renderChoices();
   if(hoverChoice&&clock-ghostRefreshed>1.2)refreshGhost(hoverChoice,hoverToken);
  }
 }catch(e){if(rev===world?.revision&&!applying)text($('evidence'),e.message)}finally{optionsBusy=false}
}
function clearGhost(){
 hoverToken++;hoverChoice=null;ghost=null;ghostRefreshed=-1;$('ghostLayer').replaceChildren();ghosts.clear();$('actionPoint').setAttribute('hidden','');$('ghostCaption').hidden=true;document.body.dataset.ghostActive='false';document.body.dataset.ghostPeer='';
}
function previewChoice(id){if(applying||!currentChoice(id)?.actionable)return;if(hoverChoice===id)return;clearGhost();hoverChoice=id;const token=hoverToken;$('ghostCaption').hidden=false;text($('ghostCaption').querySelector('strong'),'Looking ahead…');text($('ghostCaption').querySelector('span'),'');const c=currentChoice();if(c){$('actionPoint').removeAttribute('hidden');$('actionPoint').setAttribute('cx',c.point[0]);$('actionPoint').setAttribute('cy',c.point[1])}refreshGhost(id,token)}
async function refreshGhost(id,token){
 const rev=world.revision,owner=selected,at=clock;ghostRefreshed=clock;
 try{for(let tries=0;tries<45;tries++){
  if(token!==hoverToken||hoverChoice!==id||rev!==world?.revision||selected!==owner)return;
  const d=await api('future',{car:owner,at:Math.min(at,world.frames.at(-1).t),candidate:id});
  if(token!==hoverToken||rev!==world?.revision||hoverChoice!==id)return;
  if(d.status==='ready'){
   const continueLoop=ghost!==null;ghost=d;if(!continueLoop)ghostStarted=performance.now();$('ghostLayer').replaceChildren();ghosts.clear();
   for(let i=0;i<d.owners.length;i++){const o=d.owners[i];svg('path',{d:paths(d.frames.map(f=>f.cars[i].slice(0,2))),class:'ghost-path'},$('ghostLayer'));const g=svg('g',{'data-future-car':o},$('ghostLayer'));svg('rect',{x:-3,y:-1.5,width:6,height:3,rx:.8,class:'ghost-car',fill:o===selected?'#137e9d':'#c35b48'},g);ghosts.set(o,g)}
   const peers=d.owners.filter(o=>o!==owner);const peer=peers[0];text($('ghostCaption').querySelector('strong'),'Future me');text($('ghostCaption').querySelector('span'),'8×');$('ghostCaption').setAttribute('aria-label','Possible future for '+[owner,...peers].join(', '));document.body.dataset.ghostActive='true';document.body.dataset.ghostPeer=peer;return;
  }
  await wait(110);
 }
 if(token===hoverToken)text($('ghostCaption').querySelector('strong'),'Future still calculating…');
 }catch(e){if(token===hoverToken){text($('ghostCaption').querySelector('strong'),'Forecast unavailable');text($('ghostCaption').querySelector('span'),e.message)}}
}
async function takeChoice(id){
 if(applying||!currentChoice(id)?.actionable)return;const choice=currentChoice(id),rev=world.revision,owner=selected;
 applying=true;epoch++;clearGhost();for(const {row,button} of choiceNodes.values()){button.disabled=true;button.classList.toggle('pending',button.dataset.candidate===id);const decline=row.querySelector('.decline');if(decline)decline.disabled=true}
 try{
  const d=await api('apply',{car:owner,at:clock,candidate:id,request_id:choice.request?.request_id||choice.request?.id,pace:Number($('rate').value)});if(world?.revision!==rev)return;
  // Acknowledge a new world, never adopt a ghost clip or rewind the player.
  const prefix=world.frames.filter(f=>f.t<d.at-.00001);world={...world,...d,frames:[...prefix,...d.frames]};committed.push({...choice,start:d.at});options=null;clearChoices();
  showNotice(`${choice.seller} will yield to ${choice.buyer}`);document.body.dataset.lastApplied=id;document.body.dataset.lastAppliedAt=String(d.at);document.body.dataset.lastAppliedSpeed=$('rate').value;networkProblem=false;
 }catch(e){showNotice(e.message,'error',5)}finally{applying=false;renderChoices();lastOptions=0;advance();updateOptions()}
}
function cameraTarget(cars){
 if(!follows||!selected)return;
 const own=cars[world.owners.indexOf(selected)];let cx=own[0]+Math.cos(own[2])*17,cy=own[1]+Math.sin(own[2])*17,w=zoomWidth;
 const choice=visibleAction();if(choice){const pts=[own,...actionActors(choice).filter(id=>id!==selected).map(id=>cars[world.owners.indexOf(id)]),choice.point].filter(Boolean);const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);cx=(Math.min(...xs)+Math.max(...xs))/2;cy=(Math.min(...ys)+Math.max(...ys))/2;w=Math.max(w,Math.max(...xs)-Math.min(...xs)+55,(Math.max(...ys)-Math.min(...ys)+70)*box.width/Math.max(1,box.height))}
 const h=w*box.height/Math.max(1,box.width);camera=[cx-w/2,cy-h*.42,w,h];
}
function noticeEvents(){
 const e=[...(world.events||[])].reverse().find(e=>e.time<=clock+.001&&['delivered','execution_fault','physical_fault'].includes(e.kind)&&
  (e.kind!=='delivered'||clock-e.time<6)&&(e.kind!=='delivered'||e.seller===selected||e.buyer===selected||committed.some(a=>a.seller===e.seller&&a.buyer===e.buyer&&a.zone_id===e.zone_id&&actionActors(a).includes(selected))));
 if(e){const key=e.kind+':'+e.time;if(key!==lastEventKey){lastEventKey=key;if(e.kind==='delivered')showNotice(`${e.buyer} passed · ${e.seller} continues`, 'normal',4);else showNotice('Drive stopped: '+(world.failure||e.kind),'error',30)}}
}
function render(stamp,dt){
 if(!world)return;const cars=sample(world.frames,clock);cameraTarget(cars);const cursor=hovered?'pointer':drag?'grabbing':follows?'default':'grab';if(map.style.cursor!==cursor)map.style.cursor=cursor;
 const alpha=1-Math.exp(-dt*5.5);for(let i=0;i<4;i++)view[i]+=(camera[i]-view[i])*alpha;map.setAttribute('viewBox',view.join(' '));
 const pixel=Math.max(view[2]/Math.max(1,box.width),view[3]/Math.max(1,box.height)),choice=visibleAction(),peers=choice?actionActors(choice).filter(id=>id!==selected):[],peer=peers[0];
 world.owners.forEach((id,i)=>{const c=cars[i],m=bodies.get(id);m.group.setAttribute('transform',`translate(${c[0]} ${c[1]}) rotate(${c[2]*180/Math.PI})`);m.body.setAttribute('fill',id===selected?'#137e9d':peers.includes(id)?'#c35b48':id===hovered?'#37879a':c[4]==='arrived'?'#b7c3ae':'#536c5c');m.body.setAttribute('width',Math.max(6,7*pixel));m.body.setAttribute('height',Math.max(2.8,3.6*pixel));m.body.setAttribute('x',-Math.max(6,7*pixel)/2);m.body.setAttribute('y',-Math.max(2.8,3.6*pixel)/2);});
 if(selected){const c=cars[world.owners.indexOf(selected)];$('selectedLabel').removeAttribute('hidden');$('selectedLabel').setAttribute('x',c[0]+10*pixel);$('selectedLabel').setAttribute('y',c[1]-10*pixel);$('selectedLabel').style.fontSize=12*pixel+'px';text($('selectedLabel'),selected);
  const arrivedAt=world.arrivals?.[selected];const reason=c[4]==='arrived'?'Arrived'+(Number.isFinite(arrivedAt)?' · '+Math.floor(arrivedAt/60)+':'+String(Math.floor(arrivedAt%60)).padStart(2,'0'):''):c[5]==='parked_vehicle'?`Blocked by parked ${c[6]}`:c[5]==='following'?`Following ${c[6]}`:['yielding','roundabout'].includes(c[5])?`Giving way to ${c[6]}`:'Driving';text($('drivingState'),reason+(c[4]==='arrived'?'':` · ${Math.round(c[3]*3.6)} km/h`));
  const lastContract=pendingActions().find(a=>a.start<=clock&&actionActors(a).includes(selected));if(lastContract&&noticeUntil<stamp){text($('notice'),`${lastContract.seller} is yielding to ${lastContract.buyer}`);$('notice').hidden=false;$('notice').dataset.kind='normal'}
 }
 if(choice){$('actionPoint').removeAttribute('hidden');$('actionPoint').setAttribute('cx',choice.point[0]);$('actionPoint').setAttribute('cy',choice.point[1]);$('actionPoint').style.strokeDasharray=hoverChoice?'3 2':'none'}else $('actionPoint').setAttribute('hidden','');
 const labelKey=peers.join(',');if($('peerLabels').dataset.actors!==labelKey){$('peerLabels').replaceChildren(...peers.map(id=>{const label=document.createElementNS(NS,'text');label.setAttribute('class','pin-label');label.setAttribute('fill','#c35b48');label.dataset.actor=id;label.textContent=id;return label}));$('peerLabels').dataset.actors=labelKey}for(const [i,label] of [...$('peerLabels').children].entries()){const c=cars[world.owners.indexOf(label.dataset.actor)];label.setAttribute('x',c[0]+9*pixel);label.setAttribute('y',c[1]+(i%2?-13:15)*pixel);label.style.fontSize=12*pixel+'px'}
 const goalId=hovered||selected;setRoute(goalId);$('destinationLabel').style.fontSize=10*pixel+'px';$('destination').setAttribute('r',5*pixel);
 if(hovered){const c=cars[world.owners.indexOf(hovered)],p=new DOMPoint(c[0],c[1]).matrixTransform(map.getScreenCTM());$('carTooltip').hidden=false;$('carTooltip').style.left=Math.max(80,Math.min(box.width-80,p.x-box.left))+'px';$('carTooltip').style.top=Math.max(55,p.y-box.top-16)+'px';text($('carTooltip').querySelector('strong'),hovered);text($('carTooltip').querySelector('span'),'→ '+destination(hovered))}else $('carTooltip').hidden=true;
 if(ghost){const duration=Math.max(.2,ghost.frames.at(-1).t-ghost.time),at=ghost.time+((stamp-ghostStarted)/1000*8)%duration;const future=sample(ghost.frames,at);ghost.owners.forEach((id,i)=>{const c=future[i];ghosts.get(id).setAttribute('transform',`translate(${c[0]} ${c[1]}) rotate(${c[2]*180/Math.PI})`)});}
 if(options&&clock-options.time>10.2){options=null;clearChoices();clearGhost()}
 if(noticeUntil&&stamp>noticeUntil&&!pendingActions().some(a=>a.start<=clock&&actionActors(a).includes(selected))){$('notice').hidden=true;noticeUntil=0}
 noticeEvents();
 const time=`${String(Math.floor(clock/60)).padStart(2,'0')}:${String(Math.floor(clock%60)).padStart(2,'0')}`;text($('clock'),time);text($('play'),running?'Pause':world.ended&&clock>=world.frames.at(-1).t-.001?'Finished':selected?'Play':'Drive');$('play').dataset.playing=String(running);
 document.body.dataset.activeDeals=String(pendingActions().filter(a=>a.start<=clock).length);document.body.dataset.demoTime=clock.toFixed(4);document.body.dataset.demoRevision=String(world.revision);document.body.dataset.selectedCar=selected||'';document.body.dataset.playing=String(running);
}
function measure(stamp){
 if(!running||document.hidden){metrics=null;return}if(!metrics){metrics={start:stamp,clock,gaps:[],last:stamp};return}metrics.gaps.push(stamp-metrics.last);metrics.last=stamp;
 if(stamp-metrics.start>=2000){const gaps=[...metrics.gaps].sort((a,b)=>a-b);const d={fps:gaps.length*1000/(stamp-metrics.start),p95FrameMs:gaps[Math.floor(gaps.length*.95)],maxFrameMs:gaps.at(-1),simulatedSeconds:clock-metrics.clock,wallSeconds:(stamp-metrics.start)/1000,rate:Number($('rate').value)};const h=JSON.parse(document.body.dataset.playbackHistory||'[]');h.push(d);document.body.dataset.playbackHistory=JSON.stringify(h.slice(-32));metrics=null;}
}
function tick(stamp){
 const dt=lastFrame?Math.min(.1,(stamp-lastFrame)/1000):0;lastFrame=stamp;
 if(world){if(running){const latest=world.frames.at(-1).t;const wanted=clock+dt*Number($('rate').value);if(wanted>latest+.001&&!world.ended){bufferUnderruns++;document.body.dataset.bufferUnderruns=String(bufferUnderruns)}clock=Math.min(latest,wanted);if(world.ended&&clock>=latest-.001)running=false}render(stamp,dt);measure(stamp);
 if(stamp-lastAdvance>160){lastAdvance=stamp;advance()}
 if(selected&&stamp-lastOptions>(running?240:700)){lastOptions=stamp;updateOptions()}}
 requestAnimationFrame(tick);
}
function pointerCar(e){
 const p=new DOMPoint(e.clientX,e.clientY).matrixTransform(map.getScreenCTM().inverse()),cars=sample(world.frames,clock),limit=Math.max(7,10*view[2]/box.width);let nearest=null,d=limit;
 for(let i=0;i<cars.length;i++){const dist=Math.hypot(cars[i][0]-p.x,cars[i][1]-p.y);if(dist<d){d=dist;nearest=world.owners[i]}}return nearest;
}
map.addEventListener('pointermove',e=>{if(!world)return;if(drag){const scale=view[2]/box.width;camera=[drag.view[0]-(e.clientX-drag.x)*scale,drag.view[1]-(e.clientY-drag.y)*scale,drag.view[2],drag.view[3]];view=[...camera];if(Math.hypot(e.clientX-drag.x,e.clientY-drag.y)>4)drag.moved=true;return}hovered=pointerCar(e)});
map.addEventListener('pointerleave',()=>hovered=null);
map.addEventListener('pointerdown',e=>{if(!world)return;if(!pointerCar(e)){drag={x:e.clientX,y:e.clientY,view:[...view],moved:false};follows=false;$('follow').setAttribute('aria-pressed','false');map.setPointerCapture(e.pointerId)}});
map.addEventListener('pointerup',e=>{if(!world)return;if(drag){const moved=drag.moved;drag=null;if(moved)return}const id=pointerCar(e);if(id)chooseCar(id)});map.addEventListener('pointercancel',()=>drag=null);
$('play').onclick=()=>{if(selected){running=!running;lastFrame=0}};
$('follow').onclick=()=>{follows=!follows;$('follow').setAttribute('aria-pressed',String(follows))};
function zoom(scale){zoomWidth=Math.max(80,Math.min(850,(follows?zoomWidth:view[2])*scale));if(!follows){const w=zoomWidth,h=w*box.height/box.width;camera=[view[0]+(view[2]-w)/2,view[1]+(view[3]-h)/2,w,h]}}
$('zoomIn').onclick=()=>zoom(.75);$('zoomOut').onclick=()=>zoom(1/.75);$('whole').onclick=()=>{follows=false;camera=[...whole];$('follow').setAttribute('aria-pressed','false')};
$('help').onclick=()=>{$('details').hidden=!$('details').hidden;$('help').setAttribute('aria-expanded',String(!$('details').hidden))};$('rawDetails').ontoggle=()=>{if($('rawDetails').open)text($('evidence'),JSON.stringify(options,null,2))};
$('rate').onchange=()=>metrics=null;window.addEventListener('resize',()=>box=map.getBoundingClientRect());
async function reset(){
 const token=++epoch;running=false;world=null;committed=[];pendingCacheKey='';pendingCache=[];clock=0;delete document.body.dataset.lastApplied;delete document.body.dataset.lastAppliedAt;delete document.body.dataset.lastAppliedSpeed;$('rawDetails').open=false;text($('evidence'),'Select a car to inspect its current forecast.');text($('clock'),'00:00');setRoute(null);$('cars').replaceChildren();bodies.clear();text($('intro').querySelector('h1'),'Preparing the drive…');selected=null;hovered=null;options=null;clearChoices();clearGhost();$('play').disabled=true;$('trip').hidden=true;$('intro').hidden=false;$('follow').hidden=true;$('notice').hidden=true;$('selectedLabel').setAttribute('hidden','');$('details').hidden=true;$('help').setAttribute('aria-expanded','false');document.body.dataset.stage='choosing';
 try{const d=await api('new');if(token!==epoch)return;const initial=await api('advance',{revision:d.revision,to:4,after:0});if(token!==epoch)return;world={...d,...initial,frames:[...d.frames,...initial.frames]};text($('intro').querySelector('h1'),'Pick a car to follow');clock=0;bufferUnderruns=0;forecastLookahead=4;document.body.dataset.bufferUnderruns='0';document.body.dataset.playbackHistory='[]';camera=[...whole];view=[...whole];follows=false;lastEventKey='';noticeUntil=0;metrics=null;prepareCars();setRoute(null);advance()}
 catch(e){showNotice(e.message,'error',60)}
}
$('restart').onclick=reset;
await reset();requestAnimationFrame(tick);

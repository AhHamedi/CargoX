export function installEncounterInspector(view) {
  const $=id=>document.getElementById(id);
  let epoch=0,listing=null,result=null,open=false,activeOwner=null,selected='normal',timer=null;
  let refreshTimer=null,polling=false,listSignature='',mainSuspended=false,evidenceData=null;
  let busy=false,mode='live',resumeMain=false,previewCandidate=null;
  const clock=t=>String(Math.floor(t/60)).padStart(2,'0')+':'+String(Math.floor(t%60)).padStart(2,'0');
  const status=t=>$('encounterStatus').textContent=t;
  function controls(on){$('drive').disabled=!on&&mode==='checking';$('progress').disabled=!on&&mode==='checking'}
  function writeEvidence(data){evidenceData=data;if($('calcDetails').open)$('calculation').textContent=JSON.stringify(data,null,2)}
  $('calcDetails').addEventListener('toggle',()=>{if($('calcDetails').open&&evidenceData)$('calculation').textContent=JSON.stringify(evidenceData,null,2)});
  const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
  function stopRefresh(){if(refreshTimer)clearTimeout(refreshTimer);refreshTimer=null}
  async function request(body){const r=await fetch('/api/encounters',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const d=await r.json();if(!r.ok)throw Error(d.error||'Could not calculate this move');return d}
  function cancel(){if(timer)clearTimeout(timer);timer=null;if(result&&(result.status==='running'||mode==='checking'))request({action:'cancel',id:result.id}).catch(()=>{})}
  function clearResult(){for(const id of ['actionEffect','verifyPassage','calcDetails','thirdParty','replayMove'])$(id).hidden=true;$('executionProof').textContent='';$('calculation').textContent=''}
  function markSelected(){for(const b of $('actionOptions').children)b.setAttribute('aria-pressed',String(b.dataset.option===selected))}
  function optionButton(id,title,hint){const b=document.createElement('button');b.className='action-option';b.dataset.option=id;b.setAttribute('aria-pressed','false');b.disabled=false;const strong=document.createElement('strong');strong.textContent=title;const small=document.createElement('span');small.textContent=hint;b.append(strong,small);b.onclick=()=>choose(id);return b}
  function renderOptions(d){
    const defaults=[...new Set(d.normal_order.filter(o=>o.second===activeOwner).map(o=>o.first))];
    const normal=optionButton('normal','Keep driving',defaults.length?'Normal: '+defaults.join(', ')+' goes first':'Follow normal road priority');
    const signature=JSON.stringify([normal.innerText,d.candidates.map(c=>[c.id,[...new Set(c.remaining_constraints.map(r=>r.other))]])]);
    if(signature!==listSignature){listSignature=signature;$('actionOptions').replaceChildren(normal,...d.candidates.map(c=>{
      const title=c.seller===activeOwner?'Yield my turn to '+c.buyer:'Ask '+c.seller+' to yield this crossing';
      const others=[...new Set(c.remaining_constraints.filter(q=>q.car===activeOwner).map(q=>q.other))];
      return optionButton(c.id,title,others.length?'Still constrained by '+others.join(', '):c.buyer+' goes before '+c.seller);
    }));}
    if(selected!=='normal'&&!d.candidates.some(c=>c.id===selected))selected='normal';markSelected();
    $('actionTime').hidden=false;$('actionTime').textContent='Forecast · '+clock(d.time);
  }
  function setChoicesEnabled(on){for(const b of $('actionOptions').children)b.disabled=!on}
  function updateAge(){
    if(!open||mode!=='live'||!listing)return;
    const now=view.snapshot().at;
    if(listing.time>now+.001||now-listing.time>2){listing=null;listSignature='';result=null;clearResult();selected='normal';$('actionOptions').replaceChildren(optionButton('normal','Keep driving','Follow normal road priority'));markSelected();$('actionTime').textContent='Updating…';status('Updating available moves…');return}
    const age=Math.max(0,view.snapshot().at-listing.time);document.body.dataset.forecastAge=age.toFixed(3);
    const label='Forecast · '+clock(listing.time)+(age>=.5?' · '+age.toFixed(1)+' s old':'');
    if($('actionTime').textContent!==label)$('actionTime').textContent=label;
    for(const button of $('actionOptions').children)button.disabled=button.dataset.option!=='normal'&&age>2;
    if(age>2)status('Updating available moves…');
  }
  async function stream(owner,at,exact=false){
    const r=await fetch('/api/forecast-stream?at='+at+'&rate='+view.rate()+(owner?'&car='+encodeURIComponent(owner):'')+(exact?'&exact=1':'')+(!view.isPlaying()?'&current=1':''));
    const d=await r.json();if(!r.ok||d.status==='failed')throw Error(d.error||'Forecast worker failed');return d;
  }
  async function waitAssessment(owner,at,current){
    const began=performance.now();
    while(current===epoch&&performance.now()-began<30000){const d=await stream(owner,at,true);if(d.status==='ready')return d.result;await delay(70)}
    if(current===epoch)throw Error('This forecast is taking longer than expected. Try the move again.');return null;
  }
  async function refresh(){
    stopRefresh();if(mode!=='live'||!view.ready())return null;
    if(polling){refreshTimer=setTimeout(refresh,100);return null}
    polling=true;const current=epoch,owner=open?activeOwner:null,snapshot=view.snapshot();
    try{
      const packet=await stream(owner,snapshot.at);
      if(current!==epoch||mode!=='live')return null;
      document.body.dataset.forecastMetrics=JSON.stringify(packet.stats);document.body.dataset.forecastWorkers=String(packet.stats.workers);document.body.dataset.forecastPending=String(packet.stats.pending);document.body.dataset.forecastCompleted=String(packet.stats.completed);
      if(packet.status==='ready'){
        const d=packet.result;if(d.source!==snapshot.source)throw Error('The recording changed. Reload the day.');
        // A prepared future observation is never displayed before its logical time.
        if(d.time>view.snapshot().at)return null;
        document.body.dataset.reasoningAt=String(d.time);document.body.dataset.reasoningCars=String(d.cars_assessed);
        if(open){
          const eligible=new Set(packet.eligible_now||d.candidates.map(c=>c.id));
          const next={...d,candidates:d.candidates.filter(c=>eligible.has(c.id))};
          if(!listing||next.snapshot!==listing.snapshot||JSON.stringify(next.candidates.map(c=>c.id))!==JSON.stringify(listing.candidates.map(c=>c.id))){
            listing=next;renderOptions(next);if(result&&result.time!==next.time){cancel();result=null;clearResult()}
            if(!result){$('calcDetails').hidden=false;writeEvidence({current_reasoning:next,stream:packet.stats});const queue=next.rejected.find(r=>r.buyer_queued_behind_seller);status(next.candidates.length?'':queue?queue.buyer+' would queue behind '+queue.seller+'; this move cannot put it ahead.':'No feasible change to passage order now.')}
          }
          $('scope').textContent='';updateAge();
        }else $('scope').textContent='Click a car to explore its next move';
      }else if(open&&!listing)status('Calculating available moves…');
      return packet;
    }catch(e){if(current===epoch&&mode==='live'){if(open)status(e.message);$('scope').textContent='Forecasts unavailable; driving continues'}}
    finally{polling=false;if(current===epoch&&mode==='live')refreshTimer=setTimeout(refresh,view.isPlaying()?120:450)}
  }
  function close(){
    epoch++;stopRefresh();cancel();if(mainSuspended){view.pause();view.restore();mainSuspended=false;if(resumeMain)view.play()}
    open=false;listing=result=null;mode='live';busy=false;listSignature='';controls(true);$('encounterPanel').hidden=true;$('closeEncounter').hidden=true;$('actionTime').hidden=true;$('backToJourney').hidden=true;$('actionOptions').hidden=false;document.body.classList.remove('encounter-open');$('scope').textContent='Click a car to explore its next move';$('previewCaption').hidden=true;$('previewEvent').textContent='';refresh();
  }
  async function explore(){
    if(!view.ready())return;if(open)close();epoch++;stopRefresh();open=true;mode='live';activeOwner=view.snapshot().car;listing=result=null;selected='normal';listSignature='';clearResult();$('closeEncounter').hidden=false;$('encounterPanel').hidden=false;document.body.classList.add('encounter-open');$('actionOptions').hidden=false;$('actionOptions').replaceChildren();status('Calculating available moves…');return refresh();
  }
  function effectText(v,where){if(v==null)return 'Not resolved';if(Math.abs(v)<.05)return 'No change '+where;return Math.abs(v).toFixed(1)+' s '+(v>0?'earlier':'later')+' '+where}
  function paint(){
    if(!result)return;const c=result.candidate,actors=[activeOwner,...[c.seller,c.buyer].filter(o=>o!==activeOwner)],checked=result.status==='complete';
    const failed=!checked&&(['failed','cancelled'].includes(result.status)||['delivery_timeout','violated_order'].includes(result.forecast.concession_status));
    $('actionEffect').hidden=failed;$('effectKind').textContent=(mode==='live'?'Last preview · ':'')+(checked?'measured arrival effect':'estimated crossing effect');
    $('effectActors').replaceChildren(...actors.map(o=>{const row=document.createElement('div'),id=document.createElement('strong'),value=document.createElement('span');id.textContent=o;id.style.color=o===activeOwner?'var(--blue)':'var(--amber)';const v=checked?result.measured[o].destination_seconds_saved:!failed&&result.forecast.concession_status==='delivered'?result.forecast.outcomes[o].passage_seconds_saved:null;value.textContent=effectText(v,checked?'at arrival':'at crossing');value.className=v==null||Math.abs(v)<.05?'':v>0?'gain':'loss';row.append(id,value);return row}));
    $('verifyPassage').hidden=checked||['failed','cancelled'].includes(result.status);$('verifyPassage').disabled=result.status==='running';
    if(result.status==='running'){const p=result.progress;status('Checking full journey'+(p?' · '+Math.round(p.time-result.time)+' seconds simulated.':'…'))}else if(failed)status('This passing order did not complete. No benefit is claimed.');else status(checked?'':'Estimated passage order only; arrival effects need a physical check.');
    const proof=result.execution_evidence;
    if(proof){const buyer=proof.timing[c.buyer].clears,seller=proof.timing[c.seller].enters;$('executionProof').textContent=proof.order_observed?(mode==='live'?'In the preview: ':'')+c.buyer+' clears ~'+buyer.toFixed(2)+' s; '+c.seller+(seller==null?' stays behind.':' enters ~'+seller.toFixed(2)+' s.'):'This preview does not demonstrate delivery.'}
    $('calcDetails').hidden=false;writeEvidence({reasoning:listing,action:result.candidate,execution:proof,forecast:result.forecast,measured:result.measured,physical:{status:result.status,termination:result.termination,error:result.error,baseline_audit:result.baseline_audit,seconds:result.validation_seconds},scope:result.scope});
    $('thirdParty').hidden=!checked;
    if(checked){const effects=Object.entries(result.measured).filter(([o,m])=>!actors.includes(o)&&m.destination_seconds_saved!=null&&Math.abs(m.destination_seconds_saved)>.05);$('thirdPartyLabel').textContent='Other cars · '+effects.filter(([,m])=>m.destination_seconds_saved<0).length+' later, '+effects.filter(([,m])=>m.destination_seconds_saved>0).length+' earlier';$('thirdPartyRows').replaceChildren(...effects.sort((a,b)=>a[1].destination_seconds_saved-b[1].destination_seconds_saved).map(([o,m])=>{const tr=document.createElement('tr');for(const t of [o,effectText(m.destination_seconds_saved,'at arrival')]){const td=document.createElement('td');td.textContent=t;tr.append(td)}return tr}))}
    $('replayMove').hidden=mode!=='live';
  }
  async function loadReplay(current){
    const physical=result.status==='complete'||(result.status==='failed'&&result.termination);const r=await fetch('/api/encounters/'+result.id+'/replay?preview=1'+(physical?'':'&kind=forecast'));if(!r.ok)throw Error('Could not load this future');const data=await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).json();if(current!==epoch)return;if(data.source!==listing.source)throw Error('Recording changed; reopen this car');
    const heldAt=view.snapshot().at;if(!mainSuspended){resumeMain=view.isPlaying();view.remember();mainSuspended=true}view.pause();mode='preview';busy=false;previewCandidate=result.candidate;$('actionOptions').hidden=true;$('backToJourney').hidden=false;$('actionTime').textContent='Preview from '+clock(result.time);view.compare(data,'concession');view.markPassage(previewCandidate.point);controls(true);$('previewCaption').hidden=false;$('previewCaption').textContent='What if '+previewCandidate.seller+' yields to '+previewCandidate.buyer+'?';$('scope').textContent=(physical?'Physical replay':'Estimated future')+' · reference held at '+clock(heldAt);
    const out=result.forecast.outcomes;const clears=Object.values(out).map(v=>v.concession_clear).filter(v=>v!=null);const delivered=data.branches.concession.delivered_at;const end=delivered?delivered+6:clears.length?Math.max(...clears)+5:result.time+25;view.playPreview(Math.min(result.time+35,Math.max(result.time+8,end)));paint();
  }
  async function choose(id){
    if(busy||mode!=='live')return;
    if(id==='normal'){cancel();result=null;clearResult();selected='normal';markSelected();$('previewCaption').hidden=true;status('');view.play();return}
    if(!listing)return;
    const snapshot=view.snapshot(),current=++epoch;stopRefresh();busy=true;mode='preparing';selected=id;markSelected();$('backToJourney').hidden=false;status('Preparing this future; traffic continues…');
    try{
      const d=await waitAssessment(activeOwner,snapshot.at,current);if(!d||current!==epoch)return;
      const action=d.candidates.find(a=>a.id===id);if(!action){mode='live';busy=false;status('That move is no longer available at this point.');refresh();return}
      listing=d;clearResult();
      const r=await request({action:'forecast',car:activeOwner,at:d.time,snapshot:d.snapshot,candidate:id,source:d.source,require_actionable:true});if(current!==epoch)return;result=r;paint();await loadReplay(current);if(result.status==='running')poll(current);
    }catch(e){if(current===epoch){mode='live';busy=false;status(e.message);refresh()}}
  }
  async function returnLive(resume){
    if(!open)return;epoch++;stopRefresh();if(mode==='checking')cancel();
    const restorePlay=resume??resumeMain;
    if(mainSuspended){view.pause();view.restore();mainSuspended=false;if(restorePlay)view.play()}
    mode='live';busy=false;selected='normal';previewCandidate=null;$('actionOptions').hidden=false;$('backToJourney').hidden=true;$('previewCaption').hidden=true;$('previewEvent').textContent='';if(result)paint();return refresh();
  }
  async function poll(current){if(current!==epoch||!result)return;try{const r=await fetch('/api/encounters/'+result.id);const d=await r.json();if(!r.ok)throw Error(d.error);if(current!==epoch)return;const proof=result.execution_evidence;result=d;result.execution_evidence=proof;paint();if(d.status==='running')timer=setTimeout(()=>poll(current),900);else if(d.status==='complete'||(d.status==='failed'&&d.termination)){const details=await request({action:'forecast',car:activeOwner,at:result.time,candidate:result.candidate.id,source:result.source});if(current!==epoch)return;result=details;await loadReplay(current)}}catch(e){if(current===epoch){status(e.message);$('verifyPassage').disabled=false}}}
  $('verifyPassage').onclick=async()=>{if(!result)return;const current=++epoch;stopRefresh();mode='checking';$('actionOptions').hidden=true;$('backToJourney').hidden=false;$('verifyPassage').disabled=true;try{const d=await request({action:'validate',id:result.id});if(current!==epoch){await request({action:'cancel',id:d.id});return;}d.execution_evidence=result.execution_evidence;result=d;paint();poll(current)}catch(e){if(current===epoch){status(e.message);$('verifyPassage').disabled=false}}};
  $('replayMove').onclick=async()=>{if(!result||mode!=='live')return;const current=++epoch;stopRefresh();busy=true;try{await loadReplay(current)}catch(e){status(e.message);await returnLive()}};
  $('backToJourney').onclick=()=>returnLive();$('closeEncounter').onclick=close;$('layout').addEventListener('click',()=>{if(open)close()},{capture:true});
  window.addEventListener('carselected',()=>{if(view.interactive()&&(!open||view.snapshot().car!==activeOwner))explore()});
  window.addEventListener('previewfinished',()=>{if(mode==='preview')returnLive()});
  window.addEventListener('previewframe',e=>{
    if(!view.interactive())return;
    if(mode==='live'){updateAge();if(!refreshTimer&&!polling)refresh()}
    else if(mode==='preview'&&previewCandidate){const c=previewCandidate;const describe=row=>row[4]==='arrived'?'arrived':['yielding','roundabout'].includes(row[5])?'waiting for '+row[6]:row[3]<.4?'waiting':row[5]==='following'?'following '+row[6]:'moving';$('previewEvent').textContent=e.detail.filter(([o])=>[c.seller,c.buyer].includes(o)).map(([o,row])=>o+' '+describe(row)).join(' · ')}
  });
  $('actionOptions').addEventListener('keydown',e=>{if(busy||mode!=='live'||!['ArrowUp','ArrowDown'].includes(e.key))return;const buttons=[...$('actionOptions').children],i=buttons.indexOf(document.activeElement);if(i<0)return;e.preventDefault();const next=buttons[(i+(e.key==='ArrowDown'?1:-1)+buttons.length)%buttons.length];next.focus();choose(next.dataset.option)});
  document.addEventListener('keydown',e=>{if(e.key==='Escape'){if(mode==='preview'||mode==='preparing'||mode==='checking')returnLive();else if(open)close()}});
  return {open:explore,preview:async()=>{const current=epoch,snapshot=view.snapshot();const d=await waitAssessment(activeOwner,snapshot.at,current);if(!d||current!==epoch)return;listing=d;renderOptions(d);if(d.candidates.length)await choose(d.candidates[0].id)},close};
}

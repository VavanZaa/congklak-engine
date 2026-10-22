'use strict';
const $ = id => document.getElementById(id);
let state = null, sending = false, polling = false, connected = false, lastLists = '';
const human = text => String(text || '').replace(/\b([AB])([0-6])\b/g, (_,p,n) => p + (Number(n)+1));
const actor = n => n === 0 ? 'A' : 'B';
function showError(message) { $('error').textContent = message; $('error').hidden = !message; }
function limits(engine=false) { return {max_depth: engine ? 12 : Number($('depth').value), time_ms:engine ? 250 : Number($('think').value),node_budget:engine ? 200000 : 1000000,memory_mb:512,tt_mb:16,use_pvs:true}; }
async function command(op, extra={}) {
  if (!state || sending) return;
  sending = true;
  showError('');
  try {
    const response = await fetch('/api/command',{method:'POST',headers:{'Content-Type':'application/json','X-Congklak-Token':state.token},body:JSON.stringify({op,revision:state.revision,...extra})});
    const body = await response.json();
    if (!body.ok) throw new Error(body.error?.message || 'Could not complete that action.');
    render(body);
    return body;
  } catch (error) { showError(error.message); return null; }
  finally { sending=false; }
}
function node(tag, text, cls) { const el=document.createElement(tag); if(text!==undefined) el.textContent=text; if(cls)el.className=cls; return el; }
function currentRecommendation() {
  const a=state?.analysis, p=state?.position, report=a?.report;
  if(a?.current!==true||a.compatible!==true||!report||!p||a.board_key!==state.board_key||state.forced_pass||state.terminal||report.score_player!==p.controller||report.completed_iteration!==true||report.terminal!==false)return null;
  const rp=report.position, sameCounts=(x,y)=>Array.isArray(x)&&x.length===y.length&&x.every((n,i)=>n===y[i]);
  if(!rp||rp.controller!==p.controller||!Array.isArray(rp.pits)||rp.pits.length!==2||!sameCounts(rp.pits[0],p.pits[0])||!sameCounts(rp.pits[1],p.pits[1])||!sameCounts(rp.stores,p.stores)||report.input_controller!==p.controller||report.normalized_controller!==p.controller||!Array.isArray(report.forced_passes)||report.forced_passes.length)return null;
  const legal=state.legal_actions;
  if(!Array.isArray(legal)||!legal.length||legal.some(pit=>!Number.isInteger(pit)||pit<0||pit>6||p.pits[p.controller][pit]<=0))return null;
  if(!sameCounts(legal,p.pits[p.controller].map((count,pit)=>count>0?pit:-1).filter(pit=>pit>=0)))return null;
  const allowed=new Set(legal);
  if(!sameCounts(report.legal_actions,legal))return null;
  if(allowed.size!==legal.length||!Array.isArray(report.candidates)||report.candidates.length!==allowed.size||!Array.isArray(report.best_moves)||!report.best_moves.length||!Number.isInteger(report.choice_depth)||report.choice_depth<1)return null;
  const choices=new Set();
  for(const choice of report.candidates){
    if(!choice||choice.actor!==p.controller||choice.analysis_complete!==true||!Number.isInteger(choice.pit)||!allowed.has(choice.pit)||p.pits[p.controller][choice.pit]<=0||choices.has(choice.pit)||!Number.isFinite(choice.depth_result?.value_actor)||choice.depth_result.score_player!==p.controller)return null;
    choices.add(choice.pit);
  }
  if(report.best_moves.some(pit=>!Number.isInteger(pit)||!allowed.has(pit)||!choices.has(pit)||p.pits[p.controller][pit]<=0))return null;
  return report;
}
function renderBoard() {
  const p=state.position, a=p.controller, engineMode=$('mode').value==='engine';
  $('store-a').textContent=p.stores[0]; $('store-b').textContent=p.stores[1];
  $('turn').textContent=state.terminal ? (p.stores[0]===p.stores[1] ? 'The round is a draw.' : `Player ${p.stores[0]>p.stores[1]?'A':'B'} wins the round.`) : `Player ${actor(a)} to move`;
  $('turn-note').textContent=state.terminal ? `Final stores: A ${p.stores[0]} · B ${p.stores[1]}` : state.active ? 'Engine is thinking. You can stop at any time.' : state.forced_pass ? 'This row is empty. Pass to continue.' : engineMode&&a===1 ? 'H_v1 is choosing a move.' : 'Choose a filled pit.';
  const report=currentRecommendation();
  for(const side of [0,1]) {
    const row=$(side===0?'row-a':'row-b'); row.replaceChildren();
    for(const pit of (side===0?[0,1,2,3,4,5,6]:[6,5,4,3,2,1,0])) {
      const count=p.pits[side][pit], legal=side===a&&state.legal_actions.includes(pit)&&!state.active&&!state.terminal&&!(engineMode&&a===1);
      const best=report?.best_moves?.includes(pit)&&side===a;
      const button=node('button',undefined,`pit${legal?' legal':''}${best?' best':''}${count===0?' empty':''}`);
      button.disabled=!legal; button.setAttribute('aria-label',`Player ${actor(side)}, pit ${pit+1}, ${count} seeds${legal?', select to play':''}`);
      button.append(node('span',count,'pit-cup'),node('span',`${actor(side)}${pit+1}`,'pit-label'));
      button.addEventListener('click',()=>command('apply',{action:pit})); row.append(button);
    }
  }
  $('pass').hidden=!state.forced_pass||state.terminal; $('pass').disabled=state.active;
  for(const id of ['new','edit']) $(id).disabled=state.active;
  $('analyze').disabled=state.active||state.terminal||state.forced_pass;
  $('stop').disabled=!state.active;
}
function renderAnalysis() {
  const a=state.analysis, report=currentRecommendation(), has=!!report;
  $('analysis-badge').textContent=state.active?'Thinking':has?`Depth ${report.choice_depth}`:'Ready';
  $('analysis-badge').className=state.active?'badge running':'badge';
  $('analysis-empty').hidden=has; $('analysis-result').hidden=!has;
  if(!has){
    $('rankings').replaceChildren();$('pv').textContent='';$('facts').textContent='';$('analysis-summary').textContent='';$('score-label').textContent=`MARGIN FOR ${actor(state.position.controller)}`;
    $('play-best').disabled=true;$('play-best').hidden=true;
    const title=state.forced_pass?'Pass first.':state.terminal?'The round is complete.':state.active?'Thinking through the choices.':a?.report?'Board changed — analyze again.':'Every pit has a possibility.';
    const detail=state.forced_pass?'This row is empty. Apply the pass before analyzing the next player’s choices.':state.terminal?'Start a new game or import another position.':state.active?'A ranking will appear when a complete legal search depth is ready.':a?.report?'The previous recommendation does not belong to the legal choices on this board.':'Analyze to compare legal moves, captures, extra turns, and a conditional continuation.';
    $('analysis-empty-title').textContent=title;$('analysis-empty-detail').textContent=detail;
    return;
  }
  $('play-best').hidden=false;
  $('analysis-summary').textContent=`Completed depth ${report.choice_depth} · ${state.active?'searching deeper':'H_v1 estimate'} · ${report.candidates.length} legal choices`;
  $('score-label').textContent=`MARGIN FOR ${actor(report.score_player)}`;
  const ranking=$('rankings'); ranking.replaceChildren();
  const rows=[...report.candidates].sort((x,y)=>y.depth_result.value_actor-x.depth_result.value_actor||x.pit-y.pit);
  rows.forEach((choice,i)=>{
    const row=node('div',undefined,'ranking'); row.append(node('span',String(i+1).padStart(2,'0'),'number'),node('strong',`${actor(report.score_player)}${choice.pit+1}`));
    if(report.best_moves.includes(choice.pit))row.append(node('span','BEST','best-tag'));
    const v=choice.depth_result.value_actor; row.append(node('span',`${v>0?'+':''}${v}`,'score')); ranking.append(row);
  });
  const best=rows[0], explanation=a.explanations?.candidates?.find(x=>x.pit===best?.pit);
  $('pv').textContent=best?.pv?.map(m=>`${actor(m.actor)}${m.pit+1}`).join(' → ')||'No conditional line at this depth.';
  $('facts').textContent=human(explanation?.immediate||a.explanations?.summary||'');
  $('play-best').disabled=!a.current||!a.compatible||state.active||state.forced_pass||!report.best_moves?.length||$('mode').value==='engine'&&state.position.controller===1;
}
function savedItem(label,buttonLabel,callback) { const row=node('div',undefined,'saved-item'),button=node('button',buttonLabel,'button secondary');button.disabled=state.active;button.addEventListener('click',callback);row.append(node('span',label),button);return row; }
function renderLists() {
  const key=JSON.stringify([state.jobs,state.sessions,state.active,state.last_job]); if(key===lastLists)return;lastLists=key;
  $('sessions').replaceChildren(); $('jobs').replaceChildren();
  state.sessions.forEach(s=>$('sessions').append(savedItem(s.name,'Load',()=>command('load_session',{session_id:s.id}))));
  state.jobs.forEach(j=>{
    const row=savedItem(`${j.id.slice(0,8)} · depth ${Math.max(0,j.depth)} · ${j.status}`,'Load',()=>command('load_job',{job_id:j.id}));
    if(j.id===state.last_job){ const resume=node('button','Resume','button secondary');resume.disabled=state.active;resume.addEventListener('click',()=>command('resume',{job_id:j.id,limits:limits()}));row.append(resume); } $('jobs').append(row);
  });
  if(!state.sessions.length)$('sessions').append(node('span','No saved boards yet.','empty'));
  if(!state.jobs.length)$('jobs').append(node('span','Your completed depths will be saved here.','empty'));
}
function render(value) {
  if(state && value.revision<state.revision)return;
  state=value;connected=true;$('message').textContent=state.message;
  renderBoard();renderAnalysis();renderLists();
}
async function poll() {
  if(sending||polling)return;
  polling=true;
  try {const response=await fetch('/api/state');const body=await response.json();if(!body.ok)throw new Error(body.error?.message||'Engine unavailable');render(body);if($('mode').value==='engine'&&!state.active&&!state.terminal&&state.position.controller===1&&!sending){if(state.forced_pass)await command('apply',{action:'pass'});else await command('engine_turn',{limits:limits(true)});}}
  catch(error){connected=false;$('message').textContent='Local engine disconnected. Restart the launcher to reconnect to your saved table.';}
  finally{polling=false;}
}
$('analyze').onclick=()=>command('analyze',{limits:limits()});$('stop').onclick=()=>{$('mode').value='explore';command('stop');};$('new').onclick=()=>command('opening');$('pass').onclick=()=>command('apply',{action:'pass'});$('play-best').onclick=()=>command('apply_best');
$('save').onclick=()=>{$('save-name').value='My board';$('save-error').hidden=true;$('save-dialog').showModal();$('save-name').focus();$('save-name').select();};
$('close-save').onclick=()=>$('save-dialog').close();$('cancel-save').onclick=()=>$('save-dialog').close();
$('save-form').onsubmit=async event=>{event.preventDefault();const name=$('save-name').value.trim();if(!name){$('save-error').textContent='Give this board a name.';$('save-error').hidden=false;return;}const result=await command('save',{name});if(result)$('save-dialog').close();else{$('save-error').textContent=$('error').textContent||'Could not save this board. Try again.';$('save-error').hidden=false;}};
$('mode').onchange=()=>{if(state)render(state);};
function fillEditor(p) { $('editor-pits').replaceChildren();for(const side of [0,1]){const row=node('div');for(let pit=0;pit<7;pit++){const label=node('label',`${actor(side)}${pit+1}`),input=node('input');input.type='number';input.min=0;input.max=98;input.required=true;input.id=`edit-${side}-${pit}`;input.value=p.pits[side][pit];label.append(input);row.append(label);} $('editor-pits').append(row);}$('edit-store-a').value=p.stores[0];$('edit-store-b').value=p.stores[1];$('edit-actor').value=p.controller; }
function validatePosition(p) {
  if(!p||typeof p!=='object'||!Array.isArray(p.pits)||p.pits.length!==2||p.pits.some(r=>!Array.isArray(r)||r.length!==7)||!Array.isArray(p.stores)||p.stores.length!==2||![0,1].includes(p.controller))throw new Error('Use two rows of seven pits, two stores, and controller 0 or 1.');
  const counts=[...p.pits.flat(),...p.stores];
  if(counts.some(x=>!Number.isInteger(x)||x<0||x>98))throw new Error('Every pit and store must contain a whole number from 0 to 98.');
  if(counts.reduce((x,y)=>x+y,0)!==98)throw new Error('The pits and stores must total exactly 98 seeds.');
  return p;
}
$('edit').onclick=()=>{fillEditor(state.position);$('import-json').value='';$('editor-error').hidden=true;$('editor').showModal();};$('close-editor').onclick=()=>$('editor').close();
$('read-json').onclick=()=>{try{const value=JSON.parse($('import-json').value);if(value.profile&&value.profile!=='id-betawi-relay7-v0'||value.identity?.profile&&value.identity.profile!=='id-betawi-relay7-v0')throw new Error('This file uses a different rules profile.');const p=validatePosition(value.position||value);fillEditor(p);$('editor-error').hidden=true;}catch(e){$('editor-error').textContent=e.message;$('editor-error').hidden=false;}};
$('editor-form').onsubmit=async event=>{event.preventDefault();const position={pits:[0,1].map(side=>Array.from({length:7},(_,pit)=>Number($(`edit-${side}-${pit}`).value))),stores:[Number($('edit-store-a').value),Number($('edit-store-b').value)],controller:Number($('edit-actor').value)};try{validatePosition(position);}catch(e){$('editor-error').textContent=e.message;$('editor-error').hidden=false;return;}const result=await command('import',{position});if(result)$('editor').close();else{$('editor-error').textContent=$('error').textContent;$('editor-error').hidden=false;}};
poll();setInterval(poll,500);

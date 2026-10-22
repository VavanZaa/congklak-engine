'use strict';
const $ = id => document.getElementById(id);
let state = null, sending = false, polling = false, connected = false, lastLists = '';
let liveRequestedKey = null, lastHistory = '';
let boardLayout = 'left', renderedLayout = null;
try { if (localStorage.getItem('congklak.layout') === 'right') boardLayout = 'right'; } catch (_) {}
$('board-layout').value = boardLayout;
try { $('live-analysis').checked = localStorage.getItem('congklak.live') !== 'off'; } catch (_) {}
function rememberLive() { try { localStorage.setItem('congklak.live', $('live-analysis').checked?'on':'off'); } catch (_) {} }
// Only presentation changes. Positions, DOM identifiers, and commands keep canonical pit indices.
function displayPit(pit) { return boardLayout === 'left' ? 7 - pit : pit + 1; }
function pitOrder(side) {
  const bottom = boardLayout === 'left' ? [6,5,4,3,2,1,0] : [0,1,2,3,4,5,6];
  return side === 0 ? bottom : [...bottom].reverse();
}
const human = text => String(text || '').replace(/\b([AB])([0-6])\b/g, (_,p,n) => p + displayPit(Number(n)));
const actor = n => n === 0 ? 'A' : 'B';
const pitName = (side,pit) => `${actor(side)}${displayPit(pit)}`;
function displayMessage(text) {
  return String(text || '').replace(/^Player ([AB]) played pit ([1-7])\./,
    (_,player,pit) => `Player ${player} played pit ${displayPit(Number(pit)-1)}.`);
}
function renderLayout() {
  if (renderedLayout === boardLayout) return;
  const left = boardLayout === 'left';
  $('board').replaceChildren($(left?'store-a-cup':'store-b-cup'),$('pit-rows'),$(left?'store-b-cup':'store-a-cup'));
  $('orientation-a').textContent = left ? 'store ← A1 ← … ← A7' : 'A1 → … → A7 → store';
  $('orientation-b').textContent = left ? 'B7 → … → B1 → store' : 'store ← B7 ← … ← B1';
  $('layout-note').textContent = left ? 'Store on your left · sow right to left.' : 'Store on your right · sow left to right.';
  renderedLayout = boardLayout;
}
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
  finally { sending=false;if(['apply','apply_best','opening','import','load_session','load_job'].includes(op))followBoard(); }
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
  const focusedPit=document.activeElement?.id;
  renderLayout();
  const p=state.position, a=p.controller, engineMode=$('mode').value==='engine';
  $('store-a').textContent=p.stores[0]; $('store-b').textContent=p.stores[1];
  $('turn').textContent=state.terminal ? (p.stores[0]===p.stores[1] ? 'The round is a draw.' : `Player ${p.stores[0]>p.stores[1]?'A':'B'} wins the round.`) : `Player ${actor(a)} to move`;
  $('turn-note').textContent=state.terminal ? `Final stores: A ${p.stores[0]} · B ${p.stores[1]}` : state.forced_pass ? 'This row is empty. Pass to continue.' : engineMode&&a===1 ? 'H_v1 is choosing a move.' : state.history?.turn_message || 'Choose a filled pit. Landing in your own store gives another move.';
  const report=currentRecommendation();
  for(const side of [0,1]) {
    const row=$(side===0?'row-a':'row-b'); row.replaceChildren();
    for(const pit of pitOrder(side)) {
      const count=p.pits[side][pit], legal=side===a&&state.legal_actions.includes(pit)&&!state.terminal&&!(engineMode&&a===1);
      const best=report?.best_moves?.includes(pit)&&side===a;
      const button=node('button',undefined,`pit${legal?' legal':''}${best?' best':''}${count===0?' empty':''}`);
      button.id=`pit-${side}-${pit}`;
      button.disabled=!legal; button.setAttribute('aria-label',`Player ${actor(side)}, pit ${displayPit(pit)}, ${count} seeds${legal?', select to play':''}`);
      button.append(node('span',count,'pit-cup'),node('span',pitName(side,pit),'pit-label'));
      button.addEventListener('click',()=>command('apply',{action:pit})); row.append(button);
    }
  }
  $('pass').hidden=!state.forced_pass||state.terminal; $('pass').disabled=engineMode&&a===1;
  for(const id of ['new','edit']) $(id).disabled=false;
  $('analyze').disabled=state.active||state.terminal||state.forced_pass;
  $('stop').disabled=!state.active;
  $('think').disabled=$('live-analysis').checked;$('depth').disabled=$('live-analysis').checked;
  if(focusedPit?.startsWith('pit-')){const button=$(focusedPit);if(button&&!button.disabled)button.focus({preventScroll:true});}
}
function renderAnalysis() {
  renderEvaluation();
  const a=state.analysis, report=currentRecommendation(), has=!!report;
  $('analysis-badge').textContent=state.active?'Thinking':has?`Depth ${report.choice_depth}`:'Ready';
  $('analysis-badge').className=state.active?'badge running':'badge';
  const completed=a?.current&&a.compatible?a.report:null, work=a?.current?state.search?.work:null;
  const elapsed=state.search?.elapsed_ms??work?.elapsed_ms??completed?.elapsed_ms;
  const nodes=work?.nodes??completed?.nodes??completed?.coverage?.nodes;
  const status=state.active?(state.search?.mode==='live'?'Live · searching deeper':'Searching'):completed&&$('live-analysis').checked&&['max_depth','already_complete'].includes(a.stop_reason)?'Live · depth cap reached':completed&&$('live-analysis').checked&&a.stop_reason==='time_limit'?'Live · time limit reached':'Stopped';
  const stats=[status,completed?`Depth ${completed.choice_depth}`:'Waiting for a completed depth'];
  if(Number.isFinite(nodes))stats.push(`${nodes.toLocaleString()} nodes at last update`);
  if(completed?.elapsed_ms>0&&Number.isFinite(completed.nodes))stats.push(`${Math.round(completed.nodes*1000/completed.elapsed_ms).toLocaleString()} nodes/s at last depth`);
  if(Number.isFinite(elapsed))stats.push(`${(elapsed/1000).toFixed(1)} s`);
  $('search-stats').textContent=stats.join(' · ');
  $('analysis-empty').hidden=has; $('analysis-result').hidden=!has;
  if(!has){
    $('rankings').replaceChildren();$('pv').textContent='';$('facts').textContent='';$('analysis-summary').textContent='';$('score-label').textContent=`MARGIN FOR ${actor(state.position.controller)}`;
    $('play-best').disabled=true;$('play-best').hidden=true;
    const title=state.forced_pass?'Pass first.':state.terminal?'The round is complete.':state.active?'Thinking through the choices.':$('live-analysis').checked?'Preparing live analysis.':a?.report?'Board changed — analyze again.':'Every pit has a possibility.';
    const detail=state.forced_pass?'This row is empty. Apply the pass before analyzing the next player’s choices.':state.terminal?'Start a new game or import another position.':state.active?'A ranking will appear when a complete legal search depth is ready.':$('live-analysis').checked?'The engine will follow this position automatically.':a?.report?'The previous recommendation does not belong to the legal choices on this board.':'Analyze to compare legal moves, captures, extra turns, and a conditional continuation.';
    $('analysis-empty-title').textContent=title;$('analysis-empty-detail').textContent=detail;
    return;
  }
  $('play-best').hidden=false;
  $('analysis-summary').textContent=`Completed depth ${report.choice_depth} · ${state.active?'searching deeper':'H_v1 estimate'} · ${report.candidates.length} legal choices`;
  $('score-label').textContent=`MARGIN FOR ${actor(report.score_player)}`;
  const ranking=$('rankings'); ranking.replaceChildren();
  const rows=[...report.candidates].sort((x,y)=>y.depth_result.value_actor-x.depth_result.value_actor||x.pit-y.pit);
  rows.forEach((choice,i)=>{
    const row=node('div',undefined,'ranking'); row.append(node('span',String(i+1).padStart(2,'0'),'number'),node('strong',pitName(report.score_player,choice.pit)));
    if(report.best_moves.includes(choice.pit))row.append(node('span','BEST','best-tag'));
    const v=choice.depth_result.value_actor; row.append(node('span',`${v>0?'+':''}${v}`,'score')); ranking.append(row);
  });
  const best=rows[0], explanation=a.explanations?.candidates?.find(x=>x.pit===best?.pit);
  $('pv').textContent=best?.pv?.map(m=>pitName(m.actor,m.pit)).join(' → ')||'No conditional line at this depth.';
  $('facts').textContent=human(explanation?.immediate||a.explanations?.summary||'');
  $('play-best').disabled=!a.current||!a.compatible||state.forced_pass||!report.best_moves?.length||$('mode').value==='engine'&&state.position.controller===1;
}
function positionEvaluation() {
  const p=state?.position;
  try { validatePosition(p); } catch (_) { return {kind:'unknown',title:'Evaluation unavailable',detail:'Waiting for a valid board.'}; }
  const remaining=p.pits.flat().reduce((a,b)=>a+b,0), margin=p.stores[0]-p.stores[1];
  // Stores never decrease: 50 of the fixed 98 seeds proves the eventual winner.
  if(remaining===0||p.stores[0]>49||p.stores[1]>49){
    const winner=p.stores[0]===p.stores[1]?null:p.stores[0]>p.stores[1]?0:1;
    return {kind:'certain',winner,value:winner===null?0:winner===0?98:-98,
      title:winner===null?'The round is a draw.':`Player ${actor(winner)} ${remaining===0?'wins the round.':'has secured the round.'}`,
      detail:remaining===0?`Final stores: A ${p.stores[0]} · B ${p.stores[1]}.`:`${actor(winner)} already holds ${p.stores[winner]} of 98 seeds. These stored seeds cannot be taken back.`,
      chances:winner===null?'A win 0% · Draw 100% · B win 0%':`A win ${winner===0?100:0}% · Draw 0% · B win ${winner===1?100:0}%`};
  }
  const report=currentRecommendation();
  if(!report)return {kind:'unknown',title:state.forced_pass?'Pass before evaluating.':state.active?'Evaluating this position…':'No current evaluation.',
    detail:state.forced_pass?'The empty row must pass first.':state.active?'Waiting for a completed search depth on this board.':'Turn on live analysis or choose Analyze once.'};
  const best=Math.max(...report.candidates.map(c=>c.depth_result.value_actor));
  const value=report.score_player===0?best:-best;
  if(!Number.isFinite(value)||Math.abs(value)>98)return {kind:'unknown',title:'Evaluation unavailable',detail:'The score is outside the legal seed-margin range.'};
  return {kind:'estimate',value,depth:report.choice_depth,
    title:value===0?'The engine sees no seed advantage.':`The engine favors player ${value>0?'A':'B'}.`,
    detail:`Estimated margin: ${value===0?'0 seeds':`${value>0?'A':'B'} +${Math.abs(value)} seeds`} at depth ${report.choice_depth}.`,
    chances:'Win percentages are not calibrated yet. This bar shows seed advantage, not win probability.'};
}
function renderEvaluation() {
  const e=positionEvaluation(), known=e.kind!=='unknown';
  $('evaluation-title').textContent=e.title;
  $('evaluation-detail').textContent=e.detail;
  $('evaluation-chances').textContent=e.chances||'Win chances are unknown for this position.';
  $('evaluation-badge').textContent=e.kind==='certain'?'Confirmed':e.kind==='estimate'?`Depth ${e.depth} · estimate`:'Waiting';
  $('evaluation-meter').hidden=!known;$('evaluation-unknown').hidden=known;
  $('evaluation-meter').value=known?e.value:0;
  $('evaluation-meter').setAttribute('aria-valuetext',`${e.title} ${e.detail} ${e.chances||''}`);
  $('evaluation-meter').className=e.kind==='certain'&&e.winner===null?'draw':'';
}
function renderHistory() {
  const history=state.history, key=JSON.stringify(history); if(key===lastHistory)return;lastHistory=key;
  const entries=history?.entries||[], list=$('move-list');list.replaceChildren();
  $('move-count').textContent=entries.length;
  $('history-note').textContent=history?.note||'Move recording starts at this position.';
  $('history-empty').hidden=entries.length>0;
  for(const entry of entries){
    const row=node('li',undefined,'played-move');
    const title=node('div',undefined,'played-title');
    title.append(node('span',String(entry.number).padStart(2,'0'),'number'),node('strong',entry.action==='pass'?`${actor(entry.actor)} passes`:pitName(entry.actor,entry.action)),node('span',`A +${entry.store_delta[0]} · B +${entry.store_delta[1]}`,'move-gain'));
    row.append(title,node('p',entry.explanation));list.append(row);
  }
  list.scrollTop=list.scrollHeight;
}
function selectRecordTab(name, focus=false) {
  for(const tab of ['moves','rules']){const selected=tab===name,button=$(tab+'-tab');button.setAttribute('aria-selected',String(selected));button.tabIndex=selected?0:-1;$(tab+'-panel').hidden=!selected;if(selected&&focus)button.focus();}
}
for(const name of ['moves','rules']){
  $(name+'-tab').onclick=()=>selectRecordTab(name);
  $(name+'-tab').addEventListener('keydown',event=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(event.key)){event.preventDefault();selectRecordTab(event.key==='Home'?'moves':event.key==='End'?'rules':name==='moves'?'rules':'moves',true);}});
}
function savedItem(label,buttonLabel,callback) { const row=node('div',undefined,'saved-item'),button=node('button',buttonLabel,'button secondary');button.disabled=state.active;button.addEventListener('click',callback);row.append(node('span',label),button);return row; }
function renderLists() {
  const key=JSON.stringify([state.jobs,state.sessions,state.active,state.last_job]); if(key===lastLists)return;lastLists=key;
  $('sessions').replaceChildren(); $('jobs').replaceChildren();
  state.sessions.forEach(s=>$('sessions').append(savedItem(s.name,'Load',()=>command('load_session',{session_id:s.id}))));
  state.jobs.forEach(j=>{
    const row=savedItem(`${j.id.slice(0,8)} · depth ${Math.max(0,j.depth)} · ${j.status}`,'Load',()=>command('load_job',{job_id:j.id}));
    if(j.id===state.last_job){ const resume=node('button','Resume','button secondary');resume.disabled=state.active;resume.addEventListener('click',()=>{const settings=limits();settings.max_depth=Math.min(128,Math.max(settings.max_depth,j.depth+1));command('resume',{job_id:j.id,limits:settings});});row.append(resume); } $('jobs').append(row);
  });
  if(!state.sessions.length)$('sessions').append(node('span','No saved boards yet.','empty'));
  if(!state.jobs.length)$('jobs').append(node('span','Your completed depths will be saved here.','empty'));
}
function render(value) {
  if(state && state.token===value.token && value.revision<state.revision)return;
  if(state&&state.token!==value.token)liveRequestedKey=null;
  state=value;connected=true;$('message').textContent=displayMessage(state.message);
  renderBoard();renderAnalysis();renderLists();renderHistory();
}
async function followBoard() {
  if(!state||sending||state.active||state.terminal)return;
  if($('mode').value==='engine'&&state.position.controller===1){
    if(state.forced_pass)await command('apply',{action:'pass'});
    else await command('engine_turn',{limits:limits(true)});
  } else if($('live-analysis').checked&&!state.forced_pass&&liveRequestedKey!==`${state.revision}:${state.board_key}`){
    const key=`${state.revision}:${state.board_key}`;liveRequestedKey=key;
    const result=await command('live');
    if(!result){$('live-analysis').checked=false;rememberLive();}
  }
}
async function poll() {
  if(sending||polling)return;
  polling=true;
  try {const response=await fetch('/api/state');const body=await response.json();if(!body.ok)throw new Error(body.error?.message||'Engine unavailable');render(body);await followBoard();}
  catch(error){connected=false;$('message').textContent='Local engine disconnected. Restart the launcher to reconnect to your saved table.';}
  finally{polling=false;}
}
$('analyze').onclick=()=>{$('live-analysis').checked=false;rememberLive();command('analyze',{limits:limits()});};$('stop').onclick=()=>{$('live-analysis').checked=false;rememberLive();$('mode').value='explore';command('stop');};$('new').onclick=()=>command('opening');$('pass').onclick=()=>command('apply',{action:'pass'});$('play-best').onclick=()=>command('apply_best');
$('live-analysis').onchange=()=>{rememberLive();liveRequestedKey=null;if(!$('live-analysis').checked&&state?.active&&state.search?.mode==='live')command('stop');else if($('live-analysis').checked)followBoard();};
$('save').onclick=()=>{$('save-name').value='My board';$('save-error').hidden=true;$('save-dialog').showModal();$('save-name').focus();$('save-name').select();};
$('close-save').onclick=()=>$('save-dialog').close();$('cancel-save').onclick=()=>$('save-dialog').close();
$('save-form').onsubmit=async event=>{event.preventDefault();const name=$('save-name').value.trim();if(!name){$('save-error').textContent='Give this board a name.';$('save-error').hidden=false;return;}const result=await command('save',{name});if(result)$('save-dialog').close();else{$('save-error').textContent=$('error').textContent||'Could not save this board. Try again.';$('save-error').hidden=false;}};
$('mode').onchange=()=>{liveRequestedKey=null;if(state){render(state);if(state.active)command('stop');else followBoard();}};
$('board-layout').onchange=()=>{
  boardLayout=$('board-layout').value==='right'?'right':'left';
  try { localStorage.setItem('congklak.layout',boardLayout); } catch (_) {}
  lastHistory='';
  if(state)render(state);
};
function fillEditor(p) { $('editor-pits').replaceChildren();for(const side of [1,0]){const row=node('div');for(const pit of pitOrder(side)){const label=node('label',pitName(side,pit)),input=node('input');input.type='number';input.min=0;input.max=98;input.required=true;input.id=`edit-${side}-${pit}`;input.value=p.pits[side][pit];label.append(input);row.append(label);} $('editor-pits').append(row);}$('edit-store-a').value=p.stores[0];$('edit-store-b').value=p.stores[1];$('edit-actor').value=p.controller; }
function validatePosition(p) {
  if(!p||typeof p!=='object'||!Array.isArray(p.pits)||p.pits.length!==2||p.pits.some(r=>!Array.isArray(r)||r.length!==7)||!Array.isArray(p.stores)||p.stores.length!==2||![0,1].includes(p.controller))throw new Error('Use two rows of seven pits, two stores, and controller 0 or 1.');
  const counts=[...p.pits.flat(),...p.stores];
  if(counts.some(x=>!Number.isInteger(x)||x<0||x>98))throw new Error('Every pit and store must contain a whole number from 0 to 98.');
  if(counts.reduce((x,y)=>x+y,0)!==98)throw new Error('The pits and stores must total exactly 98 seeds.');
  return p;
}
function checkImportedRules(value) {
  const expected=state?.rule_contract, profile=expected?.profile||state?.profile||'id-betawi-relay7-v0';
  for(const identity of [value,value.identity]){
    if(!identity||typeof identity!=='object')continue;
    if(identity.profile!==undefined&&identity.profile!==profile)throw new Error('This file uses a different rules profile.');
    const digest=expected?.rules_digest||state?.identity?.rules_digest;
    if(identity.rules_digest!==undefined&&identity.rules_digest!==digest)throw new Error('This file uses a different rules digest.');
  }
  if(value.rule_contract!==undefined){
    const c=value.rule_contract;
    if(!expected||!c||typeof c!=='object'||Array.isArray(c)||Object.keys(c).length!==Object.keys(expected).length||Object.keys(expected).some(k=>c[k]!==expected[k]))throw new Error('This table uses sequential opening and a single round. The file uses different game rules.');
  }
}
$('edit').onclick=()=>{fillEditor(state.position);$('import-json').value='';$('editor-error').hidden=true;$('editor').showModal();};$('close-editor').onclick=()=>$('editor').close();
$('read-json').onclick=()=>{try{const value=JSON.parse($('import-json').value);if(!value||typeof value!=='object')throw new Error('Import a board object.');checkImportedRules(value);const p=validatePosition(value.position||value);fillEditor(p);$('editor-error').hidden=true;}catch(e){$('editor-error').textContent=e.message;$('editor-error').hidden=false;}};
$('editor-form').onsubmit=async event=>{event.preventDefault();const position={pits:[0,1].map(side=>Array.from({length:7},(_,pit)=>Number($(`edit-${side}-${pit}`).value))),stores:[Number($('edit-store-a').value),Number($('edit-store-b').value)],controller:Number($('edit-actor').value)};try{validatePosition(position);}catch(e){$('editor-error').textContent=e.message;$('editor-error').hidden=false;return;}const result=await command('import',{position,...(state.rule_contract?{rule_contract:state.rule_contract}:{})});if(result)$('editor').close();else{$('editor-error').textContent=$('error').textContent;$('editor-error').hidden=false;}};
poll();setInterval(poll,500);

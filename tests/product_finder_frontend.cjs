'use strict';
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert');
const root=path.resolve(__dirname,'..');
const source=fs.readFileSync(path.join(root,'product_v2/finder.js'),'utf8');
const certificate=(name)=>JSON.parse(fs.readFileSync(path.join(root,`data/dakon-opening-v1/${name}/certificate.json`),'utf8'));
const a=certificate('n7-majority-A'),b=certificate('n7-majority-B'),wide=certificate('n64-majority-A');
const idA='a'.repeat(32),idB='b'.repeat(32),idWide='c'.repeat(32);
function record(id,body){return {run_id:id,n:body.recipe.n,starter:body.recipe.starter,goal:body.recipe.goal,state:'found',outcome:'guaranteed_win',error:null,resumable:true,active:false,pausing:false,best_store:body.proof.starter_store,required:body.proof.required,simulated_actions:126,expanded_positions:50,generation:4,depth:12,cache_hits:0};}
const flush=async()=>{for(let i=0;i<8;i++)await new Promise(resolve=>setImmediate(resolve));};
function harness(runs,storage={}){
 const elements={},calls=[],intervals=[];
 class Element{
  constructor(tag='div'){this.tag=tag;this.children=[];this.value='';this.hidden=false;this.disabled=false;this.attributes={};this.events={};this._text='';}
  set textContent(value){this._text=String(value);this.children=[];}get textContent(){return this._text+this.children.map(n=>n.textContent).join('');}
  append(...nodes){this.children.push(...nodes);}replaceChildren(...nodes){this.children=nodes;this._text='';}
  setAttribute(key,value){this.attributes[key]=String(value);}addEventListener(key,fn){this.events[key]=fn;}
 }
 const el=id=>elements[id]||(elements[id]=new Element());
 el('finder-n').value='7';el('finder-goal').value='majority';el('finder-starter').value='0';
 const fixture={ok:true,token:'fixture-token',runs:structuredClone(runs),active_run:null,run_limit:100};
 const context={document:{getElementById:el,createElement:tag=>new Element(tag)},console,
  localStorage:{getItem:key=>storage[key]??null,setItem:(key,value)=>{storage[key]=String(value);}},
  setInterval:fn=>intervals.push(fn),fetch:async(url,options)=>{
   calls.push({url,options});if(h.intercept)return h.intercept(url,options);
   if(url.endsWith('/certificate')){
    const id=url.split('/')[3],cert=id===idA?a:id===idB?b:wide;
    return {json:async()=>({ok:true,run_id:id,verification:{passed:true},certificate:structuredClone(cert)})};
   }
   if(options){const req=JSON.parse(options.body);h.requests.push(req);if(h.onCommand)h.onCommand(req);}
   return {json:async()=>structuredClone(fixture)};
  }};
 const h={el,calls,intervals,storage,fixture,requests:[],context};
 vm.createContext(context);vm.runInContext(source,context);return h;
}
const cellsFromBoard=h=>{
 const board=h.el('finder-board'),rows=board.children[1].children;
 const top=rows[0].children.map(p=>Number(p.children[0].textContent));
 const bottom=rows[1].children.map(p=>Number(p.children[0].textContent)).reverse();
 return [...bottom,Number(board.children[0].children[1].textContent),...top,Number(board.children[2].children[1].textContent)];
};
const click=(h,id)=>h.el(id).events.click();
const checks=[];
(async()=>{
 let h=harness([record(idA,a.body)]);await flush();click(h,'finder-view-tab');await flush();
 assert(h.el('play-view').hidden);assert(!h.el('finder-view').hidden);
 assert.equal(h.el('finder-view-tab').attributes['aria-pressed'],'true');
 assert.equal(h.el('finder-badge').textContent,'Verified win');assert(!h.el('finder-verified').hidden);
 assert.equal(h.el('finder-resume').disabled,true);
 assert.deepEqual(cellsFromBoard(h),[...Array(7).fill(7),0,...Array(7).fill(7),0]);
 assert.equal(h.el('finder-board').children[1].children[1].children.at(-1).className,'finder-pit chosen');
 checks.push('finder_tab_and_certified_initial_board_are_separate_from_play');
 for(let i=0;i<a.body.actions.length;i++){
  click(h,'finder-next');assert.deepEqual(cellsFromBoard(h),a.body.steps[i].cells);
  assert(h.el('finder-step-detail').textContent.includes(`A${7-a.body.actions[i]}`));
 }
 assert(h.el('finder-next').disabled);assert(h.el('finder-step-detail').textContent.includes('Capture: the turn ends.'));
 assert(h.el('finder-proof').textContent.includes('at most 44'));
 click(h,'finder-prev');assert.deepEqual(cellsFromBoard(h),a.body.steps.at(-2).cells);
 h.el('finder-slider').value='4';h.el('finder-slider').events.input();assert.deepEqual(cellsFromBoard(h),a.body.steps[3].cells);
 h.el('finder-sequence').children[8].events.click();assert.deepEqual(cellsFromBoard(h),a.body.steps[8].cells);
 checks.push('every_choice_replays_exact_certificate_cells_capture_and_left_store_labels');
 const storage=h.storage;h=harness([record(idA,a.body)],storage);await flush();
 assert(!h.el('finder-view').hidden);assert.deepEqual(cellsFromBoard(h),a.body.steps[8].cells);
 click(h,'finder-first');assert.equal(h.el('finder-slider').value,0);
 assert(h.el('finder-prev').disabled);checks.push('view_run_and_replay_step_survive_page_reload');
 h=harness([record(idB,b.body)]);await flush();click(h,'finder-next');
 assert.deepEqual(cellsFromBoard(h),b.body.steps[0].cells);
 assert(h.el('finder-step-detail').textContent.includes('B7'));
 const rows=h.el('finder-board').children[1].children;
 assert.deepEqual(rows[0].children.map(p=>p.children[1].textContent),['B7','B6','B5','B4','B3','B2','B1']);
 assert.deepEqual(rows[1].children.map(p=>p.children[1].textContent),['A1','A2','A3','A4','A5','A6','A7']);
 checks.push('player_B_uses_same_canonical_mapping_and_owns_right_screen_store');
 h=harness([record(idWide,wide.body)]);await flush();
 assert.equal(h.el('finder-board').children[1].children[0].children.length,64);
 h.el('finder-slider').value=wide.body.actions.length;h.el('finder-slider').events.input();
 assert.deepEqual(cellsFromBoard(h),wide.body.steps.at(-1).cells);
 assert.equal(h.el('finder-board').children[0].children[1].textContent,'4103');
 checks.push('n64_renders_all_pits_and_replays_final_verified_counts');
 const paused={...record(idA,a.body),state:'searching',outcome:'unknown',stop_reason:'cancelled',simulated_actions:5};
 h=harness([paused]);await flush();assert(h.el('finder-replay-content').hidden);assert(h.el('finder-verified').hidden);
 assert(!h.el('finder-resume').disabled);assert(h.el('finder-badge').textContent==='Paused');
 h.onCommand=req=>{if(req.op==='resume'){h.fixture.active_run=idA;h.fixture.runs[0].active=true;}
  if(req.op==='pause')h.fixture.runs[0].pausing=true;};
 click(h,'finder-resume');await flush();assert.equal(h.requests[0].run_id,idA);assert.equal(h.requests[0].op,'resume');
 assert(!h.el('finder-pause').disabled);click(h,'finder-pause');await flush();
 assert(h.el('finder-pause').disabled);assert.equal(h.el('finder-badge').textContent,'Pausing…');
 checks.push('unfinished_runs_hide_proof_and_pause_resume_only_use_finder_commands');
 h=harness([]);await flush();h.el('finder-n').value='8';h.el('finder-n').events.input();
 assert(h.el('finder-size-note').textContent.includes('128 seeds'));
 h.onCommand=req=>{h.fixture.active_run=idA;h.fixture.runs=[{...paused,n:req.n,active:true}];};
 h.el('finder-form').events.submit({preventDefault(){}});await flush();
 assert.deepEqual(h.requests[0],{op:'start',n:8,goal:'majority',starter:0});
 assert.equal(h.calls.find(c=>c.options).options.headers['X-Congklak-Token'],'fixture-token');
 assert(h.el('finder-start').disabled);assert.equal(h.storage['congklak.finder.run'],idA);
 assert(h.calls.every(c=>c.url.startsWith('/api/finder')));checks.push('validated_start_uses_token_and_never_writes_game_api');
 h=harness([]);await flush();for(const bad of ['0','65','2.5','banana']){
  h.el('finder-n').value=bad;h.el('finder-form').events.submit({preventDefault(){}});await flush();
  assert(!h.el('finder-error').hidden);
 }assert.equal(h.requests.length,0);checks.push('invalid_n_is_refused_before_request');
 const pending={...paused,state:'pending_verification',active:true};
 h=harness([pending]);h.fixture.active_run=idA;await flush();
 assert.equal(h.el('finder-badge').textContent,'Verifying…');assert(h.el('finder-replay-content').hidden);
 checks.push('pending_verification_never_shows_guaranteed_sequence');
 h=harness([paused]);await flush();click(h,'finder-view-tab');await flush();
 let releasePoll;const old=structuredClone(h.fixture);
 h.intercept=async(url,options)=>options?{json:async()=>({...h.fixture,active_run:idA,runs:[{...paused,active:true}]})}:new Promise(resolve=>{releasePoll=()=>resolve({json:async()=>old});});
 h.intervals[0]();click(h,'finder-resume');await flush();
 assert.equal(h.el('finder-badge').textContent,'Searching…');releasePoll();await flush();
 assert.equal(h.el('finder-badge').textContent,'Searching…');checks.push('poll_started_before_command_cannot_overwrite_newer_command_state');
 const html=fs.readFileSync(path.join(root,'product_v2/index.html'),'utf8');
 assert(html.includes('id="play-view"'));assert(html.includes('src="/finder.js"'));
 assert(fs.readFileSync(path.join(root,'product_v2/style.css'),'utf8').includes('.finder-board-scroll{overflow-x:auto'));
 checks.push('shipped_assets_include_accessible_tab_and_horizontal_large_board_scroll');
 const receipt={passed:true,checks};
 fs.mkdirSync(path.join(root,'build/product-finder-v1'),{recursive:true});
 fs.writeFileSync(path.join(root,'build/product-finder-v1/frontend.json'),JSON.stringify(receipt,null,2)+'\n');
 console.log(JSON.stringify(receipt,null,2));
})().catch(error=>{console.error(error);process.exitCode=1;});

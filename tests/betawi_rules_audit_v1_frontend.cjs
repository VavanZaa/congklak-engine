// Execute the real render handlers; stale/illegal root choices must never remain visible.
const fs=require('fs'),vm=require('vm'),assert=require('assert'),crypto=require('crypto'),path=require('path');
const root=path.resolve(__dirname,'..'),elements={};
class Element{
  constructor(){this.children=[];this.value='';this.textContent='';this.hidden=false;this.disabled=false;this.className='';this.events={};this.attributes={};}
  append(...items){this.children.push(...items);} replaceChildren(...items){this.children=[...items];}
  addEventListener(kind,fn){this.events[kind]=fn;}setAttribute(name,value){this.attributes[name]=value;}showModal(){}close(){}focus(){}select(){}
}
const element=id=>elements[id]||(elements[id]=new Element());
const context={document:{getElementById:element,createElement:()=>new Element()},window:{},fetch:()=>new Promise(()=>{}),setInterval(){},console};
vm.createContext(context);vm.runInContext(fs.readFileSync(path.join(root,'product_v2/app.js'),'utf8'),context);
element('mode').value='explore';
function valid(){
  const position={pits:[Array(7).fill(7),Array(7).fill(7)],stores:[0,0],controller:0};
  const actions=[0,1,2,3,4,5,6];
  const report={position:structuredClone(position),completed_iteration:true,terminal:false,input_controller:0,normalized_controller:0,forced_passes:[],legal_actions:actions.slice(),score_player:0,choice_depth:3,best_moves:[0],candidates:actions.map(pit=>({actor:0,pit,analysis_complete:true,depth_result:{value_actor:7-pit,score_player:0},pv:[{actor:0,pit}]}))};
  return{position,board_key:'board-one',legal_actions:actions,active:false,forced_pass:false,terminal:false,analysis:{current:true,compatible:true,board_key:'board-one',report,explanations:{candidates:[{pit:0,immediate:'Adds one seed.'}]}}};
}
function render(value){context.fixture=value;vm.runInContext('state=fixture;renderAnalysis();renderBoard();',context);}
function cleared(label){
  assert.equal(element('analysis-result').hidden,true,label);
  assert.equal(element('rankings').children.length,0,label);
  for(const id of ['pv','facts','analysis-summary'])assert.equal(element(id).textContent,'',label);
  assert.equal(element('play-best').disabled,true,label);assert.equal(element('play-best').hidden,true,label);
  assert(![...element('row-a').children,...element('row-b').children].some(el=>el.className.split(' ').includes('best')),label);
}
let value=valid();render(value);
assert.equal(element('rankings').children.length,7);assert.equal(element('analysis-result').hidden,false);
assert.equal(element('play-best').disabled,false);assert.equal(element('play-best').hidden,false);
assert(element('row-a').children.find(button=>button.id==='pit-0-0').className.split(' ').includes('best'));
const checks=['valid_current_complete_ranking_and_highlight'];
const bad=[
 ['stale_after_move',s=>{s.analysis.current=false;s.board_key='after-move';s.position.pits[0][0]=0;s.legal_actions=[1,2,3,4,5,6];}],
 ['imported_different_position',s=>{s.position.stores=[1,0];s.position.pits[0][0]=6;}],
 ['changed_actor_after_load',s=>{s.position.controller=1;}],
 ['empty_pit_in_candidate_and_best',s=>{s.position.pits[0][0]=0;s.analysis.report.position=structuredClone(s.position);s.legal_actions=[1,2,3,4,5,6];}],
 ['invalid_best_empty_pit',s=>{s.position.pits[0][0]=0;s.analysis.report.position=structuredClone(s.position);s.legal_actions=[1,2,3,4,5,6];s.analysis.report.legal_actions=s.legal_actions.slice();s.analysis.report.candidates=s.analysis.report.candidates.slice(1);}],
 ['candidate_wrong_actor',s=>{s.analysis.report.candidates[0].actor=1;}],
 ['missing_legal_candidate',s=>{s.analysis.report.candidates.pop();}],
 ['supplied_legal_set_omits_nonempty_pit',s=>{s.legal_actions.pop();s.analysis.report.legal_actions.pop();s.analysis.report.candidates.pop();}],
 ['incompatible_saved_build',s=>{s.analysis.compatible=false;}],
 ['forced_pass',s=>{s.forced_pass=true;}],
 ['terminal_board',s=>{s.terminal=true;}],
 ['incomplete_iteration',s=>{s.analysis.report.completed_iteration=false;}]
];
for(const [label,mutate]of bad){render(valid());value=valid();mutate(value);render(value);cleared(label);checks.push(label);}
value=valid();value.active=true;render(value);
assert.equal(element('rankings').children.length,7);assert.equal(element('analysis-result').hidden,false);
assert.equal(element('play-best').disabled,false);assert.equal(element('analysis-badge').textContent,'Thinking');
checks.push('active_search_preserves_complete_current_depth');


function evaluation(s){render(s);return vm.runInContext('positionEvaluation()',context);}
value=valid();let e=evaluation(value);assert.equal(e.kind,'estimate');assert.equal(e.value,7);assert.equal(element('evaluation-meter').value,7);assert.equal(element('evaluation-meter').hidden,false);assert(!element('evaluation-chances').textContent.includes('%'));assert(element('evaluation-chances').textContent.includes('not calibrated'));
checks.push('A_advantage_uses_current_completed_search_without_fabricated_odds');
function bRoot(score){const s=valid();s.position.controller=1;Object.assign(s.analysis.report,{position:structuredClone(s.position),score_player:1,input_controller:1,normalized_controller:1});s.analysis.report.candidates.forEach(c=>{c.actor=1;c.depth_result={value_actor:score,score_player:1};});return s;}
e=evaluation(bRoot(12));assert.equal(e.value,-12);assert.equal(element('evaluation-meter').value,-12);assert(e.title.includes('player B'));assert(element('evaluation-meter').attributes['aria-valuetext'].includes('B +12 seeds'));checks.push('B_to_move_positive_B_score_favors_B_not_A');
e=evaluation(bRoot(-12));assert.equal(e.value,12);assert(e.title.includes('player A'));checks.push('B_to_move_negative_B_score_favors_A');
value=valid();value.analysis.report.candidates.forEach(c=>c.depth_result.value_actor=0);e=evaluation(value);assert.equal(e.value,0);assert(!e.chances.includes('50%'));checks.push('zero_seed_score_does_not_claim_fifty_percent');
for(const [label,mutate] of bad){value=valid();mutate(value);e=evaluation(value);assert.equal(e.kind,'unknown',label);assert.equal(element('evaluation-meter').hidden,true,label);assert.equal(element('evaluation-meter').value,0,label);}
checks.push('all_stale_illegal_foreign_forced_pass_and_incomplete_reports_hide_bar');
value=valid();value.analysis=null;e=evaluation(value);assert.equal(e.kind,'unknown');checks.push('no_search_does_not_invent_advantage');
value=valid();value.position={pits:[[48,0,0,0,0,0,0],[0,0,0,0,0,0,0]],stores:[50,0],controller:0};value.analysis.current=false;e=evaluation(value);assert.equal(e.kind,'certain');assert.equal(e.winner,0);assert.equal(e.value,98);assert.equal(e.chances,'A win 100% · Draw 0% · B win 0%');assert(e.title.includes('secured'));checks.push('A_fifty_seeds_proves_winner_even_with_stale_analysis');
value=valid();value.position={pits:[[0,0,0,0,0,0,0],[48,0,0,0,0,0,0]],stores:[0,50],controller:1};e=evaluation(value);assert.equal(e.kind,'certain');assert.equal(e.winner,1);assert.equal(e.value,-98);assert(e.chances.includes('B win 100%'));checks.push('B_fifty_seeds_proves_winner');
value=valid();value.position={pits:[[49,0,0,0,0,0,0],[0,0,0,0,0,0,0]],stores:[49,0],controller:0};value.analysis=null;e=evaluation(value);assert.equal(e.kind,'unknown');checks.push('forty_nine_seeds_do_not_prove_a_win');
value=valid();value.position={pits:[Array(7).fill(0),Array(7).fill(0)],stores:[49,49],controller:0};value.terminal=true;e=evaluation(value);assert.equal(e.kind,'certain');assert.equal(e.winner,null);assert.equal(e.value,0);assert.equal(e.chances,'A win 0% · Draw 100% · B win 0%');assert.equal(element('evaluation-meter').className,'draw');checks.push('final_draw_has_separate_exact_draw_outcome');
value.position.stores=[94,4];e=evaluation(value);assert.equal(e.winner,0);assert(e.detail.includes('94'));assert.equal(element('evaluation-meter').className,'');checks.push('actual_terminal_94_4_result_ignores_stale_score');
value.position.stores=[4,94];e=evaluation(value);assert.equal(e.winner,1);assert.equal(element('evaluation-meter').value,-98);checks.push('final_B_win_is_displayed_in_B_direction');
value=valid();value.analysis.report.candidates[0].depth_result.value_actor=100;e=evaluation(value);assert.equal(e.kind,'unknown');checks.push('out_of_range_estimate_is_rejected');
value=valid();value.position.stores=[50,0];e=evaluation(value);assert.equal(e.kind,'unknown');checks.push('invalid_seed_total_cannot_prove_outcome');


const contract={schema:'congklak-product-rules-v1',profile:'id-betawi-relay7-v0',rules_digest:'frozen-test-digest',opening:'sequential',rounds:'single',pits_per_side:7,initial_seeds_per_pit:7};
value=valid();Object.assign(value,{profile:contract.profile,rule_contract:contract,identity:{profile:contract.profile,rules_digest:contract.rules_digest}});render(value);
function imported(v){context.importFixture=v;return vm.runInContext('checkImportedRules(importFixture)',context);}
imported(valid().position);checks.push('bare_stable_position_can_use_explicit_table_rules');
imported({profile:contract.profile,identity:{profile:contract.profile,rules_digest:contract.rules_digest},position:valid().position});checks.push('legacy_sequential_export_still_imports');
imported({rule_contract:Object.fromEntries(Object.entries(contract).reverse()),position:valid().position});checks.push('matching_contract_accepts_any_key_order');
for(const [name,v] of [
 ['simultaneous_contract_refused',{rule_contract:{...contract,opening:'simultaneous'}}],
 ['foreign_digest_refused',{identity:{profile:contract.profile,rules_digest:'other'}}],
 ['foreign_profile_refused',{profile:'other-game'}],
 ['null_contract_refused',{rule_contract:null}],
 ['extra_contract_field_refused',{rule_contract:{...contract,hands:[7,7]}}],
 ['refill_contract_refused',{rule_contract:{...contract,rounds:'refill'}}]
]){assert.throws(()=>imported(v),undefined,name);checks.push(name);}

(async()=>{
  const messages=[];
  const base=valid();Object.assign(base,{ok:true,token:'boot-one',revision:1,jobs:[],sessions:[],last_job:null,message:'Ready',history:{note:'New game',entries:[]},search:{mode:null}});
  element('live-analysis').checked=true;
  context.backend=structuredClone(base);
  context.fetch=async(url,options)=>{
    if(options){const cmd=JSON.parse(options.body);messages.push(cmd);if(cmd.op==='live'){context.backend.active=true;context.backend.search.mode='live';}if(cmd.op==='stop'){context.backend.active=false;}return{json:async()=>structuredClone(context.backend)};}
    return{json:async()=>structuredClone(context.backend)};
  };
  await vm.runInContext('polling=false;poll()',context);
  assert.equal(messages.at(-1).op,'live');assert.equal(element('row-a').children[0].disabled,false);
  const count=messages.length;
  await vm.runInContext('poll()',context);assert.equal(messages.length,count,'no duplicate live command');
  context.backend.active=false;
  await vm.runInContext('poll()',context);assert.equal(messages.length,count,'no busy restart of exhausted root');
  context.backend.revision=2;context.backend.board_key='board-two';context.backend.analysis=null;
  context.backend.history={note:'Recording',turn_message:'A gets another move',entries:[{number:1,actor:0,action:0,store_delta:[1,0],explanation:'A store landing gives another move'}]};
  await vm.runInContext('poll()',context);assert.equal(messages.length,count+1,'automatic analysis on new board');
  assert.equal(element('move-count').textContent,1);assert.equal(element('move-list').children.length,1);
  vm.runInContext("selectRecordTab('rules')",context);assert.equal(element('moves-panel').hidden,true);assert.equal(element('rules-panel').hidden,false);
  vm.runInContext("selectRecordTab('moves')",context);assert.equal(element('moves-panel').hidden,false);
  vm.runInContext("$('stop').onclick()",context);await new Promise(resolve=>setImmediate(resolve));
  assert.equal(element('live-analysis').checked,false);assert.equal(messages.at(-1).op,'stop');
  const stopped=messages.length;await vm.runInContext('poll()',context);assert.equal(messages.length,stopped,'Stop keeps live disabled');
  element('live-analysis').checked=true;context.backend.token='boot-two';
  await vm.runInContext('poll()',context);assert.equal(messages.at(-1).op,'live','restart resumes automatic analysis');
  checks.push('live_auto_start_follow_no_duplicate_stop_and_restart','legal_moves_enabled_during_search','persistent_history_render_and_accessible_tabs');
  const receipt=process.argv[2]||path.join(root,'build/betawi-rules-audit-v1/frontend.json');
  fs.mkdirSync(path.dirname(receipt),{recursive:true});
  fs.writeFileSync(receipt,JSON.stringify({passed:true,checks})+'\n');
  console.log(JSON.stringify({passed:true,checks:checks.length}));
})().catch(e=>{console.error(e);process.exitCode=1;});

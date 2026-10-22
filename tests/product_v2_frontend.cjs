// Execute the real render handlers; stale/illegal root choices must never remain visible.
const fs=require('fs'),vm=require('vm'),assert=require('assert'),crypto=require('crypto'),path=require('path');
const root=path.resolve(__dirname,'..'),elements={};
class Element{
  constructor(){this.children=[];this.value='';this.textContent='';this.hidden=false;this.disabled=false;this.className='';this.events={};}
  append(...items){this.children.push(...items);} replaceChildren(...items){this.children=[...items];}
  addEventListener(kind,fn){this.events[kind]=fn;}setAttribute(){}showModal(){}close(){}focus(){}select(){}
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
  const receipt=process.argv[2]||path.join(root,'build/product-v2/frontend.json');
  fs.mkdirSync(path.dirname(receipt),{recursive:true});
  fs.writeFileSync(receipt,JSON.stringify({passed:true,checks})+'\n');
  console.log(JSON.stringify({passed:true,checks:checks.length}));
})().catch(e=>{console.error(e);process.exitCode=1;});

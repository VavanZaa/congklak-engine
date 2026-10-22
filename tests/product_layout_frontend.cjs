// Exercise the shipped UI on asymmetric positions, including real command handlers.
'use strict';
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert');
const root=path.resolve(__dirname,'..'),source=fs.readFileSync(path.join(root,'product_v2/app.js'),'utf8');
const checks=[];
function harness(storage={}) {
  const elements={},calls=[];
  class Element {
    constructor(tag='div'){this.tag=tag;this.children=[];this.value='';this._text='';this.hidden=false;this.disabled=false;this.checked=false;this.attributes={};this.events={};}
    set id(id){this._id=id;elements[id]=this;} get id(){return this._id;}
    set textContent(text){this._text=String(text);this.children=[];} get textContent(){return this._text+this.children.map(c=>c.textContent).join('');}
    append(...nodes){for(const n of nodes){if(n.parent)n.parent.children=n.parent.children.filter(c=>c!==n);n.parent=this;this.children.push(n);}}
    replaceChildren(...nodes){for(const n of this.children)n.parent=null;this.children=[];this._text='';this.append(...nodes);}
    addEventListener(kind,fn){this.events[kind]=fn;}setAttribute(key,value){this.attributes[key]=String(value);}
    showModal(){this.open=true;}close(){this.open=false;}focus(){}select(){}
  }
  const el=id=>elements[id]||(elements[id]=new Element());
  for(const id of ['board','pit-rows','store-a-cup','store-b-cup','row-a','row-b'])el(id).id=id;
  el('pit-rows').append(el('row-b'),el('row-a'));
  el('board').append(el('store-a-cup'),el('pit-rows'),el('store-b-cup'));
  el('mode').value='explore';
  const context={console,document:{getElementById:el,createElement:tag=>new Element(tag)},window:{},setInterval(){},
    localStorage:{getItem:key=>key==='congklak.live'?'off':storage[key],setItem:(key,value)=>storage[key]=value},fetch:()=>new Promise(()=>{})};
  vm.createContext(context);vm.runInContext(source,context);
  const run=code=>vm.runInContext(code,context);
  function render(state){context.fixture=structuredClone(state);run('render(fixture)');}
  context.fetch=async(url,options)=>{
    if(options)calls.push(JSON.parse(options.body));
    return {json:async()=>structuredClone(context.fixture)};
  };
  return {el,run,render,calls,context,storage};
}
function fixture(actor=0){
  const position={pits:[[0,2,3,4,5,6,7],[1,8,9,10,11,12,13]],stores:[3,4],controller:actor};
  const legal=position.pits[actor].flatMap((n,pit)=>n?[pit]:[]),best=legal[0];
  return {ok:true,token:'layout-fixture',revision:7,board_key:'asymmetric',position,legal_actions:legal,active:false,forced_pass:false,terminal:false,jobs:[],sessions:[],last_job:null,
    message:'Player A played pit 2. The turn ends. Player B moves next.',
    history:{note:'Recorded moves',entries:[{number:1,actor:0,action:1,store_delta:[1,0],explanation:'A gets another move.'},{number:2,actor:1,action:6,store_delta:[0,1],explanation:'B moves.'},{number:3,actor:0,action:'pass',store_delta:[0,0],explanation:'Empty row.'}]},
    analysis:{current:true,compatible:true,board_key:'asymmetric',explanations:{candidates:[{pit:best,immediate:'Choose A1. Opposite B5.'}]},report:{position:structuredClone(position),completed_iteration:true,terminal:false,input_controller:actor,normalized_controller:actor,forced_passes:[],legal_actions:legal,score_player:actor,choice_depth:3,best_moves:[best],candidates:legal.map(pit=>({actor,pit,analysis_complete:true,depth_result:{value_actor:20-pit,score_player:actor},pv:[{actor:0,pit:1},{actor:1,pit:6}]}))}}};
}
const labels=row=>row.children.map(p=>p.children[1].textContent);
const counts=row=>row.children.map(p=>Number(p.children[0].textContent));
const historyLabels=h=>h.el('move-list').children.map(row=>row.children[0].children[1].textContent);
(async()=>{
  const h=harness(),s=fixture();h.render(s);
  assert.equal(h.el('board-layout').value,'left');
  assert.deepEqual(h.el('board').children.map(n=>n.id),['store-a-cup','pit-rows','store-b-cup']);
  assert.deepEqual(labels(h.el('row-a')),['A1','A2','A3','A4','A5','A6','A7']);
  assert.deepEqual(labels(h.el('row-b')),['B7','B6','B5','B4','B3','B2','B1']);
  assert.deepEqual(counts(h.el('row-a')),[7,6,5,4,3,2,0]);
  assert.deepEqual(counts(h.el('row-b')),[1,8,9,10,11,12,13]);
  assert.equal(h.el('store-a').textContent,'3');assert.equal(h.el('store-b').textContent,'4');
  checks.push('default_left_store_and_asymmetric_rows_are_exact_horizontal_reflection');
  assert(h.el('orientation-a').textContent.includes('←'));assert(h.el('orientation-b').textContent.includes('→'));
  assert(h.el('pit-0-6').attributes['aria-label'].includes('pit 1, 7 seeds'));
  assert(h.el('pit-0-0').disabled);assert(!h.el('pit-0-6').disabled);
  checks.push('direction_accessibility_and_empty_pit_legality_follow_display');
  assert.equal(h.el('rankings').children[0].children[1].textContent,'A6');
  assert(h.el('pit-0-1').className.includes('best'));
  assert.equal(h.el('pv').textContent,'A6 → B1');assert.equal(h.el('facts').textContent,'Choose A6. Opposite B2.');
  assert.deepEqual(historyLabels(h),['A6','B1','A passes']);
  assert(h.el('message').textContent.startsWith('Player A played pit 6.'));
  checks.push('rankings_pv_explanations_history_and_status_use_same_mapping');
  await h.el('pit-0-6').events.click();assert.equal(h.calls.at(-1).op,'apply');assert.equal(h.calls.at(-1).action,6);
  checks.push('clicking_display_A1_sends_canonical_action_6');
  const b=fixture(1);h.render(b);await h.el('pit-1-0').events.click();
  assert.equal(h.calls.at(-1).action,0);assert.equal(h.el('rankings').children[0].children[1].textContent,'B7');
  checks.push('player_B_click_and_best_move_keep_canonical_action_0');
  h.render(s);h.run('fillEditor(state.position)');
  assert.deepEqual(h.el('editor-pits').children.map(row=>row.children.map(label=>label._text)),[['B7','B6','B5','B4','B3','B2','B1'],['A1','A2','A3','A4','A5','A6','A7']]);
  assert.equal(Number(h.el('edit-0-6').value),7);assert.equal(Number(h.el('edit-1-0').value),1);
  h.el('edit-0-6').value='8';h.el('edit-store-a').value='2';
  await h.el('editor-form').onsubmit({preventDefault(){}});
  const expected=structuredClone(s.position);expected.pits[0][6]=8;expected.stores[0]=2;
  assert.deepEqual(h.calls.at(-1).position,expected);
  checks.push('mirrored_editor_submits_original_coordinate_order');
  h.el('import-json').value=JSON.stringify({position:s.position});h.el('read-json').onclick();
  assert.equal(h.el('editor-error').hidden,true);assert.equal(Number(h.el('edit-0-6').value),7);
  checks.push('legacy_json_import_fills_mirrored_editor_without_reversing_saved_arrays');
  const before=h.run('JSON.stringify(state)'),commands=h.calls.length;
  h.el('board-layout').value='right';h.el('board-layout').onchange();
  assert.equal(h.run('JSON.stringify(state)'),before);assert.equal(h.calls.length,commands);
  assert.deepEqual(h.el('board').children.map(n=>n.id),['store-b-cup','pit-rows','store-a-cup']);
  assert.deepEqual(counts(h.el('row-a')),[0,2,3,4,5,6,7]);
  assert.equal(h.el('rankings').children[0].children[1].textContent,'A2');
  assert.equal(h.el('pv').textContent,'A2 → B7');assert.equal(h.el('facts').textContent,'Choose A2. Opposite B6.');
  assert.deepEqual(historyLabels(h),['A2','B7','A passes']);assert(h.el('message').textContent.startsWith('Player A played pit 2.'));
  assert.equal(h.storage['congklak.layout'],'right');
  checks.push('layout_toggle_preserves_every_state_field_and_sends_no_commands');
  checks.push('right_store_restores_all_original_labels_and_history');
  const restarted=harness(h.storage);restarted.render(s);assert.equal(restarted.el('board-layout').value,'right');
  assert.deepEqual(counts(restarted.el('row-a')),[0,2,3,4,5,6,7]);
  checks.push('layout_preference_survives_page_restart');
  h.el('board-layout').value='left';h.el('board-layout').onchange();
  assert.deepEqual(historyLabels(h),['A6','B1','A passes']);assert.equal(h.run('JSON.stringify(state)'),before);
  checks.push('round_trip_layout_change_preserves_saved_history_and_state');
  const cert=JSON.parse(fs.readFileSync(path.join(root,'build/product-evaluation-v1/opening-certificate.json'),'utf8'));
  h.context.line=cert.one_based_line;
  assert.deepEqual(Array.from(h.run('line.map(pit=>displayPit(pit-1))')),[7,1,4,7,2,2,5,4,1,6,4,7,1,2]);
  checks.push('verified_winning_line_maps_to_agreed_left_store_labels');
  const receipt=path.join(root,'build/product-layout-v1/frontend.json');
  fs.writeFileSync(receipt,JSON.stringify({passed:true,checks},null,2)+'\n');
  console.log(JSON.stringify({passed:true,checks:checks.length}));
})().catch(error=>{console.error(error);process.exitCode=1;});

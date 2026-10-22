"""Generate versioned copies by checked trace-only insertions; frozen inputs untouched."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'native/ownership_v1'
copies=('engine.hpp','rules.cpp','search.cpp','neural_relay.cpp','neural_puct_cpu.cpp','neural_value_sidecar.cpp')
changes=[]
def insert(text, old, new, count=1):
    found=text.count(old)
    if found!=count:raise RuntimeError(f'Trace insertion mismatch {found}!={count}: {old[:100]}')
    changes.append({'old':old,'new':new,'occurrences':count})
    return text.replace(old,new)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
OUT.mkdir(parents=True,exist_ok=True)
manifest={}
for name in copies:
    source=ROOT/'native'/name
    text=source.read_text()
    before=len(changes)
    if name=='engine.hpp':
        text=insert(text,'struct Control {','struct Control {\n    std::uint64_t ownership_generation=0;')
    if name in ('rules.cpp','search.cpp','neural_relay.cpp'):
        text=insert(text,'#include "engine.hpp"','#include "engine.hpp"\n#include "trace.hpp"')
    if name=='rules.cpp':
        text=insert(text,'    return s;\n}\n\nint total','    owntrace::record(s,owntrace::decode);\n    return s;\n}\n\nint total')
        for signature,op in [('int margin(const Position& s) {','margin_eval'),('bool terminal(const Position& s) {','terminal_test'),('std::vector<int> legal_actions(const Position& s) {','legal'),('Normalized normalize(const Position& s) {','normalize_in')]:
            text=insert(text,signature,signature+'\n    owntrace::record(s,owntrace::'+op+');')
        text=insert(text,'    return n;\n}\n\nTransition apply','    owntrace::record(n.state,owntrace::normalize_out);\n    return n;\n}\n\nTransition apply')
        sig='Transition apply(const Position& input, int pit, bool trace, Control* control) {'
        text=insert(text,sig,sig+'\n    owntrace::record(input,owntrace::move_in,pit);')
        text=insert(text,'    return t;\n}\n\nstd::string escape_json','    owntrace::record(t.state,owntrace::move_out,pit);\n    return t;\n}\n\nstd::string escape_json')
    if name=='search.cpp':
        text=insert(text,'        control->limits = {max_depth, nodes, time_ms, memory_bytes};','        control->limits = {max_depth, nodes, time_ms, memory_bytes};\n        control->ownership_generation=owntrace::new_control_generation();')
        text=insert(text,'    delete static_cast<cc::Control*>(opaque_control);','    if(opaque_control)owntrace::destroyed_control();\n    delete static_cast<cc::Control*>(opaque_control);')
        # The primary node and canonical PV have separate existing counters.
        text=insert(text,'        const auto normalized = normalize(input);','        owntrace::record(input,owntrace::alpha_node,-1,depth);\n        const auto normalized = normalize(input);')
        text=insert(text,'            const auto normalized = normalize(current);','            owntrace::record(current,owntrace::alpha_pv_node,-1,depth);\n            const auto normalized = normalize(current);')
        text=insert(text,'            coverage.nodes = 1;','            coverage.nodes = 1;\n            owntrace::record(root.state,owntrace::alpha_root_node,-1,depth);')
        text=insert(text,'            ++coverage.tt_probes;','            ++coverage.tt_probes;\n            owntrace::record(state,owntrace::alpha_tt_probe,-1,depth);')
        text=insert(text,'                ++coverage.tt_hits;','                ++coverage.tt_hits;\n                owntrace::record(state,owntrace::alpha_tt_hit,-1,depth);')
        text=insert(text,'            entry = &table_[hash & (table_.size() - 1)];','            entry = &table_[hash & (table_.size() - 1)];\n            if(entry->bound != Bound::empty) { Position stored; stored.cells=entry->cells;stored.pits=entry->pits;stored.controller=entry->actor;owntrace::record(stored,owntrace::alpha_tt_slot,-1,entry->depth); }')
    if name=='neural_relay.cpp':
        # Prospective full-raw-state cache contract. The old residual cache was
        # store-offset invariant but did not literally bind both store counts.
        text=insert(text,'std::array<std::uint8_t,14> pits{};','std::array<std::uint8_t,16> pits{};',2)
        text=insert(text,'        h^=static_cast<unsigned>(p.controller);','        pits[14]=p.cells[7];pits[15]=p.cells[15];\n        h^=pits[14];h*=1099511628211ull;h^=pits[15];h*=1099511628211ull;\n        h^=static_cast<unsigned>(p.controller);')
        sig='    const auto p=cc::normalize(input).state;'
        first=text.find(sig)
        if first<0:raise RuntimeError('Missing feature input')
        text=text[:first]+text[first:].replace(sig,'    owntrace::Scope feature_scope(3);\n'+sig+'\n    owntrace::record(p,owntrace::feature_input);',1)
        changes.append({'manual':'feature_scope+feature_input first normalize only'})
        text=insert(text,'        try {t=cc::apply(hypothetical,i,false,control);}','        owntrace::record(hypothetical,owntrace::feature_move_in,i);\n        try {t=cc::apply(hypothetical,i,false,control);}')
        text=insert(text,'        const int offset=43+(row*7+i)*6;','        owntrace::record(t.state,owntrace::feature_move_out,i);\n        const int offset=43+(row*7+i)*6;')
        sig='    Prediction predict(const cc::Position& input,cc::Control* control=nullptr,std::uint64_t* simulations=nullptr,std::uint64_t* deposits=nullptr) const {'
        text=insert(text,sig,sig+'\n        owntrace::record(input,owntrace::model_eval);')
        text=insert(text,'        const auto x=feature_vector(p,input_count,control,simulations,deposits);','        owntrace::record(p,owntrace::model_forward);\n        const auto x=feature_vector(p,input_count,control,simulations,deposits);')
        text=insert(text,'        return out;\n    }\n};\nstruct Entry','        owntrace::record(p,owntrace::model_done);\n        return out;\n    }\n};\nstruct Entry')
        text=insert(text,'    const Cached& predict(const cc::Position& p) {','    const Cached& predict(const cc::Position& p) {\n        owntrace::record(p,owntrace::relay_cache_probe);')
        text=insert(text,'if(e.valid&&e.actor==p.controller&&e.pits==pits) { ++eval_hits; return e; }','if(e.valid&&e.actor==p.controller&&e.pits==pits) { ++eval_hits;owntrace::record(p,owntrace::relay_cache_hit); return e; }')
        text=insert(text,'    int evaluate(const cc::Position& p) {','    int evaluate(const cc::Position& p) {\n        owntrace::record(p,owntrace::relay_eval);')
        text=insert(text,'    int node(const cc::Position& input,int depth,int alpha,int beta) {\n        c.poll(true);','    int node(const cc::Position& input,int depth,int alpha,int beta) {\n        c.poll(true);\n        owntrace::record(input,owntrace::relay_node,-1,depth);')
        text=insert(text,'            c.poll(true); p=cc::normalize(p).state;','            c.poll(true);owntrace::record(p,owntrace::relay_pv_node,-1,depth); p=cc::normalize(p).state;')
        text=insert(text,'            e=&table[h&(table.size()-1)];','            owntrace::record(p,owntrace::relay_tt_probe,-1,depth);\n            e=&table[h&(table.size()-1)];\n            if(e->bound){cc::Position stored;stored.cells=e->cells;stored.controller=e->actor;owntrace::record(stored,owntrace::relay_tt_slot,-1,e->depth);}')
        text=insert(text,'                ++tt_hits; hint=e->pit;','                ++tt_hits;owntrace::record(p,owntrace::relay_tt_hit,-1,depth); hint=e->pit;')
    if name=='neural_value_sidecar.cpp':
        sig='nn::Prediction predict(const Model& model,const cc::Position& input,cc::Control* control,Result& output) {'
        text=insert(text,sig,sig+'\n    owntrace::record(input,owntrace::sidecar_eval);')
        text=insert(text,'        ++output.branch_bypasses;','        ++output.branch_bypasses;owntrace::record(p,owntrace::sidecar_bypass);')
        text=insert(text,'    // Single exact feature extraction; no branch-specific sowing work.','    owntrace::record(p,owntrace::sidecar_branch);\n    owntrace::record(p,owntrace::model_forward);\n    // Single exact feature extraction; no branch-specific sowing work.')
        text=insert(text,'    return out;\n}\n}\nCC_API const char* cc_sidecar_abi','    owntrace::record(p,owntrace::sidecar_done);\n    return out;\n}\n}\nCC_API const char* cc_sidecar_abi')
    dest=OUT/name
    dest.write_text(text,encoding='utf-8',newline='\n')
    manifest[name]={'original':'native/'+name,'original_sha256':sha(source),'generated':'native/ownership_v1/'+name,'generated_sha256':sha(dest),'insertions':changes[before:]}
body={'schema':'ownership-trace-source-projection-v1','copies':manifest,'generator_sha256':sha(Path(__file__)),'selection_or_value_math_modifications':False,'interface_changes':['relay decision-local cache key adds both raw stores; model/rules/backend/search private context'],'trace_only_insertions':False}
p=ROOT/'build/ownership-implementation-v1/native/source-projection.json';p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps(body,sort_keys=True,separators=(',',':'))+'\n')
print(json.dumps({'generated':len(manifest),'trace_insertions':len(changes),'output':str(p)}))

source=ROOT/'congklak/quality_search.py'
text=source.read_text()
text=insert(text,'    q=n["position"];return Node','    q=n["position"];native.trace_mark(q,"puct_node");return Node')
text=insert(text,'            _expand(leaf,row)','            n.trace_mark(leaf.position,"puct_expand")\n            _expand(leaf,row)')
text=insert(text,'                    while node.expanded and not node.terminal:','                    while node.expanded and not node.terminal:\n                        n.trace_mark(node.position,"puct_select")')
dest=ROOT/'congklak/quality_ownership_v1.py'
dest.write_text(text,encoding='utf-8',newline='\n')
pp=ROOT/'build/ownership-implementation-v1/native/python-projection.json'
pp.write_text(json.dumps({'schema':'ownership-python-search-projection-v1','original':'congklak/quality_search.py','original_sha256':sha(source),'generated':'congklak/quality_ownership_v1.py','generated_sha256':sha(dest),'trace_insertions':changes[-3:],'selection_or_value_math_modifications':False},sort_keys=True,separators=(',',':'))+'\n')

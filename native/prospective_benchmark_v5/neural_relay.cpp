// Versioned exact relay-feature value search; frozen engines remain untouched.
#include "engine.hpp"
#include "trace.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <limits>
#include <memory>
#include <immintrin.h>
#include <intrin.h>

namespace nn {
constexpr int scale=256, infinity=1000000000;
constexpr const char* policy="terminal-first/exact-relay-features/double-fixed256/full-root/pvs-relay-v1";
const char* mode_name(int mode) { const char* names[]={"h-none","h-policy","value-none","value-policy"};return names[mode]; }
struct Prediction { double residual=0;std::array<double,7> logits{}; };
bool avx2_available() {
    int registers[4]{};
    __cpuid(registers,0); if(registers[0]<7)return false;
    __cpuid(registers,1);
    if((registers[2]&(1<<27))==0 || (registers[2]&(1<<28))==0)return false;
    if((_xgetbv(0)&6)!=6)return false;
    __cpuidex(registers,7,0); return (registers[1]&(1<<5))!=0;
}
// Input-major packing keeps adjacent output neurons contiguous. Each lane still
// accumulates input 0,1,... in the frozen scalar order; separate mul/add forbids FMA.
void dense_scalar(const double* packed,const double* bias,const double* x,int inputs,int outputs,double* y) {
    for(int i=0;i<outputs;++i) {
        double v=0;
        for(int j=0;j<inputs;++j)v+=packed[static_cast<std::size_t>(j)*outputs+i]*x[j];
        y[i]=std::max(0.0,v+bias[i]);
    }
}
__declspec(noinline) void dense_avx2(const double* packed,const double* bias,const double* x,int inputs,int outputs,double* y) {
    int i=0;
    for(;i+4<=outputs;i+=4) {
        __m256d v=_mm256_setzero_pd();
        for(int j=0;j<inputs;++j) {
            const __m256d w=_mm256_loadu_pd(packed+static_cast<std::size_t>(j)*outputs+i);
            v=_mm256_add_pd(v,_mm256_mul_pd(w,_mm256_set1_pd(x[j])));
        }
        v=_mm256_add_pd(v,_mm256_loadu_pd(bias+i));
        _mm256_storeu_pd(y+i,_mm256_max_pd(v,_mm256_setzero_pd()));
    }
    for(;i<outputs;++i) {
        double v=0;
        for(int j=0;j<inputs;++j)v+=packed[static_cast<std::size_t>(j)*outputs+i]*x[j];
        y[i]=std::max(0.0,v+bias[i]);
    }
}
constexpr const char* feature_policy="normalized-controller/hypothetical-own7-opponent7/exact-relay6-unclipped-v1";
std::array<double,127> feature_vector(const cc::Position& input,int input_count,cc::Control* control=nullptr,
    std::uint64_t* simulations=nullptr,std::uint64_t* deposits=nullptr) {
    owntrace::Scope feature_scope(3);
    const auto p=cc::normalize(input).state;
    owntrace::record(p,owntrace::feature_input);
    std::array<double,127> x{};int active=0;
    for(int row=0;row<2;++row)for(int i=0;i<7;++i) {
        const int count=p.cells[(row==0?p.controller:1-p.controller)*8+i];
        x[row*7+i]=count/98.0;active+=count;
        x[15+row*7+i]=count>0?1.0:0.0;x[29+row*7+i]=count==1?1.0:0.0;
    }
    x[14]=active/98.0;
    if(input_count==43||!active)return x;
    for(int row=0;row<2;++row)for(int i=0;i<7;++i) {
        if(control)control->poll();
        const int actor=row==0?p.controller:1-p.controller;
        if(!p.cells[actor*8+i])continue;
        auto hypothetical=p;hypothetical.controller=actor;
        if(simulations)++*simulations;
        const auto before=control?control->deposits:0;
        cc::Transition t;
        owntrace::record(hypothetical,owntrace::feature_move_in,i);
        try {t=cc::apply(hypothetical,i,false,control);}
        catch(...) {if(deposits&&control)*deposits+=control->deposits-before;throw;}
        if(deposits)*deposits+=control?control->deposits-before:static_cast<std::uint64_t>(t.drops);
        owntrace::record(t.state,owntrace::feature_move_out,i);
        const int offset=43+(row*7+i)*6;
        x[offset]=t.store_delta[actor]/98.0;x[offset+1]=t.drops/98.0;x[offset+2]=t.relays/98.0;
        x[offset+3]=t.reason=="own_capture"?1.0:0.0;
        x[offset+4]=!t.terminal&&t.state.controller==actor?1.0:0.0;x[offset+5]=t.terminal?1.0:0.0;
    }
    if(control)control->poll();
    return x;
}
struct Model {
    int width,input_count=43,mode=0,policy_min_depth=2;
    bool avx2=false;
    std::string id;
    std::vector<double> weights;
    std::vector<double> first,second;
    void pack() {
        first.resize(static_cast<std::size_t>(width)*input_count); second.resize(static_cast<std::size_t>(width)*width);
        for(int i=0;i<width;++i)for(int j=0;j<input_count;++j)first[static_cast<std::size_t>(j)*width+i]=weights[static_cast<std::size_t>(i)*input_count+j];
        const auto offset=static_cast<std::size_t>(width)*(input_count+1);
        for(int i=0;i<width;++i)for(int j=0;j<width;++j)second[static_cast<std::size_t>(j)*width+i]=weights[offset+static_cast<std::size_t>(i)*width+j];
    }
    Prediction predict(const cc::Position& input,cc::Control* control=nullptr,std::uint64_t* simulations=nullptr,std::uint64_t* deposits=nullptr) const {
        owntrace::record(input,owntrace::model_eval);
        const auto p=cc::normalize(input).state;
        if(cc::terminal(p)) return {};
        std::array<double,256> a{},b{};
        owntrace::record(p,owntrace::model_forward);
        const auto x=feature_vector(p,input_count,control,simulations,deposits);
        const int active=cc::total(p)-p.cells[7]-p.cells[15];
        // Export order is weight then bias for each dense layer.
        const auto dense=avx2?dense_avx2:dense_scalar;
        dense(first.data(),weights.data()+static_cast<std::size_t>(width)*input_count,x.data(),input_count,width,a.data());
        std::size_t offset=static_cast<std::size_t>(width)*(input_count+1);
        dense(second.data(),weights.data()+offset+static_cast<std::size_t>(width)*width,a.data(),width,width,b.data());
        offset+=static_cast<std::size_t>(width)*(width+1);
        double v=0;
        for(int i=0;i<width;++i) v+=weights[offset+i]*b[i];
        Prediction out;out.residual=(p.controller==0?active:-active)*std::tanh(v+weights[offset+width]);
        offset+=width+1;
        for(int action=0;action<7;++action) {
            double logit=0;
            for(int i=0;i<width;++i)logit+=weights[offset+static_cast<std::size_t>(action)*width+i]*b[i];
            out.logits[action]=logit+weights[offset+static_cast<std::size_t>(7)*width+action];
        }
        owntrace::record(p,owntrace::model_done);
        return out;
    }
};
struct Entry {
    std::array<std::uint8_t,16> cells{};
    std::uint64_t hash=0;
    int value=0;
    std::uint8_t actor=0,depth=0,bound=0;
    std::int8_t pit=-1;
};
struct Cached {
    std::array<std::uint8_t,16> pits{};
    std::uint8_t actor=0;
    bool valid=false;
    int residual=0;
    std::array<double,7> logits{};
};
struct RelayWorkspace {
    std::vector<Entry> table;std::vector<Cached> cache;bool leased=false;
    std::uint64_t epoch=0,cleared_entries=0,empty_entries=0,cleared_cache=0,empty_cache=0,control_generation=0,start_nodes=0,start_deposits=0;
};
thread_local RelayWorkspace relay_workspace;

std::uint64_t hash(const cc::Position& p,int depth) {
    std::uint64_t h=14695981039346656037ull;
    for(auto c:p.cells) { h^=c; h*=1099511628211ull; }
    h^=static_cast<unsigned>(p.controller); h*=1099511628211ull;
    h^=static_cast<unsigned>(depth); h*=1099511628211ull;
    h^=h>>33; h*=0xff51afd7ed558ccdull; return h^(h>>33);
}
struct Move { cc::Transition transition; };
std::string pv_json(const std::vector<Move>& pv) {
    std::string s="[";
    for(const auto& m:pv) {
        if(s.size()>1) s+=',';
        const auto& t=m.transition;
        s+="{\"actor\":"+std::to_string(t.actor)+",\"pit\":"+std::to_string(t.pit)
            +",\"next_controller\":"+(t.terminal?"null":std::to_string(t.state.controller))
            +",\"terminal_after\":"+(t.terminal?"true":"false")
            +",\"bonus_choice\":"+(!t.terminal&&t.reason=="own_store"?"true":"false")
            +",\"forced_passes_after\":"+cc::passes_json(t.passes)+"}";
    }
    return s+"]";
}
std::string ints(const std::vector<int>& v) {
    std::string s="["; for(int a:v) { if(s.size()>1)s+=','; s+=std::to_string(a); } return s+"]";
}
class Search {
    cc::Control& c;
    const Model& model;
    std::vector<Entry> table;
    std::vector<Cached> cache;
public:
    std::uint64_t evaluations=0,eval_hits=0,tt_hits=0,pvs_probes=0,researches=0,policy_nodes=0,feature_simulations=0,feature_deposits=0;
    Search(cc::Control& control,const Model& m):c(control),model(m) {
        c.poll();
        std::size_t capacity=0;
        if(c.tt_mb) {
            const auto entries=static_cast<std::size_t>(c.tt_mb)*1024*1024/sizeof(Entry);
            capacity=1; while(capacity<=entries/2) capacity*=2;
        }
        const auto cache_capacity=model.mode?8192:0;
        const auto bytes=capacity*sizeof(Entry)+cache_capacity*sizeof(Cached);
        if(c.peak_process_bytes>=c.limits.memory_bytes || bytes>c.limits.memory_bytes-c.peak_process_bytes) {
            c.stop_reason="memory_budget"; throw cc::Stopped{};
        }
        if(relay_workspace.leased||relay_workspace.table.size()!=capacity||relay_workspace.cache.size()!=cache_capacity)
            throw cc::Error("workspace_identity","Preallocated relay workspace missing or leased");
        table.swap(relay_workspace.table);cache.swap(relay_workspace.cache);relay_workspace.leased=true;workspace_borrowed=true;
        try {
            std::fill(table.begin(),table.end(),Entry{});std::fill(cache.begin(),cache.end(),Cached{});
            relay_workspace.cleared_entries=table.size();relay_workspace.cleared_cache=cache.size();relay_workspace.empty_entries=0;relay_workspace.empty_cache=0;
            for(const auto& entry:table) if(entry.bound==0) ++relay_workspace.empty_entries;
            for(const auto& entry:cache) if(!entry.valid) ++relay_workspace.empty_cache;
            ++relay_workspace.epoch;relay_workspace.control_generation=c.ownership_generation;
            relay_workspace.start_nodes=c.nodes;relay_workspace.start_deposits=c.deposits;
            c.memory_checked=false;c.poll();
        } catch(...) { release_workspace();throw; }
    }
    void release_workspace() { if(workspace_borrowed) { table.swap(relay_workspace.table);cache.swap(relay_workspace.cache);relay_workspace.leased=false;workspace_borrowed=false; } }
    ~Search() { release_workspace(); }
    bool workspace_borrowed=false;
    const Cached& predict(const cc::Position& p) {
        owntrace::record(p,owntrace::relay_cache_probe);
        std::array<std::uint8_t,16> pits{};
        std::uint64_t h=14695981039346656037ull;
        for(int row=0;row<2;++row) for(int i=0;i<7;++i) {
            pits[row*7+i]=p.cells[row*8+i]; h^=pits[row*7+i]; h*=1099511628211ull;
        }
        pits[14]=p.cells[7];pits[15]=p.cells[15];
        h^=pits[14];h*=1099511628211ull;h^=pits[15];h*=1099511628211ull;
        h^=static_cast<unsigned>(p.controller);
        auto& e=cache[h&(cache.size()-1)];
        if(e.valid&&e.actor==p.controller&&e.pits==pits) { ++eval_hits;owntrace::record(p,owntrace::relay_cache_hit); return e; }
        ++evaluations;
        // Round the residual separately so store-offset translation is exact.
        const auto prediction=model.predict(p,&c,&feature_simulations,&feature_deposits);
        const int residual=static_cast<int>(std::round(prediction.residual*scale));
        e={pits,static_cast<std::uint8_t>(p.controller),true,residual,prediction.logits};
        c.poll();
        return e;
    }
    int evaluate(const cc::Position& p) {
        owntrace::record(p,owntrace::relay_eval);
        if(cc::terminal(p)||model.mode<2)return cc::margin(p)*scale;
        return cc::margin(p)*scale+predict(p).residual;
    }
    int node(const cc::Position& input,int depth,int alpha,int beta) {
        c.poll(true);
        owntrace::record(input,owntrace::relay_node,-1,depth);
        const auto normalized=cc::normalize(input); const auto& p=normalized.state;
        if(normalized.terminal || depth==0) return evaluate(p);
        const int old_alpha=alpha,old_beta=beta;
        const auto h=hash(p,depth); Entry* e=nullptr; int hint=-1;
        if(!table.empty()) {
            owntrace::record(p,owntrace::relay_tt_probe,-1,depth);
            e=&table[h&(table.size()-1)];
            if(e->bound){cc::Position stored;stored.cells=e->cells;stored.controller=e->actor;owntrace::record(stored,owntrace::relay_tt_slot,-1,e->depth);}
            if(e->bound&&e->hash==h&&e->cells==p.cells&&e->actor==p.controller&&e->depth==depth) {
                ++tt_hits;owntrace::record(p,owntrace::relay_tt_hit,-1,depth); hint=e->pit;
                if(e->bound==1 || (e->bound==2&&e->value>=beta) || (e->bound==3&&e->value<=alpha)) return e->value;
            }
        }
        auto actions=cc::legal_actions(p); std::sort(actions.begin(),actions.end());
        if((model.mode&1)&&depth>=model.policy_min_depth&&actions.size()>1) {
            const auto logits=predict(p).logits;++policy_nodes;
            std::stable_sort(actions.begin(),actions.end(),[&logits](int a,int b){return logits[a]>logits[b];});
        }
        if(hint>=0) { auto i=std::find(actions.begin(),actions.end(),hint); if(i!=actions.end()) std::rotate(actions.begin(),i,i+1); }
        const bool maximize=p.controller==0; int best=maximize?-infinity:infinity,pit=-1; bool first=true;
        for(int a:actions) {
            const auto t=cc::apply(p,a,false,&c); int value;
            if(c.use_pvs&&!first&&beta>alpha+1) {
                ++pvs_probes;
                value=node(t.state,depth-1,maximize?alpha:beta-1,maximize?alpha+1:beta);
                if(value>alpha&&value<beta) { ++researches; value=node(t.state,depth-1,alpha,beta); }
            } else value=node(t.state,depth-1,alpha,beta);
            first=false;
            if((maximize&&value>best)||(!maximize&&value<best)) { best=value; pit=a; }
            if(maximize) alpha=std::max(alpha,best); else beta=std::min(beta,best);
            if(alpha>=beta) break;
        }
        c.poll();
        if(e) *e={p.cells,h,best,static_cast<std::uint8_t>(p.controller),static_cast<std::uint8_t>(depth),
                   static_cast<std::uint8_t>(best<=old_alpha?3:best>=old_beta?2:1),static_cast<std::int8_t>(pit)};
        return best;
    }
    std::vector<Move> pv(cc::Position p,int depth,int target) {
        std::vector<Move> out;
        while(true) {
            c.poll(true);owntrace::record(p,owntrace::relay_pv_node,-1,depth); p=cc::normalize(p).state;
            if(cc::terminal(p)||depth==0) {
                if(evaluate(p)!=target) throw cc::Error("pv_invariant","Neural PV leaf differs");
                return out;
            }
            bool found=false;
            for(int a:cc::legal_actions(p)) {
                auto t=cc::apply(p,a,false,&c);
                if(node(t.state,depth-1,-infinity,infinity)==target) { p=t.state; out.push_back({std::move(t)}); --depth; found=true; break; }
            }
            if(!found) throw cc::Error("pv_invariant","No neural optimal continuation");
        }
    }
    std::size_t table_bytes() const { return table.size()*sizeof(Entry); }
};
std::string analyze(const cc::Position& input,cc::Control& c,const Model& model,int start,cc_progress_fn progress,void* user) {
    if(start<0||start>128) throw cc::Error("invalid_start_depth","Depth must be 0..128");
    c.start=std::chrono::steady_clock::now(); c.nodes=0; c.deposits=0; c.last_memory_work=0;
    c.memory_checked=false; c.peak_process_bytes=0; c.stop_reason.clear();
    const auto normal=cc::normalize(input); const auto root=normal.state;
    const auto actions=cc::legal_actions(root); std::string last="null"; int completed=-1;
    try {
        Search search(c,model);
        std::vector<cc::Transition> transitions;
        for(int a:actions) transitions.push_back(cc::apply(root,a,true,&c));
        const int first=normal.terminal||c.limits.max_depth==0?0:std::max(1,start);
        const int maximum=normal.terminal?0:c.limits.max_depth;
        for(int depth=first;depth<=maximum;++depth) {
            std::string candidates="["; std::vector<int> best_moves; std::vector<Move> best_pv;
            int best=search.evaluate(root);
            if(depth>0&&!normal.terminal) best=root.controller==0?-infinity:infinity;
            for(const auto& t:transitions) {
                if(candidates.size()>1)candidates+=',';
                int value=0; std::vector<Move> pv;
                if(depth>0) {
                    value=search.node(t.state,depth-1,-infinity,infinity);
                    pv=search.pv(t.state,depth-1,value); pv.insert(pv.begin(),{t});
                    const bool better=root.controller==0?value>best:value<best;
                    if(better) { best=value; best_moves={t.pit}; best_pv=pv; }
                    else if(value==best) best_moves.push_back(t.pit);
                }
                candidates+="{\"pit\":"+std::to_string(t.pit)+",\"value_units\":"+(depth?std::to_string(value):"null")
                    +",\"transition\":"+cc::transition_json(t)+",\"pv\":"+pv_json(pv)+"}";
            }
            candidates+=']'; c.poll();
            const auto elapsed=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-c.start).count();
            const std::string report="{\"schema\":\"neural-analysis-relay-v1\",\"completed_iteration\":true,\"choice_depth\":"+std::to_string(depth)
                +",\"position\":"+cc::position_json(root)+",\"input_controller\":"+std::to_string(input.controller)
                +",\"forced_passes\":"+cc::passes_json(normal.passes)+",\"terminal\":"+(normal.terminal?"true":"false")
                +",\"model_sha256\":"+cc::escape_json(model.id)+",\"policy\":"+cc::escape_json(policy)
                +",\"mode\":"+cc::escape_json(mode_name(model.mode))+",\"policy_min_depth\":"+std::to_string(model.policy_min_depth)
                +",\"score_scale\":256,\"value_units\":"+std::to_string(best)+",\"best_moves\":"+ints(best_moves)
                +",\"candidates\":"+candidates+",\"pv\":"+pv_json(best_pv)+",\"proof_status\":\""
                +(normal.terminal?"terminal_outcome":model.mode<2?"finite_horizon_result":"finite_horizon_neural_estimate")+"\",\"nodes\":"+std::to_string(c.nodes)
                +",\"deposits\":"+std::to_string(c.deposits)+",\"elapsed_ms\":"+std::to_string(elapsed)
                +",\"tt_allocated_bytes\":"+std::to_string(search.table_bytes())+",\"nn_evaluations\":"+std::to_string(search.evaluations)
                +",\"nn_cache_hits\":"+std::to_string(search.eval_hits)+",\"tt_hits\":"+std::to_string(search.tt_hits)
                +",\"pvs_probes\":"+std::to_string(search.pvs_probes)+",\"pvs_researches\":"+std::to_string(search.researches)
                +",\"policy_ordered_nodes\":"+std::to_string(search.policy_nodes)
                +",\"feature_simulations\":"+std::to_string(search.feature_simulations)+",\"feature_deposits\":"+std::to_string(search.feature_deposits)+"}";
            if(progress&&!progress(report.c_str(),user)) { c.stop_reason="callback_failed"; throw cc::Stopped{}; }
            last=report; completed=depth;
        }
        c.stop_reason="max_depth";
    } catch(const cc::Stopped&) { if(c.stop_reason.empty())c.stop_reason="cancelled"; }
      catch(const std::bad_alloc&) { c.stop_reason="memory_budget"; }
    const auto elapsed=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-c.start).count();
    return "{\"ok\":true,\"stop_reason\":"+cc::escape_json(c.stop_reason)+",\"last_completed_depth\":"+std::to_string(completed)
        +",\"nodes\":"+std::to_string(c.nodes)+",\"deposits\":"+std::to_string(c.deposits)+",\"elapsed_ms\":"+std::to_string(elapsed)+",\"report\":"+last+"}";
}
}
CC_API void* cc_nn_create_model_relay(const float* weights,int count,int width,int input_count,const char* id,int backend,int mode,int policy_min_depth) {
    if(!weights||!id||width<16||width>256||(input_count!=43&&input_count!=127)||(mode!=0&&mode!=2)||policy_min_depth<2||policy_min_depth>128
       ||count!=width*width+(input_count+10)*width+8||std::strlen(id)!=64||backend<0||backend>2)return nullptr;
    try {
        for(int i=0;i<count;++i)if(!std::isfinite(weights[i]))return nullptr;
        const bool available=nn::avx2_available(); if(backend==2&&!available)return nullptr;
        auto m=std::make_unique<nn::Model>(); m->width=width; m->input_count=input_count; m->avx2=backend!=1&&available;
        m->mode=mode;m->policy_min_depth=policy_min_depth;
        m->id=id; m->weights.assign(weights,weights+count); m->pack(); return m.release();
    } catch(...) { return nullptr; }
}
CC_API const char* cc_nn_backend(void* model) { return model?(static_cast<nn::Model*>(model)->avx2?"avx2-double-packed":"scalar-double-packed"):"closed"; }
CC_API int cc_nn_avx2_available() { return nn::avx2_available()?1:0; }
CC_API const char* cc_nn_measure_inference(const int* cells,const int* actors,int count,int repeats,void* opaque) {
    thread_local std::string out;
    try {
        if(!cells||!actors||!opaque||count<1||count>10000||repeats<1||repeats>10000
           ||static_cast<std::uint64_t>(count)*repeats>1000000)throw cc::Error("invalid_benchmark","Inference benchmark limit");
        std::vector<cc::Position> positions; positions.reserve(count);
        for(int i=0;i<count;++i)positions.push_back(cc::decode(cells+static_cast<std::size_t>(i)*16,7,actors[i],98));
        const auto& m=*static_cast<nn::Model*>(opaque); double checksum=0;
        const auto before=std::chrono::steady_clock::now();
        // No evaluation cache, search, JSON, or Python boundary inside this loop.
        for(int r=0;r<repeats;++r)for(const auto& p:positions) {
            const auto prediction=m.predict(p);checksum+=prediction.residual;
            for(double logit:prediction.logits)checksum+=logit;
        }
        const auto elapsed=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-before).count();
        char buffer[256]; std::snprintf(buffer,sizeof(buffer),"{\"ok\":true,\"count\":%d,\"repeats\":%d,\"elapsed_ns\":%.17g,\"checksum\":%.17g}",count,repeats,elapsed,checksum);
        out=buffer;
    } catch(const cc::Error& e) { out=cc::error_json(e.code,e.what()); }
      catch(...) { out=cc::error_json("native_error","Inference benchmark failed"); }
    return out.c_str();
}
CC_API void cc_nn_destroy_model(void* model) { delete static_cast<nn::Model*>(model); }
CC_API const char* cc_nn_evaluate(const int* cells,int actor,void* opaque) {
    thread_local std::string out;
    try {
        if(!opaque)throw cc::Error("invalid_model","Model is required");
        const auto p=cc::decode(cells,7,actor,98); const auto& m=*static_cast<nn::Model*>(opaque);
        const auto prediction=m.predict(p);const double raw=cc::margin(p)+prediction.residual;
        const int model_units=cc::margin(p)*nn::scale+static_cast<int>(std::round(prediction.residual*nn::scale));
        char buffer[128];std::snprintf(buffer,sizeof(buffer),"%.17g",m.mode<2?static_cast<double>(cc::margin(p)):raw);
        out="{\"ok\":true,\"value_player0\":"+std::string(buffer)+",\"value_units\":"+std::to_string(m.mode<2?cc::margin(p)*nn::scale:model_units)+",\"score_scale\":256";
        std::snprintf(buffer,sizeof(buffer),"%.17g",raw);out+=",\"model_value_player0\":"+std::string(buffer)+",\"model_value_units\":"+std::to_string(model_units)+",\"policy_logits\":[";
        for(int i=0;i<7;++i){if(i)out+=',';std::snprintf(buffer,sizeof(buffer),"%.17g",prediction.logits[i]);out+=buffer;}
        out+="]}";
    } catch(const cc::Error& e) { out=cc::error_json(e.code,e.what()); }
      catch(...) { out=cc::error_json("native_error","Neural evaluation failed"); }
    return out.c_str();
}
CC_API const char* cc_nn_analyze(const int* cells,int actor,void* token,void* opaque,int start,cc_progress_fn progress,void* user) {
    thread_local std::string out;
    try {
        if(!token||!opaque)throw cc::Error("invalid_model","Control and model are required");
        auto& c=*static_cast<cc::Control*>(token); bool expected=false;
        if(!c.search_running.compare_exchange_strong(expected,true))throw cc::Error("control_busy","Control is running");
        struct Guard { cc::Control& c; ~Guard(){c.search_running.store(false);} } guard{c};
        out=nn::analyze(cc::decode(cells,7,actor,98),c,*static_cast<nn::Model*>(opaque),start,progress,user);
    } catch(const cc::Error& e) { out=cc::error_json(e.code,e.what()); }
      catch(const std::exception& e) { out=cc::error_json("native_error",e.what()); }
      catch(...) { out=cc::error_json("native_error","Neural search failed"); }
    return out.c_str();
}

CC_API const char* cc_nn_features(const int* cells,int actor,int input_count) {
    thread_local std::string out;
    try {
        if(input_count!=43&&input_count!=127)throw cc::Error("invalid_features","Expected43or127inputs");
        const auto p=cc::decode(cells,7,actor,98);std::uint64_t simulations=0,deposits=0;
        const auto x=nn::feature_vector(p,input_count,nullptr,&simulations,&deposits);
        out="{\"ok\":true,\"features\":[";char buffer[64];
        for(int i=0;i<input_count;++i){if(i)out+=',';std::snprintf(buffer,sizeof(buffer),"%.17g",x[i]);out+=buffer;}
        out+="],\"feature_simulations\":"+std::to_string(simulations)+",\"feature_deposits\":"+std::to_string(deposits)+"}";
    } catch(const cc::Error& e) {out=cc::error_json(e.code,e.what());}
      catch(...) {out=cc::error_json("native_error","Feature extraction failed");}
    return out.c_str();
}

CC_API int cc_benchmark_relay_workspace_prepare(unsigned mb){
    auto& pool=nn::relay_workspace;if(mb!=16||pool.leased)return 0;
    try{std::size_t capacity=1;const auto maximum=static_cast<std::size_t>(mb)*1024*1024/sizeof(nn::Entry);
        while(capacity<=maximum/2)capacity*=2;pool.table.resize(capacity);pool.cache.resize(8192);
        std::fill(pool.table.begin(),pool.table.end(),nn::Entry{});std::fill(pool.cache.begin(),pool.cache.end(),nn::Cached{});return 1;
    }catch(...){return 0;}
}
CC_API const char* cc_benchmark_relay_workspace_status(){
    thread_local std::string out;const auto& p=nn::relay_workspace;
    out="{\"tt_mb\":16,\"capacity\":"+std::to_string(p.table.size())+",\"cache_capacity\":"+std::to_string(p.cache.size())
      +",\"leased\":"+(p.leased?"true":"false")+",\"epoch\":"+std::to_string(p.epoch)+",\"cleared_entries\":"+std::to_string(p.cleared_entries)
      +",\"empty_entries\":"+std::to_string(p.empty_entries)+",\"cleared_cache\":"+std::to_string(p.cleared_cache)+",\"empty_cache\":"+std::to_string(p.empty_cache)
      +",\"control_generation\":"+std::to_string(p.control_generation)+",\"start_nodes\":"+std::to_string(p.start_nodes)+",\"start_deposits\":"+std::to_string(p.start_deposits)+"}";return out.c_str();
}
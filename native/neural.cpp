// Optional neural analyzer. The original simulator, DLL and H_v1 remain frozen.
#include "engine.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <limits>
#include <memory>

namespace nn {
constexpr int scale=256, infinity=1000000000;
constexpr const char* policy="terminal-first/neural-double-fixed256/full-root/pvs-v1";
struct Model {
    int width;
    std::string id;
    std::vector<double> weights;
    double residual(const cc::Position& input) const {
        const auto p=cc::normalize(input).state;
        if(cc::terminal(p)) return 0;
        std::array<double,256> a{},b{};
        std::array<double,15> x{};
        int active=0;
        for(int row=0;row<2;++row) for(int i=0;i<7;++i) {
            const int count=p.cells[(row==0?p.controller:1-p.controller)*8+i];
            x[row*7+i]=count/98.0; active+=count;
        }
        x[14]=active/98.0;
        // Export order is weight then bias for each dense layer.
        std::size_t offset=0;
        for(int i=0;i<width;++i) {
            double v=0;
            for(int j=0;j<15;++j) v+=weights[offset+static_cast<std::size_t>(i)*15+j]*x[j];
            a[i]=std::max(0.0,v+weights[offset+static_cast<std::size_t>(width)*15+i]);
        }
        offset+=static_cast<std::size_t>(width)*16;
        for(int i=0;i<width;++i) {
            double v=0;
            for(int j=0;j<width;++j) v+=weights[offset+static_cast<std::size_t>(i)*width+j]*a[j];
            b[i]=std::max(0.0,v+weights[offset+static_cast<std::size_t>(width)*width+i]);
        }
        offset+=static_cast<std::size_t>(width)*(width+1);
        double v=0;
        for(int i=0;i<width;++i) v+=weights[offset+i]*b[i];
        return (p.controller==0?active:-active)*std::tanh(v+weights[offset+width]);
    }
    double raw(const cc::Position& p) const { return cc::margin(p)+residual(p); }
};
struct Entry {
    std::array<std::uint8_t,16> cells{};
    std::uint64_t hash=0;
    int value=0;
    std::uint8_t actor=0,depth=0,bound=0;
    std::int8_t pit=-1;
};
struct Cached {
    std::array<std::uint8_t,14> pits{};
    std::uint8_t actor=0;
    bool valid=false;
    int residual=0;
};
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
    std::uint64_t evaluations=0,eval_hits=0,tt_hits=0,pvs_probes=0,researches=0;
    Search(cc::Control& control,const Model& m):c(control),model(m) {
        c.poll();
        std::size_t capacity=0;
        if(c.tt_mb) {
            const auto entries=static_cast<std::size_t>(c.tt_mb)*1024*1024/sizeof(Entry);
            capacity=1; while(capacity<=entries/2) capacity*=2;
        }
        const auto bytes=capacity*sizeof(Entry)+8192*sizeof(Cached);
        if(c.peak_process_bytes>=c.limits.memory_bytes || bytes>c.limits.memory_bytes-c.peak_process_bytes) {
            c.stop_reason="memory_budget"; throw cc::Stopped{};
        }
        table.resize(capacity); cache.resize(8192); c.memory_checked=false; c.poll();
    }
    int evaluate(const cc::Position& p) {
        if(cc::terminal(p)) return cc::margin(p)*scale;
        std::array<std::uint8_t,14> pits{};
        std::uint64_t h=14695981039346656037ull;
        for(int row=0;row<2;++row) for(int i=0;i<7;++i) {
            pits[row*7+i]=p.cells[row*8+i]; h^=pits[row*7+i]; h*=1099511628211ull;
        }
        h^=static_cast<unsigned>(p.controller);
        auto& e=cache[h&(cache.size()-1)];
        if(e.valid&&e.actor==p.controller&&e.pits==pits) { ++eval_hits; return cc::margin(p)*scale+e.residual; }
        ++evaluations;
        // Round the residual separately so store-offset translation is exact.
        const int residual=static_cast<int>(std::round(model.residual(p)*scale));
        e={pits,static_cast<std::uint8_t>(p.controller),true,residual};
        c.poll();
        return cc::margin(p)*scale+residual;
    }
    int node(const cc::Position& input,int depth,int alpha,int beta) {
        c.poll(true);
        const auto normalized=cc::normalize(input); const auto& p=normalized.state;
        if(normalized.terminal || depth==0) return evaluate(p);
        const int old_alpha=alpha,old_beta=beta;
        const auto h=hash(p,depth); Entry* e=nullptr; int hint=-1;
        if(!table.empty()) {
            e=&table[h&(table.size()-1)];
            if(e->bound&&e->hash==h&&e->cells==p.cells&&e->actor==p.controller&&e->depth==depth) {
                ++tt_hits; hint=e->pit;
                if(e->bound==1 || (e->bound==2&&e->value>=beta) || (e->bound==3&&e->value<=alpha)) return e->value;
            }
        }
        auto actions=cc::legal_actions(p); std::sort(actions.begin(),actions.end());
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
            c.poll(true); p=cc::normalize(p).state;
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
            const std::string report="{\"schema\":\"neural-analysis-v1\",\"completed_iteration\":true,\"choice_depth\":"+std::to_string(depth)
                +",\"position\":"+cc::position_json(root)+",\"input_controller\":"+std::to_string(input.controller)
                +",\"forced_passes\":"+cc::passes_json(normal.passes)+",\"terminal\":"+(normal.terminal?"true":"false")
                +",\"model_sha256\":"+cc::escape_json(model.id)+",\"policy\":"+cc::escape_json(policy)
                +",\"score_scale\":256,\"value_units\":"+std::to_string(best)+",\"best_moves\":"+ints(best_moves)
                +",\"candidates\":"+candidates+",\"pv\":"+pv_json(best_pv)+",\"proof_status\":\""
                +(normal.terminal?"terminal_outcome":"finite_horizon_neural_estimate")+"\",\"nodes\":"+std::to_string(c.nodes)
                +",\"deposits\":"+std::to_string(c.deposits)+",\"elapsed_ms\":"+std::to_string(elapsed)
                +",\"tt_allocated_bytes\":"+std::to_string(search.table_bytes())+",\"nn_evaluations\":"+std::to_string(search.evaluations)
                +",\"nn_cache_hits\":"+std::to_string(search.eval_hits)+",\"tt_hits\":"+std::to_string(search.tt_hits)
                +",\"pvs_probes\":"+std::to_string(search.pvs_probes)+",\"pvs_researches\":"+std::to_string(search.researches)+"}";
            if(progress&&!progress(report.c_str(),user)) { c.stop_reason="callback_failed"; throw cc::Stopped{}; }
            last=report; completed=depth;
        }
        c.stop_reason="max_depth";
    } catch(const cc::Stopped&) { if(c.stop_reason.empty())c.stop_reason="cancelled"; }
      catch(const std::bad_alloc&) { c.stop_reason="memory_budget"; }
    const auto elapsed=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-c.start).count();
    return "{\"ok\":true,\"stop_reason\":"+cc::escape_json(c.stop_reason)+",\"last_completed_depth\":"+std::to_string(completed)
        +",\"nodes\":"+std::to_string(c.nodes)+",\"elapsed_ms\":"+std::to_string(elapsed)+",\"report\":"+last+"}";
}
}
CC_API void* cc_nn_create_model(const float* weights,int count,int width,const char* id) {
    if(!weights||!id||width<16||width>256||count!=width*width+18*width+1||std::strlen(id)!=64)return nullptr;
    try {
        for(int i=0;i<count;++i)if(!std::isfinite(weights[i]))return nullptr;
        auto m=std::make_unique<nn::Model>(); m->width=width; m->id=id; m->weights.assign(weights,weights+count); return m.release();
    } catch(...) { return nullptr; }
}
CC_API void cc_nn_destroy_model(void* model) { delete static_cast<nn::Model*>(model); }
CC_API const char* cc_nn_evaluate(const int* cells,int actor,void* opaque) {
    thread_local std::string out;
    try {
        if(!opaque)throw cc::Error("invalid_model","Model is required");
        const auto p=cc::decode(cells,7,actor,98); const auto& m=*static_cast<nn::Model*>(opaque);
        const double raw=m.raw(p); char buffer[128]; std::snprintf(buffer,sizeof(buffer),"%.17g",raw);
        const int units=cc::margin(p)*nn::scale+static_cast<int>(std::round(m.residual(p)*nn::scale));
        out="{\"ok\":true,\"value_player0\":"+std::string(buffer)+",\"value_units\":"+std::to_string(units)+",\"score_scale\":256}";
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

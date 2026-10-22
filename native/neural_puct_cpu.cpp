// Structured single-pass boundary. Frozen feature extraction and dense math are reused.
#include "neural_relay.cpp"
#include <cstddef>

namespace puct_cpu {
constexpr std::uint32_t version=1;
struct Result {
    std::uint32_t abi_version=version,struct_size=sizeof(Result);
    std::int32_t status=0,terminal=0;
    std::uint64_t feature_simulations=0,feature_deposits=0,neural_evaluations=0;
    double value_player0=0;
    double policy_logits[7]{};
    char reason[32]{},message[192]{};
};
struct Features {
    std::uint32_t abi_version=version,struct_size=sizeof(Features);
    std::int32_t status=0,reserved=0;
    std::uint64_t feature_simulations=0,feature_deposits=0;
    double x[127]{};
    char reason[32]{},message[192]{};
};
template<class T> void failure(T& out,int status,const char* reason,const char* message) {
    out.status=status;
    std::snprintf(out.reason,sizeof(out.reason),"%s",reason);
    std::snprintf(out.message,sizeof(out.message),"%s",message);
}
struct Guard {
    cc::Control& control;
    explicit Guard(cc::Control& value):control(value) {
        bool expected=false;
        if(!control.search_running.compare_exchange_strong(expected,true))
            throw cc::Error("control_busy","Inference control is already running");
    }
    ~Guard(){control.search_running.store(false);}
};
}

CC_API const char* cc_puct_cpu_abi() {return "congklak-puct-cpu-structured-v1";}
CC_API std::uint32_t cc_puct_result_size() {return sizeof(puct_cpu::Result);}
CC_API std::uint32_t cc_puct_features_size() {return sizeof(puct_cpu::Features);}
CC_API std::uint32_t cc_puct_layout(int field) {
    switch(field) {
        case 0:return offsetof(puct_cpu::Result,feature_simulations);
        case 1:return offsetof(puct_cpu::Result,value_player0);
        case 2:return offsetof(puct_cpu::Result,policy_logits);
        case 3:return offsetof(puct_cpu::Result,reason);
        case 4:return offsetof(puct_cpu::Features,x);
        default:return 0xffffffffu;
    }
}
// A fractional-ms shift lets the unchanged millisecond poll stop at the given
// remaining duration. A zero duration means explicitly unlimited, not expired.
CC_API int cc_puct_begin(void* token,std::uint64_t remaining_ns) {
    if(!token)return 0;
    auto& c=*static_cast<cc::Control*>(token);
    if(c.search_running.load()||remaining_ns>43200000000000ull)return 0;
    const auto ms=(remaining_ns+999999)/1000000;
    c.start=std::chrono::steady_clock::now()+std::chrono::nanoseconds(remaining_ns)-std::chrono::milliseconds(ms);
    c.limits.time_ms=ms;c.cancelled.store(false);c.stop_reason.clear();
    c.nodes=0;c.deposits=0;c.memory_checked=false;c.last_memory_work=0;c.peak_process_bytes=0;
    return 1;
}
CC_API int cc_puct_predict(const int* cells,int actor,void* opaque,void* token,puct_cpu::Result* output) {
    if(!output)return 2;
    *output={};
    try {
        if(!opaque||!token)throw cc::Error("invalid_model","Model and control are required");
        auto& c=*static_cast<cc::Control*>(token);puct_cpu::Guard guard(c);c.poll();
        const auto p=cc::decode(cells,7,actor,98);const auto& model=*static_cast<nn::Model*>(opaque);
        if(model.width!=64||model.input_count!=127)throw cc::Error("invalid_model","Expected width64/input127");
        output->terminal=cc::terminal(p)?1:0;
        const auto prediction=model.predict(p,&c,&output->feature_simulations,&output->feature_deposits);
        output->neural_evaluations=output->terminal?0:1;
        output->value_player0=cc::margin(p)+prediction.residual;
        std::copy(prediction.logits.begin(),prediction.logits.end(),output->policy_logits);
        // A forward completed before this poll is still charged if it is late.
        c.poll();return 0;
    } catch(const cc::Stopped&) {
        const auto& c=*static_cast<cc::Control*>(token);
        puct_cpu::failure(*output,1,c.stop_reason.c_str(),"Inference stopped; partial work retained");
    } catch(const cc::Error& e) {puct_cpu::failure(*output,2,e.code.c_str(),e.what());}
      catch(const std::exception& e) {puct_cpu::failure(*output,2,"native_error",e.what());}
      catch(...) {puct_cpu::failure(*output,2,"native_error","Structured inference failed");}
    return output->status;
}
// Separate diagnostic operation: production prediction never invokes it.
CC_API int cc_puct_features(const int* cells,int actor,void* token,puct_cpu::Features* output) {
    if(!output)return 2;
    *output={};
    try {
        if(!token)throw cc::Error("invalid_control","Control is required");
        auto& c=*static_cast<cc::Control*>(token);puct_cpu::Guard guard(c);c.poll();
        const auto p=cc::decode(cells,7,actor,98);
        const auto x=nn::feature_vector(p,127,&c,&output->feature_simulations,&output->feature_deposits);
        std::copy(x.begin(),x.end(),output->x);c.poll();return 0;
    } catch(const cc::Stopped&) {
        const auto& c=*static_cast<cc::Control*>(token);
        puct_cpu::failure(*output,1,c.stop_reason.c_str(),"Diagnostic feature work stopped");
    } catch(const cc::Error& e) {puct_cpu::failure(*output,2,e.code.c_str(),e.what());}
      catch(...) {puct_cpu::failure(*output,2,"native_error","Diagnostic features failed");}
    return output->status;
}

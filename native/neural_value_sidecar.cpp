// Versioned value-only extension. The frozen base source and policy kernel remain unchanged.
#include "neural_puct_cpu.cpp"

namespace sidecar {
struct Model {
    nn::Model* base;
    std::array<double,2065> weights{};
    explicit Model(nn::Model* model):base(model){}
};
struct Result: puct_cpu::Result {
    std::uint64_t branch_calls=0,branch_bypasses=0;
    double base_ms=0,branch_ms=0;
    std::int32_t value_units=0,reserved=0;
};
nn::Prediction predict(const Model& model,const cc::Position& input,cc::Control* control,Result& output) {
    const auto p=cc::normalize(input).state;
    const int active=cc::total(p)-p.cells[7]-p.cells[15];
    auto tick=std::chrono::steady_clock::now();
    if(active<25||active>74) {
        ++output.branch_bypasses;
        const auto value=model.base->predict(p,control,&output.feature_simulations,&output.feature_deposits);
        output.base_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tick).count();
        return value;
    }
    // Single exact feature extraction; no branch-specific sowing work.
    const auto x=nn::feature_vector(p,127,control,&output.feature_simulations,&output.feature_deposits);
    const auto& base=*model.base;
    std::array<double,256> a{},b{};
    const auto dense=base.avx2?nn::dense_avx2:nn::dense_scalar;
    dense(base.first.data(),base.weights.data()+static_cast<std::size_t>(64)*127,x.data(),127,64,a.data());
    std::size_t offset=static_cast<std::size_t>(64)*128;
    dense(base.second.data(),base.weights.data()+offset+static_cast<std::size_t>(64)*64,a.data(),64,64,b.data());
    offset+=static_cast<std::size_t>(64)*65;
    double v=0;
    for(int i=0;i<64;++i)v+=base.weights[offset+i]*b[i];
    const double control_logit=v+base.weights[offset+64];
    offset+=65;
    nn::Prediction out;
    // Preserve the original loop and addition order exactly.
    for(int action=0;action<7;++action) {
        double logit=0;
        for(int i=0;i<64;++i)logit+=base.weights[offset+static_cast<std::size_t>(action)*64+i]*b[i];
        out.logits[action]=logit+base.weights[offset+static_cast<std::size_t>(7)*64+action];
    }
    output.base_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tick).count();
    tick=std::chrono::steady_clock::now();
    double correction=0;
    for(int i=0;i<16;++i) {
        double h=0;
        for(int j=0;j<127;++j)h+=model.weights[static_cast<std::size_t>(i)*127+j]*x[j];
        h=std::max(0.0,h+model.weights[2032+i]);
        correction+=model.weights[2048+i]*h;
    }
    correction+=model.weights[2064];
    out.residual=(p.controller==0?active:-active)*std::tanh(control_logit+correction);
    ++output.branch_calls;
    output.branch_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tick).count();
    return out;
}
}
CC_API const char* cc_sidecar_abi(){return "congklak-value-sidecar-native-v1";}
CC_API std::uint32_t cc_sidecar_result_size(){return sizeof(sidecar::Result);}
CC_API std::uint32_t cc_sidecar_layout(int field){
    switch(field){case 0:return offsetof(sidecar::Result,branch_calls);case 1:return offsetof(sidecar::Result,branch_bypasses);case 2:return offsetof(sidecar::Result,base_ms);case 3:return offsetof(sidecar::Result,branch_ms);case 4:return offsetof(sidecar::Result,value_units);default:return 0xffffffffu;}
}
CC_API void* cc_sidecar_create(void* base,const float* weights,int count){
    if(!base||!weights||count!=2065)return nullptr;
    auto* frozen=static_cast<nn::Model*>(base);
    if(frozen->width!=64||frozen->input_count!=127)return nullptr;
    auto m=std::make_unique<sidecar::Model>(frozen);
    for(int i=0;i<count;++i){if(!std::isfinite(weights[i]))return nullptr;m->weights[i]=weights[i];}
    return m.release();
}
CC_API void cc_sidecar_destroy(void* model){delete static_cast<sidecar::Model*>(model);}
CC_API int cc_sidecar_predict(const int* cells,int actor,void* opaque,void* token,sidecar::Result* output){
    if(!output)return 2;
    *output={};output->struct_size=sizeof(sidecar::Result);
    try {
        if(!opaque||!token)throw cc::Error("invalid_model","Sidecar and control required");
        auto& c=*static_cast<cc::Control*>(token);puct_cpu::Guard guard(c);c.poll();
        const auto p=cc::decode(cells,7,actor,98);
        output->terminal=cc::terminal(p)?1:0;
        const auto prediction=sidecar::predict(*static_cast<sidecar::Model*>(opaque),p,&c,*output);
        output->neural_evaluations=output->terminal?0:1;
        output->value_player0=cc::margin(p)+prediction.residual;
        output->value_units=cc::margin(p)*nn::scale+static_cast<std::int32_t>(std::round(prediction.residual*nn::scale));
        std::copy(prediction.logits.begin(),prediction.logits.end(),output->policy_logits);
        c.poll();return 0;
    }catch(const cc::Stopped&){const auto& c=*static_cast<cc::Control*>(token);puct_cpu::failure(*output,1,c.stop_reason.c_str(),"Sidecar inference stopped; partial work retained");}
     catch(const cc::Error& e){puct_cpu::failure(*output,2,e.code.c_str(),e.what());}
     catch(const std::exception& e){puct_cpu::failure(*output,2,"native_error",e.what());}
     catch(...){puct_cpu::failure(*output,2,"native_error","Sidecar inference failed");}
    return output->status;
}

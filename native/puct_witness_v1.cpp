#include <cstdint>
#include <cstring>
#include <limits>
#define API extern "C" __declspec(dllexport)
API const char* cc_puct_witness_abi(){return "immutable-packed24/puct-order-witness-v1";}
API int cc_puct_witness(const unsigned char* raw,std::uint64_t count,std::uint64_t* out,unsigned char* states){
    if(!raw||!out||!states||count<1||count>2000000)return 0;
    // out: events, model_done, sidecar_done, expand, last_done,
    // last_expand, preceding_puct_node_at_last_done, puct_nodes.
    std::memset(out,0,8*sizeof(std::uint64_t));std::memset(states,0,34);
    const auto missing=std::numeric_limits<std::uint64_t>::max();out[4]=out[5]=out[6]=missing;
    std::uint64_t node=missing;unsigned char node_state[17]{};
    for(std::uint64_t i=0;i<count;++i){
        const auto* row=raw+i*24;unsigned total=0;for(unsigned j=0;j<16;++j)total+=row[j];
        std::int16_t action=0,depth=0;std::memcpy(&action,row+20,2);std::memcpy(&depth,row+22,2);
        const auto op=row[17];if(total!=98||row[16]>1||op<1||op>37||row[18]<1||row[18]>4||row[19]||action < -1||depth < -1)return 0;
        if(op==28){node=i;std::memcpy(node_state,row,17);++out[7];}
        if(op==30){++out[3];out[5]=i;}
        if(op==35||op==36){++out[op==35?1:2];out[4]=i;out[6]=node;std::memcpy(states,row,17);std::memcpy(states+17,node_state,17);}
    }
    out[0]=count;return 1;
}

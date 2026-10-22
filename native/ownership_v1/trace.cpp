#include "trace.hpp"
#include <algorithm>
#include <cstring>
#include <limits>

namespace owntrace {
struct Buffer {
    bool enabled=false, overflow=false, invalid=false;
    unsigned char scope=0;
    std::uint64_t capacity=0, attempted=0;
    std::array<std::uint64_t,64> counts{};
    std::vector<Event> events;
};
thread_local Buffer buffer;
std::atomic<std::uint64_t> controls_created{0},controls_destroyed{0};
std::uint64_t new_control_generation(){return ++controls_created;}
void destroyed_control(){++controls_destroyed;}
unsigned char scope(){return buffer.scope;}
void set_scope(unsigned char value){buffer.scope=value;}
void record(const cc::Position& p, Op op, int action, int depth) {
    auto& b=buffer;
    if(!b.enabled)return;
    ++b.attempted;
    const auto k=static_cast<unsigned>(op);
    if(k>=b.counts.size()||p.pits!=7||p.controller<0||p.controller>1||
       action < -1||action>32767||depth < -1||depth>32767) {
        b.invalid=true;throw cc::Error("ownership_trace_invalid","Trace accepts complete seven-pit states only");
    }
    unsigned total=0;for(auto n:p.cells)total+=n;
    if(total!=98){b.invalid=true;throw cc::Error("ownership_trace_invalid","Seeds in hand or non98 state cannot enter ownership trace");}
    if(b.events.size()>=b.capacity){b.overflow=true;throw cc::Error("ownership_trace_overflow","Ownership event capacity exhausted: HOLD");}
    Event row{};std::copy(p.cells.begin(),p.cells.end(),row.cells);
    row.actor=static_cast<unsigned char>(p.controller);row.operation=static_cast<unsigned char>(op);
    row.scope=b.scope;row.action=static_cast<std::int16_t>(action);row.depth=static_cast<std::int16_t>(depth);
    b.events.push_back(row);++b.counts[k];
}
}
CC_API const char* cc_ownership_trace_abi(){return "congklak-complete-state-ownership-trace-v1/packed24";}
CC_API int cc_ownership_trace_begin(std::uint64_t capacity,unsigned char engine_scope){
    auto& b=owntrace::buffer;
    if(b.enabled||capacity<1||capacity>2000000||engine_scope<1||engine_scope>4)return 0;
    try {b.events.clear();b.events.reserve(static_cast<std::size_t>(capacity));}
    catch(...){return 0;}
    b.capacity=capacity;b.scope=engine_scope;b.attempted=0;b.counts.fill(0);
    b.overflow=false;b.invalid=false;b.enabled=true;return 1;
}
CC_API void cc_ownership_trace_end(){owntrace::buffer.enabled=false;}
CC_API std::uint64_t cc_ownership_trace_count(){return owntrace::buffer.events.size();}
CC_API std::uint64_t cc_ownership_trace_attempted(){return owntrace::buffer.attempted;}
CC_API int cc_ownership_trace_overflow(){return owntrace::buffer.overflow?1:0;}
CC_API int cc_ownership_trace_invalid(){return owntrace::buffer.invalid?1:0;}
CC_API std::uint64_t cc_ownership_trace_op_count(int k){return k>=0&&k<64?owntrace::buffer.counts[k]:0;}
CC_API const owntrace::Event* cc_ownership_trace_data(){return owntrace::buffer.events.data();}
CC_API int cc_ownership_trace_mark(const int* cells,int actor,unsigned char op,int action,int depth){
    try {const auto p=cc::decode(cells,7,actor,98);owntrace::record(p,static_cast<owntrace::Op>(op),action,depth);return 1;}
    catch(...){return 0;}
}
CC_API std::uint64_t cc_ownership_controls_created(){return owntrace::controls_created.load();}
CC_API std::uint64_t cc_ownership_controls_destroyed(){return owntrace::controls_destroyed.load();}
CC_API std::uint64_t cc_ownership_control_serial(void* pointer){return pointer?static_cast<cc::Control*>(pointer)->ownership_generation:0;}

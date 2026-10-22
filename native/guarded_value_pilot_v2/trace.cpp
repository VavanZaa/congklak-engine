#define NOMINMAX
#include "trace.hpp"
#include <windows.h>
#include <bcrypt.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <sstream>

namespace fast {
using Key=std::array<unsigned char,15>;
using Digest=std::array<unsigned char,32>;
constexpr std::uint64_t max_claims=8000000;
bool canonical(const Key& key){
    unsigned a=0;for(int i=0;i<14;++i)a+=key[i];
    if(a<25||a>98||key[14]>1)return false;
    Key other{};for(int i=0;i<7;++i){other[i]=key[i+7];other[i+7]=key[i];}other[14]=1-key[14];
    return key<=other;
}
Key quotient(const owntrace::Event& row){
    Key a{},b{};for(int i=0;i<7;++i){a[i]=row.cells[i];a[i+7]=row.cells[i+8];b[i]=row.cells[i+8];b[i+7]=row.cells[i];}
    a[14]=row.actor;b[14]=1-row.actor;return std::min(a,b);
}
std::uint64_t hash(const Key& key){
    std::uint64_t a=0,b=0;std::memcpy(&a,key.data(),8);std::memcpy(&b,key.data()+8,7);
    auto h=(a^((b<<29)|(b>>35)))*0x9e3779b185ebca87ull;h^=h>>31;h*=0xc2b2ae3d27d4eb4full;return h^(h>>29);
}
Digest sha(const void* data,std::size_t bytes){
    BCRYPT_ALG_HANDLE alg=nullptr;BCRYPT_HASH_HANDLE h=nullptr;DWORD object_size=0,received=0;
    if(BCryptOpenAlgorithmProvider(&alg,BCRYPT_SHA256_ALGORITHM,nullptr,0)<0)throw cc::Error("trace_hash","SHA256 provider unavailable");
    struct AlgClose{BCRYPT_ALG_HANDLE h;~AlgClose(){BCryptCloseAlgorithmProvider(h,0);}} close_alg{alg};
    if(BCryptGetProperty(alg,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&object_size),sizeof(object_size),&received,0)<0)
        throw cc::Error("trace_hash","SHA256 object size unavailable");
    std::vector<unsigned char> object(object_size);Digest out{};
    if(BCryptCreateHash(alg,&h,object.data(),object_size,nullptr,0,0)<0)throw cc::Error("trace_hash","SHA256 allocation failed");
    struct HashClose{BCRYPT_HASH_HANDLE h;~HashClose(){BCryptDestroyHash(h);}} close_hash{h};
    if(bytes>std::numeric_limits<ULONG>::max()||(bytes&&BCryptHashData(h,const_cast<PUCHAR>(static_cast<const unsigned char*>(data)),static_cast<ULONG>(bytes),0)<0)||
       BCryptFinishHash(h,out.data(),static_cast<ULONG>(out.size()),0)<0)throw cc::Error("trace_hash","SHA256 execution failed");
    return out;
}
template<class T> std::string hex(const T& value){
    static const char alphabet[]="0123456789abcdef";std::string out;out.reserve(value.size()*2);
    for(auto c:value){out+=alphabet[c>>4];out+=alphabet[c&15];}return out;
}
#pragma pack(push,1)
struct Owned {Key key;std::uint64_t first_event;unsigned char cells[16];unsigned char actor;};
struct Violation {std::uint64_t owned_index;std::uint32_t flags,other_owner;};
struct Record19 {Key key;std::uint32_t value;};
#pragma pack(pop)
static_assert(sizeof(Owned)==40&&sizeof(Violation)==16&&sizeof(Record19)==19);
struct Unique {Owned row{};std::uint64_t occurrences=0;};
struct UniqueSlot {Key key{};std::uint32_t index=0,epoch=0;};
struct ClaimSlot {Key key{};std::uint32_t owner=0;std::uint64_t stage=0;bool used=false;};
struct Stage {bool published=false;std::uint32_t owner=0;std::uint64_t previous=0;};
struct Guard {
    std::vector<Record19> statics;
    std::vector<ClaimSlot> claims;
    std::vector<Stage> stages;
    std::uint32_t owner_count=0;
    std::uint64_t generation=0,active_stage=0,total_claims=0,new_count=0,next_total=0;
    Digest static_hash{},claims_hash{},delta_hash{},next_hash{};
    explicit Guard(std::uint32_t owners):owner_count(owners),claims(16777216),stages(1){stages[0].published=true;claims_hash=sha(nullptr,0);}
    bool visible(const ClaimSlot& slot)const{return slot.used&&slot.stage<stages.size()&&stages[slot.stage].published;}
    std::pair<std::size_t,bool> find(const Key& key,bool include_active=false)const{
        const auto mask=claims.size()-1;std::size_t free=claims.size();auto i=static_cast<std::size_t>(hash(key))&mask;
        for(std::size_t n=0;n<claims.size();++n,i=(i+1)&mask){
            const auto& slot=claims[i];
            if(!slot.used)return {free<claims.size()?free:i,false};
            const bool staged=include_active&&active_stage&&slot.stage==active_stage;
            if(slot.key==key&&(visible(slot)||staged))return {i,true};
            if(!visible(slot)&&!staged&&free==claims.size())free=i;
        }
        return {free,false};
    }
    std::uint32_t static_flags(const Key& key)const{
        auto i=std::lower_bound(statics.begin(),statics.end(),key,[](const Record19& a,const Key& b){return a.key<b;});
        return i!=statics.end()&&i->key==key?i->value:0;
    }
};
struct SinkHeader {char magic[8];std::uint64_t capacity;volatile LONG64 count;volatile LONG64 flags;unsigned char reserved[32];};
static_assert(sizeof(SinkHeader)==64);
}

namespace owntrace {
struct Buffer {
    bool enabled=false,overflow=false,invalid=false,sealed=false,unique_overflow=false;
    unsigned char scope=0;
    std::uint64_t capacity=0,attempted=0,unique_capacity=0;
    std::array<std::uint64_t,64> counts{};
    std::vector<Event> events;
    std::vector<fast::UniqueSlot> slots;
    std::vector<fast::Unique> unique;
    std::vector<fast::Owned> owned;
    std::vector<fast::Violation> violations;
    std::uint32_t epoch=0;
    HANDLE sink_file=INVALID_HANDLE_VALUE,sink_mapping=nullptr;
    fast::SinkHeader* sink=nullptr;
};
thread_local Buffer buffer;
std::atomic<std::uint64_t> controls_created{0},controls_destroyed{0};
std::uint64_t new_control_generation(){return ++controls_created;}
void destroyed_control(){++controls_destroyed;}
unsigned char scope(){return buffer.scope;}
void set_scope(unsigned char value){buffer.scope=value;}
void record(const cc::Position& p,Op op,int action,int depth){
    auto& b=buffer;if(!b.enabled)return;++b.attempted;const auto k=static_cast<unsigned>(op);
    if(b.sealed||k<1||k>37||p.pits!=7||p.controller<0||p.controller>1||b.scope<1||b.scope>4||
       action < -1||action>32767||depth < -1||depth>32767){b.invalid=true;throw cc::Error("ownership_trace_invalid","Invalid complete-state trace metadata");}
    unsigned total=0;for(auto n:p.cells)total+=n;
    if(total!=98){b.invalid=true;throw cc::Error("ownership_trace_invalid","Seeds in hand/non98 state cannot enter trace");}
    if(b.events.size()>=b.capacity){b.overflow=true;if(b.sink)InterlockedOr64(&b.sink->flags,1);throw cc::Error("ownership_trace_overflow","Complete trace capacity exhausted: HOLD");}
    Event row{};std::copy(p.cells.begin(),p.cells.end(),row.cells);row.actor=static_cast<unsigned char>(p.controller);
    row.operation=static_cast<unsigned char>(op);row.scope=b.scope;row.action=static_cast<std::int16_t>(action);row.depth=static_cast<std::int16_t>(depth);
    const auto index=b.events.size();b.events.push_back(row);++b.counts[k];
    if(b.sink){auto* data=reinterpret_cast<unsigned char*>(b.sink)+64;std::memcpy(data+index*sizeof(Event),&row,sizeof(row));MemoryBarrier();InterlockedExchange64(&b.sink->count,static_cast<LONG64>(b.events.size()));}
    if(98-static_cast<int>(row.cells[7])-static_cast<int>(row.cells[15])>=25){
        auto key=fast::quotient(row);std::size_t i=static_cast<std::size_t>(fast::hash(key))&(b.slots.size()-1);
        while(b.slots[i].epoch==b.epoch&&b.slots[i].key!=key)i=(i+1)&(b.slots.size()-1);
        auto& slot=b.slots[i];
        if(slot.epoch!=b.epoch){
            if(b.unique.size()>=b.unique_capacity){b.unique_overflow=true;b.overflow=true;if(b.sink)InterlockedOr64(&b.sink->flags,8);throw cc::Error("ownership_unique_capacity","Unique owned-state capacity exhausted: HOLD");}
            slot.key=key;slot.epoch=b.epoch;slot.index=static_cast<std::uint32_t>(b.unique.size());fast::Unique u;u.row.key=key;u.row.first_event=index;std::copy(std::begin(row.cells),std::end(row.cells),u.row.cells);u.row.actor=row.actor;b.unique.push_back(u);}
        ++b.unique[slot.index].occurrences;
    }
}
}

CC_API const char* cc_ownership_trace_abi(){return "congklak-complete-state-ownership-trace-v1/packed24";}
CC_API int cc_ownership_trace_begin(std::uint64_t capacity,unsigned char engine_scope){
    auto& b=owntrace::buffer;if(b.enabled||b.sink||capacity<1||capacity>16000000||engine_scope<1||engine_scope>4)return 0;
    try{
        b.unique_capacity=std::min<std::uint64_t>(capacity,4000000);
        b.events.clear();b.events.reserve(static_cast<std::size_t>(capacity));b.unique.clear();b.unique.reserve(static_cast<std::size_t>(b.unique_capacity));b.owned.clear();b.violations.clear();
        std::size_t slots=2;while(slots<b.unique_capacity*2)slots*=2;if(b.slots.size()!=slots){b.slots.clear();b.slots.resize(slots);b.epoch=0;}
        if(++b.epoch==0){for(auto& s:b.slots)s.epoch=0;++b.epoch;}
    }catch(...){return 0;}
    b.capacity=capacity;b.scope=engine_scope;b.attempted=0;b.counts.fill(0);b.overflow=false;b.invalid=false;b.unique_overflow=false;b.sealed=false;b.enabled=true;return 1;
}
CC_API void cc_ownership_trace_end(){owntrace::buffer.enabled=false;}
CC_API std::uint64_t cc_ownership_trace_count(){return owntrace::buffer.events.size();}
CC_API std::uint64_t cc_ownership_trace_attempted(){return owntrace::buffer.attempted;}
CC_API int cc_ownership_trace_overflow(){return owntrace::buffer.overflow?1:0;}
CC_API int cc_ownership_trace_invalid(){return owntrace::buffer.invalid?1:0;}
CC_API std::uint64_t cc_ownership_trace_op_count(int k){return k>=0&&k<64?owntrace::buffer.counts[k]:0;}
CC_API const owntrace::Event* cc_ownership_trace_data(){return owntrace::buffer.events.data();}
CC_API int cc_ownership_trace_mark(const int* cells,int actor,unsigned char op,int action,int depth){
    try{const auto p=cc::decode(cells,7,actor,98);owntrace::record(p,static_cast<owntrace::Op>(op),action,depth);return 1;}catch(...){return 0;}
}
CC_API std::uint64_t cc_ownership_controls_created(){return owntrace::controls_created.load();}
CC_API std::uint64_t cc_ownership_controls_destroyed(){return owntrace::controls_destroyed.load();}
CC_API std::uint64_t cc_ownership_control_serial(void* p){return p?static_cast<cc::Control*>(p)->ownership_generation:0;}

CC_API void* cc_fast_guard_create(const unsigned char* records,std::uint64_t count,std::uint32_t owners){
    try{
        if((count&&!records)||count>16000000||owners<1)return nullptr;
        auto result=std::make_unique<fast::Guard>(owners);result->statics.resize(static_cast<std::size_t>(count));
        if(count)std::memcpy(result->statics.data(),records,static_cast<std::size_t>(count)*19);
        for(std::size_t i=0;i<count;++i)if(!fast::canonical(result->statics[i].key)||result->statics[i].value>3||(i&&result->statics[i-1].key>=result->statics[i].key))return nullptr;
        result->static_hash=fast::sha(records,static_cast<std::size_t>(count)*19);return result.release();
    }catch(...){return nullptr;}
}
CC_API void cc_fast_guard_destroy(void* p){delete static_cast<fast::Guard*>(p);}
namespace {
int claims(void* p,const unsigned char* bytes,std::uint64_t count,bool restore){
    if(!p||(count&&!bytes)||count>fast::max_claims)return 0;auto& g=*static_cast<fast::Guard*>(p);if(g.active_stage)return 0;
    std::vector<fast::Record19> records(static_cast<std::size_t>(count));if(count)std::memcpy(records.data(),bytes,static_cast<std::size_t>(count)*19);
    for(std::size_t i=0;i<count;++i)if(!fast::canonical(records[i].key)||records[i].value<1||records[i].value>g.owner_count||(i&&records[i-1].key>=records[i].key))return 0;
    if(!restore){std::uint64_t new_count=0;for(const auto& row:records){auto f=g.find(row.key);if(f.second&&g.claims[f.first].owner!=row.value)return 0;if(!f.second)++new_count;}if(g.total_claims+new_count>fast::max_claims)return 0;}
    if(restore){for(auto& slot:g.claims)slot.used=false;g.stages.clear();g.stages.resize(1);g.stages[0].published=true;g.generation=0;g.total_claims=0;}
    for(const auto& row:records){auto f=g.find(row.key);if(f.first==g.claims.size())return 0;if(!f.second)++g.total_claims;auto& slot=g.claims[f.first];slot.key=row.key;slot.owner=row.value;slot.stage=0;slot.used=true;}
    if(restore)g.claims_hash=fast::sha(bytes,static_cast<std::size_t>(count)*19);
    else{
        std::vector<fast::Record19> all;all.reserve(static_cast<std::size_t>(g.total_claims));for(const auto& slot:g.claims)if(g.visible(slot))all.push_back({slot.key,slot.owner});
        std::sort(all.begin(),all.end(),[](const auto& a,const auto& b){return a.key<b.key;});g.claims_hash=fast::sha(all.data(),all.size()*19);
    }return 1;
}
}
CC_API int cc_fast_guard_restore(void* p,const unsigned char* b,std::uint64_t n){try{return claims(p,b,n,true);}catch(...){return 0;}}
CC_API int cc_fast_guard_update(void* p,const unsigned char* b,std::uint64_t n){try{return claims(p,b,n,false);}catch(...){return 0;}}
CC_API int cc_fast_guard_restore_generation(void* p,std::uint64_t generation){
    try{if(!p||generation>1000000)return 0;auto& g=*static_cast<fast::Guard*>(p);
        if(g.active_stage||g.generation||g.stages.size()!=1)return 0;
        g.stages.resize(static_cast<std::size_t>(generation+1));g.stages[0].published=true;g.generation=generation;return 1;
    }catch(...){return 0;}
}
CC_API int cc_fast_guard_publish_stage(void* p,std::uint64_t generation,std::uint32_t owner){
    if(!p)return 0;auto& g=*static_cast<fast::Guard*>(p);
    if(!generation||g.active_stage!=generation||generation>=g.stages.size()||g.stages[generation].owner!=owner||g.stages[generation].previous!=g.generation)return 0;
    g.stages[generation].published=true;g.generation=generation;g.total_claims=g.next_total;g.claims_hash=g.next_hash;g.active_stage=0;return 1;
}
CC_API void cc_fast_guard_discard_stage(void* p){if(p)static_cast<fast::Guard*>(p)->active_stage=0;}
CC_API const char* cc_fast_guard_status(void* p){
    thread_local std::string result;
    if(!p){result="{\"ok\":false}";return result.c_str();}const auto& g=*static_cast<fast::Guard*>(p);
    result="{\"ok\":true,\"generation\":"+std::to_string(g.generation)+",\"active_stage\":"+std::to_string(g.active_stage)+",\"claims_count\":"+std::to_string(g.total_claims)+",\"claims_hash\":\""+fast::hex(g.claims_hash)+"\",\"static_hash\":\""+fast::hex(g.static_hash)+"\"}";return result.c_str();
}
CC_API const char* cc_fast_guard_stage_certificate(void* p){
    thread_local std::string result;if(!p){result="{\"valid\":false}";return result.c_str();}const auto& g=*static_cast<fast::Guard*>(p);
    const auto owner=g.active_stage?g.stages[g.active_stage].owner:0;
    result="{\"valid\":"+std::string(g.active_stage?"true":"false")+",\"generation\":"+std::to_string(g.active_stage)+",\"owner\":"+std::to_string(owner)
      +",\"previous_generation\":"+std::to_string(g.generation)+",\"previous_claim_hash\":\""+fast::hex(g.claims_hash)+"\",\"stage_delta_hash\":\""+fast::hex(g.delta_hash)
      +"\",\"staged_claim_hash\":\""+fast::hex(g.next_hash)+"\",\"stage_new_count\":"+std::to_string(g.new_count)+",\"staged_total_claims\":"+std::to_string(g.next_total)
      +",\"guard_records_sha256\":\""+fast::hex(g.static_hash)+"\"}";return result.c_str();
}
CC_API const char* cc_fast_finalize(void* guard,std::uint32_t owner){
    thread_local std::string result;auto& b=owntrace::buffer;bool valid=false,guard_passed=false;std::string error;
    std::array<std::uint64_t,64> counts{};std::uint32_t scopes=0;std::uint64_t high=0,played_in=std::numeric_limits<std::uint64_t>::max(),played_out=played_in;
    fast::Digest raw_hash{},owned_hash{},owned_keys_hash{},violations_hash{};fast::Guard* g=static_cast<fast::Guard*>(guard);
    b.sealed=true;if(b.sink)InterlockedOr64(&b.sink->flags,2);
    try{
        // Preserve the complete captured prefix footprint even when the declared
        // attempt overflowed.  Such a certificate stays unusable, never empty.
        b.owned.clear();for(const auto& row:b.unique)b.owned.push_back(row.row);
        std::sort(b.owned.begin(),b.owned.end(),[](const auto& a,const auto& c){return a.key<c.key;});
        if(b.events.empty()||b.overflow||b.invalid||b.attempted!=b.events.size())throw cc::Error("trace_incomplete","Empty/truncated or invalid trace: HOLD");
        for(std::size_t i=0;i<b.events.size();++i){
            const auto& row=b.events[i];unsigned total=0;for(auto n:row.cells)total+=n;
            if(total!=98||row.actor>1||row.operation<1||row.operation>37||row.scope<1||row.scope>4||row.reserved||row.action < -1||row.depth < -1)
                throw cc::Error("trace_invalid","Complete-state native validation failed");
            ++counts[row.operation];scopes|=1u<<row.scope;
            if(row.operation==owntrace::played_input)played_in=i;if(row.operation==owntrace::played_output)played_out=i;
            if(98-static_cast<int>(row.cells[7])-static_cast<int>(row.cells[15])>=25){
                ++high;auto key=fast::quotient(row);auto slot=static_cast<std::size_t>(fast::hash(key))&(b.slots.size()-1);
                while(b.slots[slot].epoch==b.epoch&&b.slots[slot].key!=key)slot=(slot+1)&(b.slots.size()-1);
                if(b.slots[slot].epoch!=b.epoch||b.unique[b.slots[slot].index].row.first_event>i)throw cc::Error("trace_footprint","High-state native coverage missing");
            }
        }
        if(counts!=b.counts)throw cc::Error("trace_partition","Native event count partition differs");
        std::uint64_t occurrences=0;for(const auto& row:b.unique)occurrences+=row.occurrences;
        if(occurrences!=high)throw cc::Error("trace_footprint","Native high-state occurrence partition differs");
        valid=true;
        if(!g||owner<1||owner>g->owner_count||g->active_stage)throw cc::Error("guard_identity","Guard/owner missing or previous staged claims not published");
        b.violations.clear();
        for(std::size_t i=0;i<b.owned.size();++i){const auto& row=b.owned[i];auto flags=g->static_flags(row.key);auto live=g->find(row.key);std::uint32_t other=0;
            if(live.second&&g->claims[live.first].owner!=owner){flags|=4;other=g->claims[live.first].owner;}
            if(flags)b.violations.push_back({i,flags,other});}
        guard_passed=b.violations.empty();
        if(guard_passed){
            const auto token=static_cast<std::uint64_t>(g->stages.size());if(token>1000000)throw cc::Error("guard_capacity","Stage descriptor capacity exhausted");
            g->stages.push_back({false,owner,g->generation});g->active_stage=token;
            std::vector<fast::Record19> delta;delta.reserve(b.owned.size());
            for(const auto& row:b.owned){auto live=g->find(row.key,true);if(live.second)continue;
                if(live.first==g->claims.size()||g->total_claims+delta.size()>=fast::max_claims)throw cc::Error("guard_capacity","Claim table capacity exhausted");
                auto& slot=g->claims[live.first];slot.key=row.key;slot.owner=owner;slot.stage=token;slot.used=true;delta.push_back({row.key,owner});}
            g->new_count=delta.size();g->next_total=g->total_claims+delta.size();g->delta_hash=fast::sha(delta.data(),delta.size()*19);
            std::vector<unsigned char> chain;chain.insert(chain.end(),g->claims_hash.begin(),g->claims_hash.end());chain.insert(chain.end(),g->delta_hash.begin(),g->delta_hash.end());
            const auto add=[&chain](auto value){const auto* bytes=reinterpret_cast<const unsigned char*>(&value);chain.insert(chain.end(),bytes,bytes+sizeof(value));};
            add(token);add(owner);add(g->new_count);g->next_hash=fast::sha(chain.data(),chain.size());
        }
    }catch(const std::exception& exc){error=exc.what();guard_passed=false;if(g)g->active_stage=0;}
    try{raw_hash=fast::sha(b.events.data(),b.events.size()*24);owned_hash=fast::sha(b.owned.data(),b.owned.size()*40);std::vector<fast::Key> keys;keys.reserve(b.owned.size());for(const auto& row:b.owned)keys.push_back(row.key);owned_keys_hash=fast::sha(keys.data(),keys.size()*15);violations_hash=fast::sha(b.violations.data(),b.violations.size()*16);}
    catch(const std::exception& exc){error=exc.what();valid=false;guard_passed=false;if(g)g->active_stage=0;}
    std::ostringstream out;out<<"{\"schema\":\"native-fast-trace-certificate-v1\",\"valid\":"<<(valid?"true":"false")
      <<",\"guard_passed\":"<<(guard_passed?"true":"false")<<",\"event_count\":"<<b.events.size()<<",\"attempted_events\":"<<b.attempted
      <<",\"raw_event_capacity\":"<<b.capacity<<",\"unique_owned_capacity\":"<<b.unique_capacity<<",\"unique_owned_overflow\":"<<(b.unique_overflow?"true":"false")<<",\"global_claim_capacity\":"<<fast::max_claims
      <<",\"overflow\":"<<(b.overflow?"true":"false")<<",\"invalid\":"<<(b.invalid?"true":"false")<<",\"event_sha256\":\""<<fast::hex(raw_hash)
      <<"\",\"owned_sha256\":\""<<fast::hex(owned_hash)<<"\",\"owned_keys_sha256\":\""<<fast::hex(owned_keys_hash)<<"\",\"violations_sha256\":\""<<fast::hex(violations_hash)<<"\",\"owned_count\":"<<b.owned.size()
      <<",\"owned_occurrences\":"<<high<<",\"violations_count\":"<<b.violations.size()<<",\"scope_mask\":"<<scopes<<",\"played_input_index\":"<<played_in
      <<",\"played_output_index\":"<<played_out<<",\"operation_counts\":[";
    for(std::size_t i=0;i<b.counts.size();++i){if(i)out<<',';out<<b.counts[i];}out<<"],\"owner\":"<<owner;
    if(g)out<<",\"static_hash\":\""<<fast::hex(g->static_hash)<<"\",\"guard_records_sha256\":\""<<fast::hex(g->static_hash)<<"\",\"previous_generation\":"<<g->generation<<",\"previous_claim_hash\":\""<<fast::hex(g->claims_hash)
      <<"\",\"generation\":"<<g->active_stage<<",\"stage_delta_hash\":\""<<fast::hex(g->delta_hash)<<"\",\"staged_claim_hash\":\""<<fast::hex(g->next_hash)
      <<"\",\"stage_new_count\":"<<g->new_count<<",\"staged_total_claims\":"<<g->next_total;
    out<<",\"error\":"<<cc::escape_json(error)<<'}';result=out.str();return result.c_str();
}
CC_API const unsigned char* cc_fast_owned_data(){return reinterpret_cast<const unsigned char*>(owntrace::buffer.owned.data());}
CC_API const unsigned char* cc_fast_violation_data(){return reinterpret_cast<const unsigned char*>(owntrace::buffer.violations.data());}
CC_API int cc_fast_profile_load(const unsigned char* raw,std::uint64_t count){
    if((count&&!raw)||count>16000000||!cc_ownership_trace_begin(std::max<std::uint64_t>(count,1),1))return 0;
    try{for(std::size_t i=0;i<count;++i){owntrace::Event row;std::memcpy(&row,raw+i*24,24);if(row.reserved)throw cc::Error("trace_invalid","Reserved byte differs");
        cc::Position p;std::copy(std::begin(row.cells),std::end(row.cells),p.cells.begin());p.controller=row.actor;owntrace::set_scope(row.scope);
        owntrace::record(p,static_cast<owntrace::Op>(row.operation),row.action,row.depth);}return 1;
    }catch(...){owntrace::buffer.invalid=true;return 0;}
}
// Pure saved-byte capacity fixture. Production trace_begin always restores the
// declared min(raw_capacity,4m) bound; this helper never invokes an engine.
CC_API int cc_epoch_profile_load_limited(const unsigned char* raw,std::uint64_t count,std::uint64_t unique_limit){
    if((count&&!raw)||count>16000000||unique_limit<1||unique_limit>4000000||!cc_ownership_trace_begin(std::max<std::uint64_t>(count,1),1))return 0;
    owntrace::buffer.unique_capacity=std::min(owntrace::buffer.unique_capacity,unique_limit);
    try{for(std::size_t i=0;i<count;++i){owntrace::Event row;std::memcpy(&row,raw+i*24,24);if(row.reserved)throw cc::Error("trace_invalid","Reserved byte differs");
        cc::Position p;std::copy(std::begin(row.cells),std::end(row.cells),p.cells.begin());p.controller=row.actor;owntrace::set_scope(row.scope);
        owntrace::record(p,static_cast<owntrace::Op>(row.operation),row.action,row.depth);}return 1;
    }catch(...){owntrace::buffer.invalid=true;return 0;}
}
CC_API int cc_fast_profile_expected(std::uint64_t attempted,int overflow,int invalid,const std::uint64_t* counts){
    auto& b=owntrace::buffer;if(!b.enabled||b.sealed||(overflow!=0&&overflow!=1)||(invalid!=0&&invalid!=1))return 0;
    b.attempted=attempted;b.overflow=overflow!=0;b.invalid=invalid!=0;if(counts)std::copy(counts,counts+64,b.counts.begin());return 1;
}
CC_API int cc_fast_sink_open(const wchar_t* path){
    auto& b=owntrace::buffer;if(!b.enabled||b.events.size()||b.sink||!path)return 0;
    HANDLE file=CreateFileW(path,GENERIC_READ|GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
    if(file==INVALID_HANDLE_VALUE)return 0;LARGE_INTEGER bytes;bytes.QuadPart=64+static_cast<LONGLONG>(b.capacity)*24;
    if(!SetFilePointerEx(file,bytes,nullptr,FILE_BEGIN)||!SetEndOfFile(file)){CloseHandle(file);return 0;}
    HANDLE map=CreateFileMappingW(file,nullptr,PAGE_READWRITE,0,0,nullptr);if(!map){CloseHandle(file);return 0;}
    auto* view=static_cast<fast::SinkHeader*>(MapViewOfFile(map,FILE_MAP_WRITE,0,0,0));
    if(!view){CloseHandle(map);CloseHandle(file);return 0;}
    std::memset(view,0,sizeof(*view));std::memcpy(view->magic,"OTRACE2",7);view->capacity=b.capacity;
    b.sink_file=file;b.sink_mapping=map;b.sink=view;return 1;
}
CC_API int cc_fast_sink_close(){
    auto& b=owntrace::buffer;if(!b.sink)return 1;const auto count=b.sink->count;bool okay=FlushViewOfFile(b.sink,0)!=0;
    okay=UnmapViewOfFile(b.sink)!=0&&okay;CloseHandle(b.sink_mapping);b.sink=nullptr;b.sink_mapping=nullptr;
    LARGE_INTEGER end;end.QuadPart=64+count*24;okay=SetFilePointerEx(b.sink_file,end,nullptr,FILE_BEGIN)!=0&&okay;
    okay=SetEndOfFile(b.sink_file)!=0&&okay;okay=FlushFileBuffers(b.sink_file)!=0&&okay;CloseHandle(b.sink_file);b.sink_file=INVALID_HANDLE_VALUE;return okay?1:0;
}

// Prospective construction-only component projection; no search/trace hook changes.
namespace {
std::vector<fast::Record19> component_records(const unsigned char* records,std::uint64_t count,std::uint32_t allowed){
    if((count&&!records)||count>16000000)throw cc::Error("guard_component_input","Invalid shared component record count");
    std::vector<fast::Record19> rows(static_cast<std::size_t>(count));
    if(count)std::memcpy(rows.data(),records,static_cast<std::size_t>(count)*19);
    for(std::size_t i=0;i<count;++i){
        if(!fast::canonical(rows[i].key)||!rows[i].value||(i&&rows[i-1].key>=rows[i].key))
            throw cc::Error("guard_component_input","Invalid canonical shared component record");
        rows[i].value=rows[i].value==allowed?0u:2u;
    }
    return rows;
}
}
CC_API int cc_fast_guard_project_hash(const unsigned char* records,std::uint64_t count,std::uint32_t allowed,unsigned char* out){
    try{if(!out)return 0;auto rows=component_records(records,count,allowed);auto digest=fast::sha(rows.data(),rows.size()*19);std::memcpy(out,digest.data(),32);return 1;}catch(...){return 0;}
}
CC_API void* cc_fast_guard_create_component(const unsigned char* records,std::uint64_t count,std::uint32_t owners,std::uint32_t allowed){
    try{if(!owners)return nullptr;auto rows=component_records(records,count,allowed);auto result=std::make_unique<fast::Guard>(owners);result->statics=std::move(rows);result->static_hash=fast::sha(result->statics.data(),result->statics.size()*19);return result.release();}catch(...){return nullptr;}
}

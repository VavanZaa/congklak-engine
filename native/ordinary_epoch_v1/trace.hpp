#pragma once
#include "engine.hpp"
#include <cstddef>

namespace owntrace {
enum Op : unsigned char {
    decode=1, normalize_in, normalize_out, legal, terminal_test, margin_eval,
    move_in, move_out, alpha_node, alpha_tt_probe, alpha_tt_hit, alpha_tt_slot,
    relay_node, relay_tt_probe, relay_tt_hit, relay_tt_slot, relay_eval,
    relay_cache_probe, relay_cache_hit, model_eval, model_forward,
    feature_input, feature_move_in, feature_move_out, sidecar_eval,
    sidecar_branch, sidecar_bypass, puct_node, puct_select, puct_expand,
    played_input, played_output, alpha_pv_node, relay_pv_node, model_done, sidecar_done, alpha_root_node
};
#pragma pack(push,1)
struct Event {
    unsigned char cells[16];
    unsigned char actor, operation, scope, reserved;
    std::int16_t action, depth;
};
#pragma pack(pop)
static_assert(sizeof(Event)==24);
void record(const cc::Position&, Op, int action=-1, int depth=-1);
unsigned char scope();
void set_scope(unsigned char);
std::uint64_t new_control_generation();
void destroyed_control();
struct Scope {
    unsigned char previous;
    explicit Scope(unsigned char value): previous(scope()) { set_scope(value); }
    ~Scope(){set_scope(previous);}
};
}
CC_API const char* cc_ownership_trace_abi();
CC_API int cc_ownership_trace_begin(std::uint64_t capacity, unsigned char engine_scope);
CC_API void cc_ownership_trace_end();
CC_API std::uint64_t cc_ownership_trace_count();
CC_API std::uint64_t cc_ownership_trace_attempted();
CC_API int cc_ownership_trace_overflow();
CC_API int cc_ownership_trace_invalid();
CC_API std::uint64_t cc_ownership_trace_op_count(int operation);
CC_API const owntrace::Event* cc_ownership_trace_data();
CC_API int cc_ownership_trace_mark(const int*,int,unsigned char,int,int);
CC_API std::uint64_t cc_ownership_controls_created();
CC_API std::uint64_t cc_ownership_controls_destroyed();
CC_API std::uint64_t cc_ownership_control_serial(void*);
CC_API void* cc_fast_guard_create(const unsigned char*,std::uint64_t,std::uint32_t);
CC_API void cc_fast_guard_destroy(void*);
CC_API int cc_fast_guard_restore(void*,const unsigned char*,std::uint64_t);
CC_API int cc_fast_guard_update(void*,const unsigned char*,std::uint64_t);
CC_API int cc_fast_guard_publish_stage(void*,std::uint64_t,std::uint32_t);
CC_API void cc_fast_guard_discard_stage(void*);
CC_API const char* cc_fast_guard_status(void*);
CC_API const char* cc_fast_finalize(void*,std::uint32_t);
CC_API const unsigned char* cc_fast_owned_data();
CC_API const unsigned char* cc_fast_violation_data();
CC_API int cc_fast_profile_load(const unsigned char*,std::uint64_t);
CC_API int cc_fast_sink_open(const wchar_t*);
CC_API int cc_fast_sink_close();
CC_API int cc_fast_guard_restore_generation(void*,std::uint64_t);
CC_API int cc_fast_profile_expected(std::uint64_t,int,int,const std::uint64_t*);
CC_API const char* cc_fast_guard_stage_certificate(void*);

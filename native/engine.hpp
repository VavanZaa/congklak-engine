#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define CC_API extern "C" __declspec(dllexport)
#else
#define CC_API extern "C" __attribute__((visibility("default")))
#endif

namespace cc {
inline constexpr const char* ABI_VERSION = "congklak-native-v1";
struct Position {
    std::array<std::uint8_t,16> cells{};
    int pits = 7;
    int controller = 0;
};
struct Error : std::runtime_error {
    std::string code;
    Error(std::string c, std::string m) : std::runtime_error(m), code(std::move(c)) {}
};
struct Stopped : std::exception {
    const char* what() const noexcept override { return "search stopped"; }
};
struct Limits {
    int max_depth = 16;
    std::uint64_t node_budget = 1000000;
    std::uint64_t time_ms = 1000;
    std::uint64_t memory_bytes = 512ull*1024*1024;
};
struct Control {
    Limits limits;
    std::atomic<bool> cancelled{false};
    std::atomic<bool> search_running{false};
    int tt_mb = 0; // additive options: legacy controls retain the baseline
    bool use_pvs = false;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::uint64_t nodes = 0;
    std::uint64_t deposits = 0;
    std::uint64_t last_memory_work = 0;
    std::uint64_t peak_process_bytes = 0;
    bool memory_checked = false;
    std::string stop_reason;
    void poll(bool decision_node = false); // implemented by search.cpp
};
struct Pass { int from; int to; };
struct Event {
    std::string type;
    int actor = -1;
    int slot = -1;
    int opposite = -1;
    int count = 0;
    int before = -1;
    int after = -1;
    int hand = -1;
    int to = -1;
};
struct Normalized {
    Position state;
    int input_controller = 0;
    bool terminal = false;
    std::vector<Pass> passes;
};
struct Transition {
    Position state;
    int actor = 0;
    int pit = 0;
    int last_slot = -1;
    int drops = 0;
    int relays = 0;
    std::array<int,2> store_delta{};
    std::string reason;
    bool terminal = false;
    std::vector<Pass> passes;
    std::vector<Event> events;
};
Position decode(const int* cells, int pits, int actor, int expected_total);
int total(const Position&);
int margin(const Position&);
bool terminal(const Position&);
std::vector<int> legal_actions(const Position&);
Normalized normalize(const Position&);
Transition apply(const Position&, int pit, bool trace, Control* control = nullptr);
std::string escape_json(const std::string&);
std::string position_json(const Position&);
std::string passes_json(const std::vector<Pass>&);
std::string event_json(const Event&, int pits);
std::string normalized_json(const Normalized&);
std::string transition_json(const Transition&);
std::string error_json(const std::string& code, const std::string& message);
}

using cc_progress_fn = int (*)(const char*, void*);
CC_API const char* cc_abi_version();
CC_API const char* cc_validate(const int* cells, int pits, int actor, int expected_total);
CC_API const char* cc_normalize(const int* cells, int pits, int actor, int expected_total);
CC_API const char* cc_apply(const int* cells, int pits, int actor, int expected_total,
                          int action, int trace, void* control);
CC_API void* cc_create_control(int max_depth, std::uint64_t nodes, std::uint64_t time_ms,
                              std::uint64_t memory_bytes);
CC_API void cc_cancel(void* control);
CC_API void cc_destroy_control(void* control);
CC_API const char* cc_set_search_options(void* control, int tt_mb, int use_pvs);
CC_API const char* cc_analyze(const int* cells, int pits, int actor, int expected_total,
                            void* control, int start_depth, cc_progress_fn progress, void* user);

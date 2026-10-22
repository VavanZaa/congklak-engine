#include "engine.hpp"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <new>
#include <memory>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace cc {
namespace {

constexpr int infinity = 1000000000;
constexpr const char* evaluator_id = "H_v1";
constexpr const char* policy_id = "terminal-first/no-oracle/full-root-window-v2";
// H_v1 plus the terminal-first finite-choice-horizon policy.  A table belongs
// to one Search instance and cannot be reused across engines or policies.
constexpr std::uint32_t tt_semantic_id = 0x02010001;

std::uint64_t process_bytes() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    counters.cb = sizeof(counters);
    if (!GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)))
        throw Error("memory_monitor_failed", "Cannot read process working set");
    return static_cast<std::uint64_t>(counters.WorkingSetSize);
#else
    // Linux's resident page count measures current use, rather than a previous
    // request's high-water mark. The fallback is conservative on other hosts.
    if (FILE* file = std::fopen("/proc/self/statm", "r")) {
        unsigned long long pages = 0, resident = 0;
        const int count = std::fscanf(file, "%llu %llu", &pages, &resident);
        std::fclose(file);
        const long page_size = sysconf(_SC_PAGESIZE);
        if (count == 2 && page_size > 0)
            return resident * static_cast<std::uint64_t>(page_size);
    }
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0)
        throw Error("memory_monitor_failed", "Cannot read process resident memory");
#ifdef __APPLE__
    return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
    return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024;
#endif
#endif
}

[[noreturn]] void stop(Control& control, const char* reason) {
    control.stop_reason = reason;
    throw Stopped{};
}

std::string integer_list(const std::vector<int>& values) {
    std::string out = "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) out += ',';
        out += std::to_string(values[i]);
    }
    return out + ']';
}

struct PvChoice {
    int actor;
    int pit;
    int next_controller;
    bool terminal;
    bool bonus;
    std::vector<Pass> passes;
};

PvChoice pv_choice(const Transition& transition) {
    return {transition.actor, transition.pit, transition.state.controller,
            transition.terminal, !transition.terminal && transition.reason == "own_store", transition.passes};
}

std::string pv_json(const std::vector<PvChoice>& pv) {
    std::string out = "[";
    for (std::size_t i = 0; i < pv.size(); ++i) {
        if (i) out += ',';
        const auto& move = pv[i];
        out += "{\"actor\":" + std::to_string(move.actor)
             + ",\"pit\":" + std::to_string(move.pit)
             + ",\"next_controller\":"
             + (move.terminal ? "null" : std::to_string(move.next_controller))
             + ",\"bonus_choice\":" + (move.bonus ? "true" : "false")
             + ",\"forced_passes_after\":" + passes_json(move.passes)
             + ",\"terminal_after\":" + (move.terminal ? "true" : "false") + '}';
    }
    return out + ']';
}

struct Coverage {
    std::uint64_t nodes = 0;
    std::uint64_t transitions = 0;
    std::uint64_t terminal_leaves = 0;
    std::uint64_t horizon_leaves = 0;
    std::uint64_t cutoffs = 0;
    std::uint64_t controller_changes = 0;
    std::uint64_t completed_turns = 0;
    std::uint64_t terminal_transitions = 0;
    std::uint64_t opponent_choices = 0;
    std::uint64_t tt_probes = 0;
    std::uint64_t tt_hits = 0;
    std::uint64_t tt_exact_hits = 0;
    std::uint64_t tt_bound_cutoffs = 0;
    std::uint64_t pvs_probes = 0;
    std::uint64_t pvs_researches = 0;
    std::uint64_t pv_reconstruction_nodes = 0;
    int deepest_choice_depth = 0;
};

void edge_coverage(Coverage& coverage, const Transition& transition, int root_actor) {
    ++coverage.transitions;
    if (transition.actor != root_actor) ++coverage.opponent_choices;
    if (!transition.terminal && transition.state.controller != transition.actor)
        ++coverage.controller_changes;
    if (transition.terminal) ++coverage.terminal_transitions;
    else if (transition.reason != "own_store") ++coverage.completed_turns;
}

void add_coverage(Coverage& target, const Coverage& source) {
    target.nodes += source.nodes;
    target.transitions += source.transitions;
    target.terminal_leaves += source.terminal_leaves;
    target.horizon_leaves += source.horizon_leaves;
    target.cutoffs += source.cutoffs;
    target.controller_changes += source.controller_changes;
    target.completed_turns += source.completed_turns;
    target.terminal_transitions += source.terminal_transitions;
    target.opponent_choices += source.opponent_choices;
    target.tt_probes += source.tt_probes;
    target.tt_hits += source.tt_hits;
    target.tt_exact_hits += source.tt_exact_hits;
    target.tt_bound_cutoffs += source.tt_bound_cutoffs;
    target.pvs_probes += source.pvs_probes;
    target.pvs_researches += source.pvs_researches;
    target.pv_reconstruction_nodes += source.pv_reconstruction_nodes;
    target.deepest_choice_depth = std::max(target.deepest_choice_depth, source.deepest_choice_depth);
}

std::string coverage_json(const Coverage& coverage) {
    return "{\"nodes\":" + std::to_string(coverage.nodes)
        + ",\"transitions\":" + std::to_string(coverage.transitions)
        + ",\"terminal_leaves\":" + std::to_string(coverage.terminal_leaves)
        + ",\"horizon_leaves\":" + std::to_string(coverage.horizon_leaves)
        + ",\"leaves\":" + std::to_string(coverage.terminal_leaves + coverage.horizon_leaves)
        + ",\"alpha_beta_cutoffs\":" + std::to_string(coverage.cutoffs)
        + ",\"controller_changes\":" + std::to_string(coverage.controller_changes)
        + ",\"completed_turns\":" + std::to_string(coverage.completed_turns)
        + ",\"terminal_transitions\":" + std::to_string(coverage.terminal_transitions)
        + ",\"opponent_choices\":" + std::to_string(coverage.opponent_choices)
        + ",\"tt_probes\":" + std::to_string(coverage.tt_probes)
        + ",\"tt_hits\":" + std::to_string(coverage.tt_hits)
        + ",\"tt_exact_hits\":" + std::to_string(coverage.tt_exact_hits)
        + ",\"tt_bound_cutoffs\":" + std::to_string(coverage.tt_bound_cutoffs)
        + ",\"pvs_probes\":" + std::to_string(coverage.pvs_probes)
        + ",\"pvs_researches\":" + std::to_string(coverage.pvs_researches)
        + ",\"pv_reconstruction_nodes\":" + std::to_string(coverage.pv_reconstruction_nodes)
        + ",\"deepest_choice_depth\":" + std::to_string(coverage.deepest_choice_depth)
        + ",\"scope\":\"visited_search_tree\"}";
}

std::string physical_interval_json(const Position& state, int score_player) {
    const int difference = margin(state);
    const int active = total(state) - state.cells[state.pits] - state.cells[2 * state.pits + 1];
    const int lower = difference - active;
    const int upper = difference + active;
    const int display_lower = score_player == 0 ? lower : -upper;
    const int display_upper = score_player == 0 ? upper : -lower;
    return "{\"lower_player0\":" + std::to_string(lower)
         + ",\"upper_player0\":" + std::to_string(upper)
         + ",\"score_player\":" + std::to_string(score_player)
         + ",\"lower\":" + std::to_string(display_lower)
         + ",\"upper\":" + std::to_string(display_upper)
         + ",\"scope\":\"eventual_game_physical_seed_interval\",\"used_for_pruning\":false}";
}

std::string depth_result_json(int value, int actor, int depth, bool terminal_root = false) {
    return "{\"value_player0\":" + std::to_string(value)
        + ",\"value_actor\":" + std::to_string(actor == 0 ? value : -value)
        + ",\"value\":" + std::to_string(actor == 0 ? value : -value)
        + ",\"score_player\":" + std::to_string(actor)
        + ",\"choice_depth\":" + std::to_string(depth)
        + ",\"bound\":\"exact\",\"scope\":\"finite_choice_horizon\""
        + ",\"evaluator\":\"" + evaluator_id + "\",\"policy\":\"" + policy_id + "\""
        + ",\"terminal_root\":" + (terminal_root ? "true" : "false") + '}';
}

struct NodeResult {
    int value = 0;
    std::vector<PvChoice> pv;
    bool ends_terminal = false;
};

enum class Bound : std::uint8_t { empty, exact, lower, upper };

struct TtEntry {
    std::uint64_t hash = 0;
    std::array<std::uint8_t, 16> cells{};
    std::uint32_t semantic = 0;
    std::int16_t value = 0;
    std::uint8_t pits = 0;
    std::uint8_t actor = 0;
    std::uint8_t depth = 0;
    Bound bound = Bound::empty;
    std::int8_t best_pit = -1;
};

std::uint64_t state_hash(const Position& state, int depth) {
    std::uint64_t hash = 14695981039346656037ull;
    const auto mix = [&hash](std::uint8_t value) {
        hash ^= value;
        hash *= 1099511628211ull;
    };
    for (const auto value : state.cells) mix(value);
    mix(static_cast<std::uint8_t>(state.pits));
    mix(static_cast<std::uint8_t>(state.controller));
    mix(static_cast<std::uint8_t>(depth));
    for (int shift = 0; shift < 32; shift += 8)
        mix(static_cast<std::uint8_t>(tt_semantic_id >> shift));
    // Direct mapping consumes low bits, so spread the structured key fully.
    hash ^= hash >> 33;
    hash *= 0xff51afd7ed558ccdull;
    hash ^= hash >> 33;
    hash *= 0xc4ceb9fe1a85ec53ull;
    return hash ^ (hash >> 33);
}

class Search {
public:
    Search(Control& control, int actor) : control_(control), root_actor_(actor) {}

    void initialize() {
        if (!control_.tt_mb) return;
        const std::uint64_t budget = static_cast<std::uint64_t>(control_.tt_mb) * 1024 * 1024;
        const std::uint64_t max_entries = budget / sizeof(TtEntry);
        std::size_t capacity = 1;
        while (capacity <= max_entries / 2) capacity *= 2;
        const std::uint64_t bytes = capacity * sizeof(TtEntry);
        // Admission precedes the allocation; periodically sampled resident
        // memory also includes this table after its pages are initialized.
        const auto resident = process_bytes();
        control_.peak_process_bytes = std::max(control_.peak_process_bytes, resident);
        if (resident >= control_.limits.memory_bytes
            || bytes > control_.limits.memory_bytes - resident)
            stop(control_, "memory_budget");
        table_.resize(capacity);
        table_bytes_ = bytes;
        control_.memory_checked = false;
        control_.poll();
    }

    std::uint64_t table_bytes() const { return table_bytes_; }
    std::size_t table_capacity() const { return table_.size(); }

    NodeResult node(const Position& input, int depth, int alpha, int beta,
                    int ply, Coverage& coverage, bool reconstruction = false) {
        control_.poll(true);
        ++coverage.nodes;
        if (reconstruction) ++coverage.pv_reconstruction_nodes;
        coverage.deepest_choice_depth = std::max(coverage.deepest_choice_depth, ply);
        const auto normalized = normalize(input);
        const auto& state = normalized.state;
        if (normalized.terminal) {
            ++coverage.terminal_leaves;
            return {margin(state), {}, true};
        }
        if (depth == 0) {
            ++coverage.horizon_leaves;
            return {margin(state), {}, false};
        }

        const int original_alpha = alpha, original_beta = beta;
        const std::uint64_t hash = state_hash(state, depth);
        TtEntry* entry = nullptr;
        int hint = -1;
        if (!table_.empty()) {
            ++coverage.tt_probes;
            entry = &table_[hash & (table_.size() - 1)];
            if (entry->bound != Bound::empty && entry->hash == hash
                && entry->semantic == tt_semantic_id && entry->depth == depth
                && entry->pits == state.pits && entry->actor == state.controller
                && entry->cells == state.cells) {
                ++coverage.tt_hits;
                hint = entry->best_pit;
                if (entry->bound == Bound::exact) {
                    ++coverage.tt_exact_hits;
                    return {entry->value, {}, false};
                }
                // Bound entries never masquerade as exact values and never
                // tighten the window.  They only prove an existing cutoff.
                if ((entry->bound == Bound::lower && entry->value >= beta)
                    || (entry->bound == Bound::upper && entry->value <= alpha)) {
                    ++coverage.tt_bound_cutoffs;
                    return {entry->value, {}, false};
                }
            }
        }

        auto actions = legal_actions(state);
        std::sort(actions.begin(), actions.end());
        if (actions.empty())
            throw Error("search_invariant", "Normalized nonterminal position has no legal actions");
        if (hint >= 0) {
            const auto found = std::find(actions.begin(), actions.end(), hint);
            if (found != actions.end()) std::rotate(actions.begin(), found, found + 1);
        }
        const bool maximizing = state.controller == 0;
        int best = maximizing ? -infinity : infinity;
        int best_pit = -1;
        bool first = true;
        for (const int pit : actions) {
            const auto transition = apply(state, pit, false, &control_);
            edge_coverage(coverage, transition, root_actor_);
            int value;
            if (control_.use_pvs && !first && beta > alpha + 1) {
                ++coverage.pvs_probes;
                // Windows remain in fixed player-0 coordinates.  The current
                // controller determines MAX/MIN; bonus moves require no sign
                // flip and forced passes are normalized by the transition.
                const int narrow_alpha = maximizing ? alpha : beta - 1;
                const int narrow_beta = maximizing ? alpha + 1 : beta;
                value = node(transition.state, depth - 1, narrow_alpha, narrow_beta,
                             ply + 1, coverage, reconstruction).value;
                if ((maximizing && value > alpha && value < beta)
                    || (!maximizing && value < beta && value > alpha)) {
                    ++coverage.pvs_researches;
                    value = node(transition.state, depth - 1, alpha, beta,
                                 ply + 1, coverage, reconstruction).value;
                }
            } else {
                value = node(transition.state, depth - 1, alpha, beta,
                             ply + 1, coverage, reconstruction).value;
            }
            first = false;
            if ((maximizing && value > best) || (!maximizing && value < best)) {
                best = value;
                best_pit = pit;
            }
            if (maximizing) alpha = std::max(alpha, best);
            else beta = std::min(beta, best);
            if (alpha >= beta) {
                ++coverage.cutoffs;
                break;
            }
        }
        // Exceptions propagate before this publication.  An interrupted
        // subtree can leave complete child entries, but never its own entry.
        control_.poll();
        if (entry) {
            TtEntry complete;
            complete.hash = hash;
            complete.cells = state.cells;
            complete.semantic = tt_semantic_id;
            complete.value = static_cast<std::int16_t>(best);
            complete.pits = static_cast<std::uint8_t>(state.pits);
            complete.actor = static_cast<std::uint8_t>(state.controller);
            complete.depth = static_cast<std::uint8_t>(depth);
            complete.best_pit = static_cast<std::int8_t>(best_pit);
            complete.bound = best <= original_alpha ? Bound::upper
                : best >= original_beta ? Bound::lower : Bound::exact;
            *entry = complete;
        }
        return {best, {}, false};
    }

    NodeResult canonical_pv(const Position& input, int depth, int target,
                            int ply, Coverage& coverage) {
        NodeResult result;
        result.value = target;
        Position current = input;
        while (true) {
            control_.poll(true);
            ++coverage.nodes;
            ++coverage.pv_reconstruction_nodes;
            coverage.deepest_choice_depth = std::max(coverage.deepest_choice_depth, ply);
            const auto normalized = normalize(current);
            current = normalized.state;
            if (normalized.terminal || depth == 0) {
                if (margin(current) != target)
                    throw Error("pv_invariant", "Canonical continuation does not reach its exact horizon value");
                result.ends_terminal = normalized.terminal;
                if (normalized.terminal) ++coverage.terminal_leaves;
                else ++coverage.horizon_leaves;
                return result;
            }
            auto actions = legal_actions(current);
            std::sort(actions.begin(), actions.end());
            bool found = false;
            for (const int pit : actions) {
                const auto transition = apply(current, pit, false, &control_);
                edge_coverage(coverage, transition, root_actor_);
                const int value = node(transition.state, depth - 1, -infinity, infinity,
                                       ply + 1, coverage, true).value;
                if (value == target) {
                    result.pv.push_back(pv_choice(transition));
                    current = transition.state;
                    --depth;
                    ++ply;
                    found = true;
                    break;
                }
            }
            if (!found)
                throw Error("pv_invariant", "No legal continuation matches the exact minimax value");
        }
    }

private:
    Control& control_;
    int root_actor_;
    std::vector<TtEntry> table_;
    std::uint64_t table_bytes_ = 0;
};

struct RootCandidate {
    int actor = 0;
    int pit = 0;
    Transition transition;
    bool transition_complete = false;
    NodeResult result;
    Coverage coverage;
};

std::string candidate_json(const RootCandidate& candidate, int depth, bool scored) {
    std::string out = "{\"actor\":" + std::to_string(candidate.actor)
        + ",\"pit\":" + std::to_string(candidate.pit)
        + ",\"depth_result\":"
        + (scored ? depth_result_json(candidate.result.value, candidate.actor, depth) : "null")
        + ",\"analysis_complete\":" + (scored ? "true" : "false")
        + ",\"transition\":"
        + (candidate.transition_complete ? transition_json(candidate.transition) : "null");
    if (candidate.transition_complete)
        out += ",\"physical_interval\":" + physical_interval_json(candidate.transition.state, candidate.actor);
    out += ",\"pv\":" + (scored ? pv_json(candidate.result.pv) : "[]")
        + ",\"pv_truncated\":" + ((scored && candidate.result.ends_terminal) ? "false" : "true")
        + ",\"coverage\":" + coverage_json(candidate.coverage) + '}';
    return out;
}

std::string candidates_json(const std::vector<RootCandidate>& candidates, int depth, bool scored) {
    std::string out = "[";
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (i) out += ',';
        out += candidate_json(candidates[i], depth, scored);
    }
    return out + ']';
}

std::string root_metadata_json(const Normalized& root, const std::vector<int>& actions) {
    const int actor = root.terminal ? root.input_controller : root.state.controller;
    return "\"input_controller\":" + std::to_string(root.input_controller)
        + ",\"normalized_controller\":" + (root.terminal ? "null" : std::to_string(root.state.controller))
        + ",\"score_player\":" + std::to_string(actor)
        + ",\"forced_passes\":" + passes_json(root.passes)
        + ",\"terminal\":" + (root.terminal ? "true" : "false")
        + ",\"position\":" + position_json(root.state)
        + ",\"legal_actions\":" + integer_list(actions)
        + ",\"physical_interval\":" + physical_interval_json(root.state, actor)
        + ",\"evaluator\":\"" + evaluator_id + "\",\"policy\":\"" + policy_id + "\""
        + ",\"engine_version\":\"" + ABI_VERSION + "\"";
}

std::string report_json(const Normalized& root, const std::vector<int>& actions,
                        const std::vector<RootCandidate>& candidates, int depth,
                        const NodeResult& best, const std::vector<int>& best_moves,
                        const Coverage& coverage, const Control& control,
                        const Search& search) {
    const int actor = root.terminal ? root.input_controller : root.state.controller;
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - control.start).count();
    return "{\"completed_iteration\":true,\"choice_depth\":" + std::to_string(depth)
        + ',' + root_metadata_json(root, actions)
        + ",\"depth_result\":" + depth_result_json(best.value, actor, depth, root.terminal)
        + ",\"best_moves\":" + integer_list(best_moves)
        + ",\"candidates\":" + candidates_json(candidates, depth, depth > 0 && !root.terminal)
        + ",\"pv\":" + pv_json(best.pv)
        + ",\"pv_truncated\":" + (best.ends_terminal ? "false" : "true")
        + ",\"proof_status\":\"" + (root.terminal ? "terminal_outcome" : "finite_horizon_result") + "\""
        + ",\"coverage\":" + coverage_json(coverage)
        + ",\"search_options\":{\"tt_mb\":" + std::to_string(control.tt_mb)
        + ",\"use_pvs\":" + (control.use_pvs ? "true" : "false")
        + ",\"tt_allocated_bytes\":" + std::to_string(search.table_bytes())
        + ",\"tt_capacity\":" + std::to_string(search.table_capacity())
        + ",\"tt_key\":\"full-state/normalized-controller/exact-remaining-depth/H_v1/policy-v2\""
        + ",\"pv_policy\":\"lexicographic-optimal/full-window-reconstruction\"}"
        + ",\"nodes\":" + std::to_string(control.nodes)
        + ",\"deposits\":" + std::to_string(control.deposits)
        + ",\"elapsed_ms\":" + std::to_string(elapsed)
        + ",\"peak_process_bytes\":" + std::to_string(control.peak_process_bytes) + '}';
}

std::string analyze(const Position& position, Control& control, int start_depth,
                    cc_progress_fn progress, void* user) {
    if (start_depth < 0 || start_depth > 128)
        throw Error("invalid_start_depth", "start_depth must be in 0..128");
    control.start = std::chrono::steady_clock::now();
    control.nodes = 0;
    control.deposits = 0;
    control.stop_reason.clear();
    control.last_memory_work = 0;
    control.peak_process_bytes = 0;
    control.memory_checked = false;

    const auto root = normalize(position);
    auto actions = legal_actions(root.state);
    std::sort(actions.begin(), actions.end());
    if (root.terminal) actions.clear();
    const int actor = root.terminal ? root.input_controller : root.state.controller;
    std::vector<RootCandidate> candidates;
    candidates.reserve(actions.size());
    for (int pit : actions) {
        RootCandidate candidate;
        candidate.actor = actor;
        candidate.pit = pit;
        candidates.push_back(std::move(candidate));
    }
    std::string last_report;
    int last_depth = -1;
    Search search(control, actor);
    try {
        control.poll();
        search.initialize();
        // Root evidence is immutable and shared across iterations. An aborted
        // sow never supplies a partial successor or a partial trace.
        for (auto& candidate : candidates) {
            candidate.transition = apply(root.state, candidate.pit, true, &control);
            candidate.transition_complete = true;
        }

        int first_depth = root.terminal || control.limits.max_depth == 0 ? 0 : std::max(1, start_depth);
        const int last_requested_depth = root.terminal ? 0 : control.limits.max_depth;
        for (int depth = first_depth; depth <= last_requested_depth; ++depth) {
            Coverage coverage;
            NodeResult best;
            std::vector<int> best_moves;
            control.poll(true);
            coverage.nodes = 1;
            if (root.terminal || depth == 0) {
                best = {margin(root.state), {}, root.terminal};
                if (root.terminal) ++coverage.terminal_leaves;
                else ++coverage.horizon_leaves;
            } else {
                best.value = actor == 0 ? -infinity : infinity;
                for (auto& candidate : candidates) {
                    candidate.coverage = {};
                    edge_coverage(candidate.coverage, candidate.transition, actor);
                    // Every displayed alternative receives its own full
                    // window; no previous root score narrows this search.
                    candidate.result = search.node(candidate.transition.state, depth - 1,
                                                   -infinity, infinity, 1, candidate.coverage);
                    candidate.result = search.canonical_pv(candidate.transition.state, depth - 1,
                                                          candidate.result.value, 1, candidate.coverage);
                    candidate.result.pv.insert(candidate.result.pv.begin(), pv_choice(candidate.transition));
                    add_coverage(coverage, candidate.coverage);
                    const int value = candidate.result.value;
                    const bool improves = actor == 0 ? value > best.value : value < best.value;
                    if (improves) {
                        best = candidate.result;
                        best_moves.assign(1, candidate.pit);
                    } else if (value == best.value) {
                        best_moves.push_back(candidate.pit);
                    }
                }
            }
            control.poll();
            std::string complete_report = report_json(root, actions, candidates, depth,
                                                       best, best_moves, coverage, control, search);
            if (progress && !progress(complete_report.c_str(), user))
                stop(control, "callback_failed");
            // Callback acceptance is the publication boundary. Until it
            // succeeds, last_report still names the previous durable result.
            last_report = std::move(complete_report);
            last_depth = depth;
        }
        control.stop_reason = "max_depth";
    } catch (const Stopped&) {
        if (control.stop_reason.empty()) control.stop_reason = "cancelled";
    } catch (const std::bad_alloc&) {
        control.stop_reason = "memory_budget";
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - control.start).count();
    return "{\"ok\":true,\"stop_reason\":" + escape_json(control.stop_reason)
        + ",\"last_completed_depth\":" + std::to_string(last_depth)
        + ",\"nodes\":" + std::to_string(control.nodes)
        + ",\"deposits\":" + std::to_string(control.deposits)
        + ",\"elapsed_ms\":" + std::to_string(elapsed)
        + ",\"peak_process_bytes\":" + std::to_string(control.peak_process_bytes)
        + ",\"initial\":{" + root_metadata_json(root, actions)
        + ",\"candidates\":" + candidates_json(candidates, 0, false) + "}"
        + ",\"report\":" + (last_report.empty() ? "null" : last_report) + '}';
}

} // namespace

void Control::poll(bool decision_node) {
    if (cancelled.load(std::memory_order_relaxed)) stop(*this, "cancelled");
    const auto elapsed = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count());
    if (limits.time_ms && elapsed >= limits.time_ms)
        stop(*this, "time_budget");
    const std::uint64_t work = nodes + deposits;
    if (!memory_checked || work - last_memory_work >= 1024) {
        std::uint64_t resident = 0;
        try {
            resident = process_bytes();
        } catch (const Error&) {
            stop(*this, "memory_monitor_failed");
        }
        memory_checked = true;
        last_memory_work = work;
        peak_process_bytes = std::max(peak_process_bytes, resident);
        if (limits.memory_bytes && resident >= limits.memory_bytes)
            stop(*this, "memory_budget");
    }
    if (decision_node) {
        if (limits.node_budget && nodes >= limits.node_budget)
            stop(*this, "node_budget");
        ++nodes;
    }
}

} // namespace cc

CC_API void* cc_create_control(int max_depth, std::uint64_t nodes, std::uint64_t time_ms,
                               std::uint64_t memory_bytes) {
    constexpr std::uint64_t mib = 1024ull * 1024;
    if (max_depth < 0 || max_depth > 128 || memory_bytes < 16 * mib || memory_bytes > 24576 * mib)
        return nullptr;
    try {
        auto* control = new cc::Control;
        control->limits = {max_depth, nodes, time_ms, memory_bytes};
        return control;
    } catch (...) {
        return nullptr;
    }
}

CC_API void cc_cancel(void* opaque_control) {
    if (opaque_control)
        static_cast<cc::Control*>(opaque_control)->cancelled.store(true, std::memory_order_relaxed);
}

CC_API void cc_destroy_control(void* opaque_control) {
    delete static_cast<cc::Control*>(opaque_control);
}

CC_API const char* cc_set_search_options(void* opaque_control, int tt_mb, int use_pvs) {
    thread_local std::string result;
    try {
        if (!opaque_control)
            throw cc::Error("invalid_control", "search options require a valid control token");
        auto& control = *static_cast<cc::Control*>(opaque_control);
        constexpr std::uint64_t mib = 1024ull * 1024;
        if (tt_mb < 0 || tt_mb > 1024 || (use_pvs != 0 && use_pvs != 1)
            || static_cast<std::uint64_t>(tt_mb) * mib + 16 * mib > control.limits.memory_bytes)
            throw cc::Error("invalid_search_options", "TT must be 0..1024 MiB, leave 16 MiB of process memory, and PVS must be 0 or 1");
        bool expected = false;
        if (!control.search_running.compare_exchange_strong(expected, true))
            throw cc::Error("control_busy", "Cannot modify search options while the control is running");
        control.tt_mb = tt_mb;
        control.use_pvs = use_pvs != 0;
        control.search_running.store(false);
        result = "{\"ok\":true,\"tt_mb\":" + std::to_string(tt_mb)
            + ",\"use_pvs\":" + (use_pvs ? "true" : "false") + '}';
    } catch (const cc::Error& error) {
        result = cc::error_json(error.code, error.what());
    } catch (...) {
        result = cc::error_json("internal_error", "Cannot set native search options");
    }
    return result.c_str();
}

CC_API const char* cc_analyze(const int* cells, int pits, int actor, int expected_total,
                             void* opaque_control, int start_depth,
                             cc_progress_fn progress, void* user) {
    thread_local std::string result;
    try {
        if (!opaque_control)
            throw cc::Error("invalid_control", "analyze requires a valid control token");
        const auto position = cc::decode(cells, pits, actor, expected_total);
        auto& control = *static_cast<cc::Control*>(opaque_control);
        bool expected = false;
        if (!control.search_running.compare_exchange_strong(expected, true))
            throw cc::Error("control_busy", "Control already has a running search");
        struct RunningGuard {
            cc::Control& control;
            ~RunningGuard() { control.search_running.store(false); }
        } guard{control};
        result = cc::analyze(position, control, start_depth, progress, user);
    } catch (const cc::Error& error) {
        result = cc::error_json(error.code, error.what());
    } catch (const std::exception& error) {
        result = cc::error_json("internal_error", error.what());
    } catch (...) {
        result = cc::error_json("internal_error", "Unknown native exception");
    }
    return result.c_str();
}

#include "engine.hpp"

#include <exception>
#include <limits>
#include <sstream>
#include <utility>

namespace cc {
namespace {

int ring_size(const Position& s) { return 2 * (s.pits + 1); }
int row_start(const Position& s, int actor) { return actor * (s.pits + 1); }
int store_slot(const Position& s, int actor) { return row_start(s, actor) + s.pits; }

void check_shape(const Position& s) {
    if (s.pits < 1 || s.pits > 7)
        throw Error("invalid_pit_count", "The native test kernel supports one through seven pits per side.");
    if (s.controller != 0 && s.controller != 1)
        throw Error("invalid_controller", "The controller must be player 0 or player 1.");
    int sum = 0;
    for (int i = 0; i < 16; ++i) {
        if (i >= ring_size(s) && s.cells[i] != 0)
            throw Error("unsupported_state", "Unused ring slots must be zero.");
        sum += s.cells[i];
    }
    if (sum > 255)
        throw Error("unsupported_seed_total", "The native stable-state encoding supports at most 255 total seeds.");
}

bool row_empty(const Position& s, int actor) {
    const int first = row_start(s, actor);
    for (int i = 0; i < s.pits; ++i)
        if (s.cells[first + i] != 0) return false;
    return true;
}

void record(std::vector<Event>& events, bool trace, const char* type, int actor,
            int slot = -1, int count = 0, int before = -1, int after = -1,
            int hand = -1, int opposite = -1, int to = -1) {
    if (!trace) return;
    Event e;
    e.type = type;
    e.actor = actor;
    e.slot = slot;
    e.count = count;
    e.before = before;
    e.after = after;
    e.hand = hand;
    e.opposite = opposite;
    e.to = to;
    events.push_back(std::move(e));
}

std::string integers_json(const std::vector<int>& values) {
    std::string result = "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) result += ',';
        result += std::to_string(values[i]);
    }
    result += ']';
    return result;
}

} // namespace

Position decode(const int* cells, int pits, int actor, int expected_total) {
    if (cells == nullptr) throw Error("invalid_position", "A 16-slot count array is required.");
    if (pits < 1 || pits > 7)
        throw Error("invalid_pit_count", "The native test kernel supports one through seven pits per side.");
    if (actor != 0 && actor != 1)
        throw Error("invalid_controller", "The controller must be player 0 or player 1.");
    if (expected_total < -1 || expected_total > 255)
        throw Error("unsupported_seed_total", "The expected seed total must be -1 (infer) or zero through 255.");

    Position s;
    s.pits = pits;
    s.controller = actor;
    int sum = 0;
    const int size = ring_size(s);
    for (int i = 0; i < 16; ++i) {
        if (cells[i] < 0 || cells[i] > 255)
            throw Error("invalid_seed_count", "Every stable count must be an integer from zero through 255.");
        if (i >= size && cells[i] != 0)
            throw Error("unsupported_state", "Unused ring slots must be zero.");
        sum += cells[i];
        s.cells[i] = static_cast<std::uint8_t>(cells[i]);
    }
    if (sum > 255)
        throw Error("unsupported_seed_total", "The native stable-state encoding supports at most 255 total seeds.");
    if (expected_total >= 0 && sum != expected_total)
        throw Error("seed_total_mismatch", "Stable counts do not conserve the declared seed total.");
    return s;
}

int total(const Position& s) {
    check_shape(s);
    int sum = 0;
    for (int i = 0; i < ring_size(s); ++i) sum += s.cells[i];
    return sum;
}

int margin(const Position& s) {
    check_shape(s);
    return static_cast<int>(s.cells[store_slot(s, 0)]) - s.cells[store_slot(s, 1)];
}

bool terminal(const Position& s) {
    check_shape(s);
    return row_empty(s, 0) && row_empty(s, 1);
}

std::vector<int> legal_actions(const Position& s) {
    check_shape(s);
    std::vector<int> actions;
    const int first = row_start(s, s.controller);
    for (int i = 0; i < s.pits; ++i)
        if (s.cells[first + i] != 0) actions.push_back(i);
    return actions;
}

Normalized normalize(const Position& s) {
    check_shape(s);
    Normalized n;
    n.state = s;
    n.input_controller = s.controller;
    n.terminal = row_empty(s, 0) && row_empty(s, 1);
    if (!n.terminal && row_empty(s, s.controller)) {
        n.state.controller = 1 - s.controller;
        n.passes.push_back(Pass{s.controller, n.state.controller});
    }
    return n;
}

Transition apply(const Position& input, int pit, bool trace, Control* control) {
    check_shape(input);
    if (row_empty(input, 0) && row_empty(input, 1))
        throw Error("terminal_position", "A terminal position has no legal pit choice.");
    if (row_empty(input, input.controller))
        throw Error("forced_pass_required", "Normalize the forced pass before choosing a pit for the other player.");
    if (pit < 0 || pit >= input.pits)
        throw Error("invalid_action", "The action must name a local pit on the controller's row.");
    const int actor = input.controller;
    const int source = row_start(input, actor) + pit;
    if (input.cells[source] == 0)
        throw Error("empty_pit", "The selected pit is empty.");
    if (control != nullptr) control->poll(false);

    // Publish only the final scratch state. An exception from polling discards it.
    Transition t;
    t.state = input;
    t.actor = actor;
    t.pit = pit;
    const int size = ring_size(input);
    const int own_store = store_slot(input, actor);
    const int skipped_store = store_slot(input, 1 - actor);
    const int initial_store0 = input.cells[store_slot(input, 0)];
    const int initial_store1 = input.cells[store_slot(input, 1)];

    int cursor = source;
    int hand = t.state.cells[source];
    t.state.cells[source] = 0;
    record(t.events, trace, "pickup", actor, source, hand, hand, 0, hand);

    for (;;) {
        int last_before = 0;
        while (hand != 0) {
            cursor = (cursor + 1) % size;
            if (cursor == skipped_store) cursor = (cursor + 1) % size;
            last_before = t.state.cells[cursor];
            t.state.cells[cursor] = static_cast<std::uint8_t>(last_before + 1);
            --hand;
            ++t.drops;
            if (control != nullptr) {
                ++control->deposits;
                if ((t.drops & 63) == 0) control->poll(false);
            }
            record(t.events, trace, "deposit", actor, cursor, 1,
                   last_before, last_before + 1, hand);
        }

        t.last_slot = cursor;
        if (cursor == own_store) {
            t.reason = "own_store";
            break;
        }
        if (last_before != 0) {
            hand = t.state.cells[cursor];
            t.state.cells[cursor] = 0;
            ++t.relays;
            record(t.events, trace, "relay", actor, cursor, hand, hand, 0, hand);
            continue;
        }

        const int own_first = row_start(input, actor);
        if (cursor >= own_first && cursor < own_first + input.pits) {
            const int local = cursor - own_first;
            const int opposite = row_start(input, 1 - actor) + input.pits - 1 - local;
            if (t.state.cells[opposite] != 0) {
                const int captured = t.state.cells[cursor] + t.state.cells[opposite];
                const int store_before = t.state.cells[own_store];
                t.state.cells[cursor] = 0;
                t.state.cells[opposite] = 0;
                t.state.cells[own_store] = static_cast<std::uint8_t>(store_before + captured);
                record(t.events, trace, "capture", actor, cursor, captured,
                       store_before, store_before + captured, 0, opposite);
                t.reason = "own_capture";
            } else {
                t.reason = "own_empty";
            }
        } else {
            t.reason = "opponent_empty";
        }
        break;
    }

    t.terminal = row_empty(t.state, 0) && row_empty(t.state, 1);
    if (t.terminal) {
        record(t.events, trace, "terminal", actor, -1, 0,
               t.state.cells[store_slot(input, 0)], t.state.cells[store_slot(input, 1)]);
    } else {
        const int nominal = t.reason == "own_store" ? actor : 1 - actor;
        t.state.controller = nominal;
        record(t.events, trace, t.reason == "own_store" ? "bonus" : "turn_end", actor, cursor);
        const Normalized n = normalize(t.state);
        t.state = n.state;
        t.passes = n.passes;
        for (const Pass& pass : t.passes)
            record(t.events, trace, "pass", pass.from, -1, 0, -1, -1, -1, -1, pass.to);
    }
    t.store_delta = {static_cast<int>(t.state.cells[store_slot(input, 0)]) - initial_store0,
                     static_cast<int>(t.state.cells[store_slot(input, 1)]) - initial_store1};
    if (control != nullptr) control->poll(false);
    return t;
}

std::string escape_json(const std::string& value) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string result = "\"";
    for (unsigned char c : value) {
        switch (c) {
        case '"': result += "\\\""; break;
        case '\\': result += "\\\\"; break;
        case '\b': result += "\\b"; break;
        case '\f': result += "\\f"; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default:
            if (c < 0x20) {
                result += "\\u00";
                result += hex[c >> 4];
                result += hex[c & 15];
            } else result += static_cast<char>(c);
        }
    }
    result += '"';
    return result;
}

std::string position_json(const Position& s) {
    check_shape(s);
    std::string result = "{\"pits\":[";
    for (int actor = 0; actor < 2; ++actor) {
        if (actor != 0) result += ',';
        result += '[';
        for (int i = 0; i < s.pits; ++i) {
            if (i != 0) result += ',';
            result += std::to_string(s.cells[row_start(s, actor) + i]);
        }
        result += ']';
    }
    result += "],\"stores\":[" + std::to_string(s.cells[store_slot(s, 0)]) + ','
        + std::to_string(s.cells[store_slot(s, 1)]) + "],\"controller\":"
        + std::to_string(s.controller) + '}';
    return result;
}

std::string passes_json(const std::vector<Pass>& passes) {
    std::string result = "[";
    for (std::size_t i = 0; i < passes.size(); ++i) {
        if (i != 0) result += ',';
        result += "{\"from\":" + std::to_string(passes[i].from)
            + ",\"to\":" + std::to_string(passes[i].to) + '}';
    }
    result += ']';
    return result;
}

std::string event_json(const Event& e, int pits) {
    (void)pits;
    std::string result = "{\"type\":" + escape_json(e.type);
    const auto optional = [&result](const char* key, int value) {
        if (value >= 0) result += ",\"" + std::string(key) + "\":" + std::to_string(value);
    };
    optional("actor", e.actor);
    optional("slot", e.slot);
    optional("opposite", e.opposite);
    result += ",\"count\":" + std::to_string(e.count);
    optional("before", e.before);
    optional("after", e.after);
    optional("hand", e.hand);
    optional("to", e.to);
    result += '}';
    return result;
}

std::string normalized_json(const Normalized& n) {
    return "{\"ok\":true,\"position\":" + position_json(n.state)
        + ",\"input_controller\":" + std::to_string(n.input_controller)
        + ",\"normalized_controller\":" + std::to_string(n.state.controller)
        + ",\"terminal\":" + (n.terminal ? "true" : "false")
        + ",\"legal_actions\":" + integers_json(legal_actions(n.state))
        + ",\"passes\":" + passes_json(n.passes)
        + ",\"margin_player0\":" + std::to_string(margin(n.state)) + '}';
}

std::string transition_json(const Transition& t) {
    std::string result = "{\"ok\":true,\"position\":" + position_json(t.state)
        + ",\"actor\":" + std::to_string(t.actor)
        + ",\"pit\":" + std::to_string(t.pit)
        + ",\"reason\":" + escape_json(t.reason)
        + ",\"last_slot\":" + std::to_string(t.last_slot)
        + ",\"drops\":" + std::to_string(t.drops)
        + ",\"relays\":" + std::to_string(t.relays)
        + ",\"store_delta\":[" + std::to_string(t.store_delta[0]) + ','
        + std::to_string(t.store_delta[1]) + "]"
        + ",\"terminal\":" + (t.terminal ? "true" : "false")
        + ",\"passes\":" + passes_json(t.passes) + ",\"events\":[";
    for (std::size_t i = 0; i < t.events.size(); ++i) {
        if (i != 0) result += ',';
        result += event_json(t.events[i], t.state.pits);
    }
    result += "]}";
    return result;
}

std::string error_json(const std::string& code, const std::string& message) {
    return "{\"ok\":false,\"error\":{\"code\":" + escape_json(code)
        + ",\"message\":" + escape_json(message) + "}}";
}

} // namespace cc

namespace {
thread_local std::string rules_result;

template <typename Operation>
const char* export_result(Operation operation, cc::Control* control = nullptr) noexcept {
    try {
        rules_result = operation();
    } catch (const cc::Error& e) {
        rules_result = cc::error_json(e.code, e.what());
    } catch (const cc::Stopped&) {
        rules_result = cc::error_json("cancelled", control && !control->stop_reason.empty()
            ? control->stop_reason : "The operation stopped before publishing a successor.");
        if (control != nullptr) {
            // Counters establish whether cancellation occurred during sowing;
            // they do not expose a partially applied position or event stream.
            rules_result.pop_back();
            rules_result += ",\"deposits\":" + std::to_string(control->deposits)
                + ",\"nodes\":" + std::to_string(control->nodes) + '}';
        }
    } catch (const std::exception& e) {
        rules_result = cc::error_json("native_error", e.what());
    } catch (...) {
        rules_result = cc::error_json("native_error", "Unknown native exception.");
    }
    return rules_result.c_str();
}
} // namespace

CC_API const char* cc_abi_version() { return cc::ABI_VERSION; }

CC_API const char* cc_validate(const int* cells, int pits, int actor, int expected_total) {
    return export_result([=] {
        return cc::normalized_json(cc::normalize(cc::decode(cells, pits, actor, expected_total)));
    });
}

CC_API const char* cc_normalize(const int* cells, int pits, int actor, int expected_total) {
    return cc_validate(cells, pits, actor, expected_total);
}

CC_API const char* cc_apply(const int* cells, int pits, int actor, int expected_total,
                          int action, int trace, void* control) {
    cc::Control* native_control = static_cast<cc::Control*>(control);
    return export_result([=] {
        return cc::transition_json(cc::apply(cc::decode(cells, pits, actor, expected_total),
                                             action, trace != 0, native_control));
    }, native_control);
}

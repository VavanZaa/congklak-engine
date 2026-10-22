# Literal forensics of saved capture-position lines

This helper investigates the weighted H10-to-H12 **teacher-Q drift** on the four previously consumed capture roots. It does not measure a newly trained model decline. Models, policy, search settings, sampling/loss settings and previous allocation HOLD remain fixed. No new label, search, game, neural forward, Native runtime call or training is permitted.

`congklak/capture_forensics.py` imports only the standard library and loads `tests/reference.py`, the independent Python seed-by-seed interpreter. It calls its literal normalization, legal-action, sowing and emitted-event reconstruction functions. It never calls minimax, Native, neural evaluation or a producer. Source identity includes the interpreter's exact file hash.

Public APIs:

```text
build_case(row, items_by_stage, attempts_by_stage) -> case
validate_case(row, items_by_stage, attempts_by_stage, case) -> case
replay_line(state, saved_pv, actual_depth, expected_value=None) -> line
compare_lines(left_line, right_line) -> comparison
rank_diagnostics(before_Q, after_Q, actor) -> ranks_and_gaps
```

The row is the exact original study wrapper, with its nested source root, raw state/key, actor, stratum and legal mask. Inputs are maps `h8`, `h10`, `h12` to the saved complete endpoint items and attempts. The helper distinguishes the original source-root digest from the study-wrapper digest. It validates each item/attempt hash link, all three requested depths, complete consecutive history prefixes, and the exact endpoint report at the final history entry. Auxiliary horizons cannot replace an endpoint.

All four roots and all legal actions are in scope. Their legal-action counts are 4/5/5/7: 63 endpoint action-Q values, 120 stage-history reports and 750 root/all-action PV references before deduplication. Each case keeps all 30 stage-history report references from H8/H10/H12. Overlapping actual-depth full-Q/target/best-set signatures must agree; PV differences may remain explicit. The H12 job's saved depth1–12 sequence is the single descriptive depth trajectory. Duplicate histories or root/chosen-action lines do not become independent samples.

`report_index` maps every stage/depth to its raw report hash, full Q/best signature and root/action line IDs. `lines` deduplicates identical raw-root/depth/saved-PV/value inputs, while `source_references` retains every source stage, actual depth, report hash and root-or-action identity. `counts` distinguishes source PV references, deduplicated literal lines, actual literal choices performed for those lines and referenced choices before deduplication. Per-line sowing has the interpreter's one-million-deposit guard; all recorded paths have at most their actual horizon of 12 choices. A guard violation refuses the case rather than inventing a partial result.

Every replay step records its one-based choice index, actual actor, machine pit and human pit, full raw states before/after, store gain, captured seeds/indices, relay events, deposits, bonus, forced passes and terminal status. Controller changes use the actual normalized controller; bonus turns and forced passes are not conflated. Cumulative player-specific capture and store totals remain fixed-player-0 facts. Store-margin change is separated algebraically into capture-store gain and other sowing-store gain. The literal event stream is reconstructed to verify seed conservation and its hash is retained.

Each leaf records its raw state, stores, controller, active seeds, terminal flag and saved score. For the frozen H_v1 store-margin evaluator, the saved PV score must equal the literal leaf store margin; no new evaluator is called. A nonterminal leaf still has future choices, even when this recorded-horizon equality holds. This is not proof of eventual game value or unrecorded branch quality.

Each action retains endpoint player-0 Q and actor-signed Q, full best-set membership, gap from best, signed/absolute H8→H10, H10→H12 and H8→H12 changes, and saved-line comparisons at all three horizon pairs. Immediate capturing, bonus and relay facts remain action-specific: a capture-class root means some action captures, not that every drifting action does. Saved immediate transition fields and capture transfers must match literal replay across every report.

`compare_lines` reports the identical raw-state/action prefix, the earliest different recorded choice with both controllers/mechanics, leaf states and tail store/capture facts. If one line merely ends before the other, it is a prefix relationship, not an invented action disagreement. Later trajectories beyond the split are unmatched observations. Longer PVs or later captures cannot isolate a cause of a score change.

Ranking diagnostics preserve the full tied best sets. They distinguish strict pairwise reversals, tie changes and numeric actor-relative gap changes. `before_best_anchor_gaps` preserves every previous best action as an anchor; the endpoint H10→H12 version is also exposed explicitly as `H10_best_H12_anchor_gaps`. It uses `actor_sign*(delta_best-delta_action)`. Mean absolute Q shift, a common score translation, centered dispersion and ranking changes remain different concepts; the exact decomposition belongs to the companion arithmetic report and is not inferred from PV mechanics.

Four source-bound owner methods in `tests/capture_forensics_author.py` use only the first two already-consumed capture cases and tiny literal synthetic positions. They cover all saved history/action/provenance links; hand-checkable capture/relay/bonus/pass/terminal facts; shared-prefix/full-tie/actor-relative gap semantics; and rehashed corruption plus byte-stable rebuilding. `build/capture-forensics-source-freeze.json` must pin source, tests, this document, scope, interpreter and existing source evidence before formal fixtures. The actual report is `build/capture-forensics-author.json`. Independent producer process-kill tests are separate checks, not additional owner methods here.

The companion producer persists complete case derivations. Completed semantic validation may repeat pure arithmetic and literal replay; no-work means no new case commit, Native/model activation, search, training or game. Findings will be written separately after the production diagnostic closes, so this source-bound protocol remains unchanged.

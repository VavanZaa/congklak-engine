# Dakon-n: find one guaranteed winning opening

This standalone CPU tool searches for one winning sequence, verifies it, and
stops. It does not minimize its length, enumerate every sequence, train a model,
or modify the seven-pit browser app. It requires only Python's standard library.

## Objectives and scope

Dakon-n has n pits per player and n seeds in every pit, for 2n² seeds. The
supported engineering range is **1 through 64 pits per side**. This is a
parameterized extension of our sequential relay contract, with permanent stores,
opponent-store skipping, source revisits, own-store bonus choices, own-empty
capture, and empty-row passes. A mirrored board is equivalent; certificate
output includes both right-store and left-store pit labels.

The default `majority` objective banks at least **n²+1** seeds. That guarantees
the single-round win because the opponent can never obtain as many seeds.
The separate `refill_threshold` objective banks at least **2n²−n+1**, leaving
fewer than n seeds elsewhere. For n=7 these targets are 50 and 92 respectively.
The latter certifies a seed threshold; this tool does not implement or claim a
complete multiround, closed-pit or simultaneous-opening variant.

The search continues only genuine own-store bonus choices. A capture or empty
landing ends that branch unless its resulting store already reaches the target.
Control returning through an opponent's empty-row pass is not mistaken for a
bonus. The opponent never chooses before a successful witness secures its goal.

## Run, interrupt, resume and verify

Run from the project directory. Choose a new dedicated directory for each run:

```powershell
py -3.12 -B -m congklak.dakon_opening find --n 8 --run-dir data/dakon-opening-v1/my-n8
```

For the stronger threshold or the other starting player:

```powershell
py -3.12 -B -m congklak.dakon_opening find --n 7 --goal refill_threshold --starter B --run-dir data/dakon-opening-v1/my-n7-threshold-B
```

Ctrl+C stops cooperatively at a safe boundary. Time and action limits also pause
without claiming that no win exists. Resume the saved frontier with:

```powershell
py -3.12 -B -m congklak.dakon_opening resume --run-dir data/dakon-opening-v1/my-n8 --max-seconds 60
py -3.12 -B -m congklak.dakon_opening status --run-dir data/dakon-opening-v1/my-n8
py -3.12 -B -m congklak.dakon_opening verify data/dakon-opening-v1/my-n8/certificate.json
```

The verifier can also run on its own, without importing the searcher:

```powershell
py -3.12 -B -m congklak.dakon_opening_verify data/dakon-opening-v1/my-n8/certificate.json
```

`--max-seconds` defaults to 10, `--max-actions` to 1,000,000, and checkpoints occur
every 256 simulated choices or approximately one second of active search.
Both budgets apply to additional work in the current invocation; zero disables
that particular limit. `--checkpoint-actions` changes the checkpoint interval.
An overnight invocation can use `--max-seconds 28800 --max-actions 0`.

JSON progress identifies the objective, target, current path depth, best store
seen, simulated choices, expanded positions, cache hits, checkpoint generation,
and stop reason. Progress comes from completed checkpoint boundaries. The
read-only `status` command reads the durable slots directly; `status.json` is a
convenience projection that can lag if a process dies just after a checkpoint.

Outcomes are deliberately distinct:

- `guaranteed_win`: independent seed-by-seed verification accepted the witness.
- `unknown`: a limit, cancellation, or incomplete verification stopped the run.
- `no_first_turn_witness`: this restricted bonus-chain search was exhausted.
  This is not a claim that the entire game is a loss or that later-turn winning
  strategies do not exist. It is not an independently certified impossibility
  result.

## Search and proof

The search is deterministic depth-first backtracking in canonical pit order.
It stops at the first threshold-reaching successor. A bounded FIFO cache stores
only fully searched failed states; eviction affects speed, not legal coverage.
Each continued bonus increases the starter's store, so this restricted search
does not need a repeated-cycle draw rule.

The production simulator distributes complete sowing circuits arithmetically,
then deposits the remaining seeds. Its board and counters have no fixed native
16-slot or 255-seed limit. The existing app's native DLL and neural models are
unchanged and are not dependencies of this tool.

A separate verifier drops every seed individually. It checks the initial board,
every action, relay/capture result, control and pass event, conservation, the
absence of an opponent choice before the goal, and the exact winning bound.
Certificates include the complete endpoint sequence, rule contract, objective,
starting player, source hashes, and a checksum. They explicitly state that no
shortest-sequence proof was attempted. A pending witness is not published as a
verified win. Verification can be interrupted and resumed from its saved witness.

## Persistence and practical limits

Run directories must be below `data/dakon-opening-v1/` or
`build/dakon-opening-v1/`; existing nonempty directories require `resume`.
The finder refuses the product and training directories. One OS-held writer
lock prevents two searchers from owning the same run.

`identity.json` pins n, objective, starter, rules, ordering, cache capacity and
the two source files. Resuming with different code or rules is refused. Two
alternating checksummed frontier files preserve a previous generation if the
newest is truncated or inconsistent. Writes flush and sync the temporary file
before atomically replacing its destination. The complete DFS path, next-child
cursors, bounded cache order and work counters survive a restart.

Work since the last durable checkpoint can be recomputed after a hard kill.
Reported counters describe committed logical work; they do not account for all
physical work lost in an abruptly killed process. Actual power-loss durability
and energy use have not been measured.

Each run is limited to 128 MiB of stored files, 16 MiB per JSON record, and at
most 10,000 cached states. These are application limits, not an operating-system
memory reservation. Time checks are cooperative; saving a checkpoint and ordinary
OS scheduling can add latency beyond a requested limit.

## Acceptance

Run the focused suite:

```powershell
py -3.12 -B -m unittest discover -s tests -p test_dakon_opening.py -v
```

It compares arithmetic sowing with the independent literal verifier, the old
literal interpreter, and the existing native kernel where supported. It tests
wide boards and large counts, checksum-valid invalid inputs, false bonus
handoffs, two-goal separation, completed-run idempotence, write failure,
checkpoint fallback, and real process kills around initialization, checkpoint
replacement and certificate publication. Receipts are kept separately in
`build/dakon-opening-v1/`. All generated results are non-training artifacts.

## Retained example runs

The following first-found majority witnesses were generated by the public CLI
and accepted by the standalone literal verifier. Each reaches its goal before
the opponent chooses. The lengths are observations, not minima.

| n | Choices | Starter store / total | Simulated choices in the search |
| --- | ---: | ---: | ---: |
| 7 | 13 | 54 / 98 | 126 |
| 8 | 23 | 79 / 128 | 82 |
| 12 | 36 | 147 / 288 | 163 |
| 18 | 60 | 329 / 648 | 266 |
| 64 | 265 | 4,103 / 8,192 | 1,077 |

The separate n=7 refill-threshold run banks 92 in 38 choices. A B-start n=7 run
also verifies the 54-seed witness. An n=9 CLI run was paused after five simulated
choices, then resumed to a verified win.

Certificates are in `data/dakon-opening-v1/n<N>-majority-A/certificate.json` for
the rows above. The n=7 stronger objective uses
`data/dakon-opening-v1/n7-refill_threshold-A/certificate.json`. Work counts refer
to this first-witness search, not the number of complete possible sequences.
No cross-hardware speedup or energy claim is made.

Results and acceptance: [production runs](../build/dakon-opening-v1/production-results.json),
[64-pit run](../build/dakon-opening-v1/wide-board-result.json),
[recovery receipts](../build/dakon-opening-v1/recovery.json), and
[acceptance](../build/dakon-opening-v1/acceptance.json).

## Browser controls

The local product includes a separate **Opening finder** tool. Choose n, start,
pause or resume a durable run, then replay an independently verified witness
choice by choice. Its library uses `data/dakon-opening-v1/browser/`, separate
from playable games and CLI examples. See the [product guide](../product_v2/README.md#opening-finder-in-the-browser)
for controls, limits and restart behavior. Browser-integration acceptance and
recovery receipts use `build/product-finder-v1/`; the original CLI certificates
and source identities remain unchanged.

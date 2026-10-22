# Verified K=4 Congklak oracle

The exact oracle covers every normalized stable seven-pit position with zero through four seeds outside the stores under `id-betawi-relay7-v0`. It covers **5,462 entries**, including two terminal controller encodings, and **9,520 legal edges**. These are all syntactically valid states in the declared domain; a table entry does not assert historical reachability from the opening. The 6,120 raw allocations contain 658 empty-row-controller encodings that normalize to covered entries through an explicit forced pass.

Existing stores do not influence the transition. Sowing examines pit occupancy; the opponent store is skipped; landing in the own store ends the action irrespective of its previous count; capture examines only the landing pit and opposite pit; passes and termination inspect rows. Stores never feed seeds back into play. Thus two states with identical pits and controller have identical next pits/controller, legal actions and store awards. By induction over the complete dependency graph, their eventual optimal margins differ only by the existing store offset `D = store0-store1`.

Each key contains fourteen pit counts and the controller. The stored value `F` is the margin of **future** awards, not the final margin. Terminal entries have `F=0` and a valid zero move mask. Every edge records `deltaD = deltaStore0-deltaStore1`. Player 0 maximizes `deltaD + F(child)`; player 1 minimizes it. Lookup returns `D+F`, along with every tied optimal pit. The verifier checks `abs(F)<=R`, conservation parity and immediate award accounting. Omitting `deltaD` would produce incorrect values.

Same-active-seed dependencies are common: 5,968 of the edges preserve `R`. Construction therefore generates the entire closed graph rather than assuming each move descends to a smaller seed layer. Sink-first graph resolution completes every entry and establishes an acyclic graph in this K=4 domain. Any unresolved cycle blocks publication with `unsupported_cycle`; the builder never invents a draw.

The builder uses the accepted native transition code. Before publication, the independent seed-by-seed interpreter in [tests/reference.py](../tests/reference.py) regenerates the full domain and every successor/award edge. The independently regenerated graph recomputes every value and tied optimal mask. It also checks both extreme product-98 store offsets on every edge: **19,040 store-offset transitions**. The manifest pins the reference source SHA256, rules digest, native binary SHA256, encoding, objective, quotient recurrence and logical graph/value checksums. Keep that reference source available when distributing the builder or verifier.

The SQLite artifact uses WAL, `synchronous=FULL`, one OS-locked builder and transactional chunks. A chunk commits state/edge/value rows together with the build stage. Missing completion metadata means incomplete work, even when rows are already present. Only a fully verified artifact receives a complete manifest. Restart repeats unfinished enumeration/graph/verification work while reusing committed states. Read-only lookups validate the whole artifact and retain that SQLite read snapshot, preventing subsequent writes from changing the verified view. Exact table evidence and its manifest should remain pinned by retained pilot labels.

K=4 is a small inline artifact and requires a 256 MiB free-space reserve. Admission includes sibling managed files, WAL and lock artifacts against the 50 GiB physical quota. Bulk pilot work has its own larger reserve. No incomplete or uncovered table entry is returned as exact.

```powershell
py -3.12 -m congklak.oracle build
py -3.12 -m congklak.oracle verify
py -3.12 -m unittest discover -s tests -p test_oracle.py -v
```

The default artifact is `data/oracle-k4.sqlite3`. `build(path, native=None, chunk_size=128, on_progress=None, cancel=None)` resumes its committed boundary. Returning `False` from the progress callback cooperatively stops after the announced commit. `Oracle(path)` is a context manager; `lookup(position)` returns `covered`, input and normalized controllers, pass events and, inside coverage, the exact player-0 value and optimal move mask. A terminal zero mask and an uncovered position remain distinct.

This phase exposes the oracle as separate exact lookup and exact pilot supervision. The ordinary finite-horizon `H_v1` search teacher keeps its declared leaf recurrence; the oracle is not silently substituted at search leaves.

Acceptance exercises all 6,120 raw-controller lookups, independent minimax values and optimal masks, uncovered/terminal behavior, single-owner locking, incompatible identities, malformed metadata, damaged/missing values and capacity refusal. Real child processes are killed before a pending chunk commit, after enumeration/graph/solve/verification commits, before manifest publication and after complete publication. Resumed logical manifests equal uninterrupted construction; SQLite file bytes and timings need not match.

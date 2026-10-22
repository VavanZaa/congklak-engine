# Congklak local table v2

Start from the repository folder with `tools\start-product-v2.ps1` or double-click
`Play Congklak.cmd`. The app runs at http://127.0.0.1:18767/.

The Betawi rules and H_v1 CPU search are unchanged. A final seed in your own
store grants another choice. An empty opponent row can force a pass back to you.
The board and move record now explain these consecutive turns.

The **Moves played** tab records each selected pit, store gains, and the next-turn
reason. Board and history commit together in SQLite. Saving a board also saves
its recorded moves. New game/import starts a fresh record; loading an analysis
starts a record from its root. Moves made before this update were not recorded
and cannot be reconstructed reliably.

**Live analysis** automatically follows legal board changes. It displays completed
depths, move scores, a conditional line, elapsed time, and nodes/s at the last
completed depth. You can move during analysis: the previous worker is cancelled
and joined before the move commits. Live searches are bounded to one hour or
depth 128, use fixed 512 MiB search memory and a 16 MiB TT, and never play a move
in two-player mode. Stop disables live analysis. Think time and depth settings
apply to Analyze once. In engine mode, H_v1 plays B at the existing 250 ms limit.

The app retains `data/product-v1` so existing boards and analysis checkpoints are
available. V2 adds a separate checksummed history table without changing v1's
current-board schema. Use the v2 launcher for further play. User and engineering
fixtures remain in non-training product directories. No models or search math
are changed.

Acceptance commands:

```
python -B -m unittest tests.test_product_v2 -v
node tests/product_v2_frontend.cjs
```

The recovery test kills its own child server during active live analysis,
restarts it, compares the durable board/history/report, and resumes to a deeper
horizon. Test fixtures live only in `data/product-v2-fixtures`; receipts live in
`build/product-v2`.


The position evaluation bar always uses A's perspective: gold on the left is A,
teal on the right is B. It maps the best current seed-margin estimate from -98
(B) to +98 (A), and is not a win percentage. A stale/incomplete/foreign ranking
shows an unknown placeholder, never an apparent 50/50 estimate. Actor changes
cannot reverse the displayed player perspective.

Final results and a player holding at least 50 seeds have exact outcomes because
stores never decrease. Only these proven outcomes show win/draw percentages.
Other positions need outcome calibration before numeric win probabilities can
be displayed responsibly. Live analysis follows each new board as before.


Opening check: `python -B tools/verify-product-opening-v1.py` replays the legal
one-based pit sequence `1,7,4,1,6,6,3,4,7,2,4,1,7,6` with the native transition
function and the independent literal interpreter. For either starting side,
each choice is an own-store bonus, the same side retains control with no passes,
and its store reaches 51 while the opponent's store remains 0. The other 47 seeds
cannot overturn that result. This is a constructive winning strategy from this
exact sequential opening, not a statement that all first-player moves win or
that other Congklak profiles have the same result. The evaluation meter still
represents current finite-depth search, with exact percentages only for outcomes
that are already secured on the board. This check changes no rules or models.

The October 17 rule audit found published opening variants. The user selected
sequential opening, consistent with the Ministry's [Congklak Betawi description](https://kebudayaan.kemdikbud.go.id/ditwdb/congklak-betawi/).
A [Betawi Depok field study](https://www.journal.lppmunindra.ac.id/index.php/Deiksis/article/download/15917/6222#page=8)
also describes choosing a starter by suit. A separate UIN Alauddin thesis
describes simultaneous starts. No single universal opening was established.
The brief descriptions leave some capture edge cases undefined; those remain
explicit frozen implementation conventions.

The app now labels its sequential, single-round scope visibly. A separate
`product_rules` table binds current and named saved games to the frozen rule
profile and digest. Legacy game, session, history and report payloads remain
unchanged. Migration adds missing bindings in one transaction, validates existing
bindings, and never overwrites foreign or corrupt identities. New saves bind their
rules in the same transaction as the saved board and history. Exports include
`rule_contract`; API and editor imports reject incompatible metadata. Legacy
bare stable-count imports use the explicitly selected table profile. Saved
analysis rule identity is checked before loading its board.

Full source audit and test receipts: `build/betawi-rules-audit-v1/`.
Run `python -B -m unittest tests.test_betawi_rules_audit_v1 -v` and
`node tests/betawi_rules_audit_v1_frontend.cjs` for this update's acceptance checks.

## Board layout

The browser now defaults to a store on each player's left, with sowing from
right to left as seen from that player's seat. Player A's store is on the left
of the screen and B's is on the right. The Board layout selector restores the
original right-store view; its preference survives a page restart.

Mirroring is presentation only. Pit counts, player identity, legal actions,
rules, scores, analysis jobs and saved JSON retain their original coordinates.
For a canonical zero-based pit `i`, the left-store view displays `7-i`; the
original right-store view displays `i+1`. Clicks and editor submissions always
send canonical indices. Board labels, rankings, conditional lines, protocol
explanations, move history and played-pit messages use the selected mapping.
The editor shows B above A to match the board. Imported and exported JSON keeps
its original pit order in either layout.

The certificate's original one-based sequence remains unchanged. In the default
left-store view, its displayed sequence is
`7,1,4,7,2,2,5,4,1,6,4,7,1,2`, securing the same 51 seeds.

Layout checks: `node tests/product_layout_frontend.cjs`. The existing frontend
scripts accept an optional output receipt path, so a new run can preserve older
audit receipts. Current layout receipts are in `build/product-layout-v1/`.

## Opening finder in the browser

Open **Opening finder** next to **Play & analyze**. Choose n (1–64), the
starting player and a target, then select **Find a winning sequence**. Each
side starts with n pits containing n seeds each. The default target banks
`n² + 1` seeds before the opponent chooses, guaranteeing a win in our defined
sequential, single-round profile. The optional stronger target banks
`2n² − n + 1`; it does not simulate refills or later rounds.

Progress shows completed choices tested, expanded positions, best store count,
frontier depth and checkpoint generation. These are committed work counters,
not a percentage of the unknown search space. One finder worker runs at a time.
**Pause** cooperatively saves a safe frontier; **Resume** continues it. Each
work session has a 10-minute cap. Browser disconnection leaves the local
worker running within that limit. After a server restart unfinished runs are
paused until explicitly resumed. A corrupt newest checkpoint falls back to
the older valid slot; unavailable or incompatible runs show an error.

Only independently verified certificates enable replay. **Next choice**,
**Previous**, **Reset**, the slider and the numbered line all select exact
stable positions from the certificate. Each step includes every relay within
that pit choice. Labels use the preferred left-store view, with the next pit
highlighted. Captures that end the turn are explained, including a final
capture that reaches the winning target. Large boards scroll horizontally.
The selected run, replay position and last submitted form settings survive a
page reload. The finder stops at its first witness; it proves no shortest line.
An exhausted search means no uninterrupted first-turn witness for its target,
not that the entire game is lost.

The browser library uses `data/dakon-opening-v1/browser/<opaque-run-id>/`
(100-run limit), separate from `data/product-v1/boards.sqlite3` and saved
analyses. No game schema or payload changes, neural loading, training or
research admission occur. Product fixtures use their own finder directories
under `build/dakon-opening-v1/product-fixtures/`. Existing CLI runs and source
identities stay unchanged. Finder endpoints use the same localhost Host,
Origin and command-token checks as the playable board; no client paths are
accepted.

Acceptance: `python -B -m unittest tests.test_product_finder_v1 -v` and
`node tests/product_finder_frontend.cjs`. New receipts, before/after logical
database comparisons and browser screenshots are in `build/product-finder-v1/`.

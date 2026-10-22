# Congklak and Dakon rules research and simulator contract

Research date: 14 October 2026 (Asia/Bangkok). Scope: planning only. No engine implementation or claim of a new solution.

The target is Indonesian Congklak/Dakon, initially position analysis and best-move explanations. A name alone does not identify an executable game. The sources below document materially different games, and several cultural descriptions omit edge cases. Freeze one versioned contract before generating training data, books, transposition tables, or tablebases.

## Evidence register

Source IDs below resolve to direct links. Cultural authority establishes a documented tradition; it does not make an abbreviated description a complete tournament specification. Publication dates are distinguished from repository upload dates.

| ID | Source and inspected location | Evidence and limits |
| --- | --- | --- |
| I1 | [Ministry heritage: Congklak Betawi](https://kebudayaan.kemdikbud.go.id/ditwdb/congklak-betawi/), 16 September 2019, registration 201900924 | Seven pits per player, seven seeds each, right-hand store; sequential starter; relay, store bonus, opposite capture, opponent-empty stop. Ends with all seeds in stores. Does not explicitly resolve landing-seed capture, opposite-empty capture, passing, or store skipping. Full indexed article read; direct open returned an error. |
| I2 | [Nurfalah and Fauzia, government teacher/parent manual](https://repositori.kemendikdasmen.go.id/23987/), 2020; [PDF](https://repositori.kemendikdasmen.go.id/23987/1/Model%202020-Model%20Permainan%20Trad%20Sosem%20AUD4-5-Panduan%20Congklak.pdf), general rules and printed p. 13 | Sequential start, skip opponent store, own-empty shooting, own-store continuation, opponent-empty handover. Ending wording can mean no playable seeds on one side; settlement is unspecified. Repository metadata and indexed rule excerpts inspected; full PDF fetch repeatedly failed. An early-childhood guide, not an exhaustive adjudication manual. |
| I3 | [Ministry heritage: Dakon Jawa Tengah](https://kebudayaan.kemdikbud.go.id/ditwdb/dakon-jawa-tengah/), 16 September 2019, registration 201900947 | Records andhok, own-side mbedhil opposite capture, opponent-side mikul, ngacang, bero, buk (equal result), and kobongan. Mikul mechanics are not fully defined here. Full indexed article read; direct open failed. |
| I4 | [Sleman local government: Sendangsari Dakon/Congklak](https://sendangsarisid.slemankab.go.id/home/data-kebudayaan/dakon-congklak/) | Documents mbedhil versus gotongan agreements, protected partial refill nandur kacang, and closed bera pits. Its historical section and later play instructions conflict over opening, direction, store continuation and ending. Do not merge its paragraphs into one algorithm. Full indexed article read; direct open returned 403. Similar local-government pages appear to repeat this text, so they are not independent corroboration. |
| I5 | [Supriyono, government book: Serunya Permainan Tradisional Anak Zaman Dulu](https://www.budikemdikbud.id/storage/book/MfeLXyIM1NBkLinxMYRkqdsehrtjT4WvA3GSCpTD.pdf), 2018, printed pp. 23-24 | Sequential starter, clockwise wording, relay and bonus continuation. Explicitly says no gain if the opposite pit is empty. Ownership restrictions and landing-seed handling are incomplete. Relevant extracted pages read; a later screenshot request redirected to the library homepage. |
| I6 | [Museum Prof. Dr. R. Soegarda Poerbakawatja collection](https://museumsoegarda.purbalinggakab.go.id/koleksi/4/detail/30), entry dated 15 March 2021 | Surakarta board has ten small pits and two stores; museum explicitly acknowledges rule differences. Seven-by-seven is a useful target, not universal across Indonesian objects/traditions. Full page inspected. |
| I7 | [Banjarmasin government: traditional games](https://bagpem.banjarmasinkota.go.id/2014/06/permainan-tradisional-anak-indonesia.html), June 2014, Congklak section | Alternating selection, counterclockwise wording, opposite capture requiring opponent seeds, and all-seeds-stored ending. Short description leaves relay details incomplete. Full indexed Congklak section inspected. |
| M1 | [IIUM Indigenous Fest 2019 tournament rules](https://centre.iium.edu.my/sdc/wp-content/uploads/sites/40/2024/08/rulesCongkak-1.pdf), pp. 1-3 | Simultaneous first board; capture only after one circuit; include own store, skip opponent store; empty landing ends motion. Includes physical pacing, touch commitments and a five-minute format. These are event-specific Malaysian rules. Full extracted rule pages inspected. |
| M2 | [Malaysia Veterinary Department tournament rules](https://jpvpk.gov.my/images/v3/Sukan2020/Syarat/CONGKAK.pdf), 2020 path, especially p. 2 | Capture after traversing opponent village; own-store continuation; expressly no burned houses; uses menang tapak and best-of-board/team scoring. Exact capture and ending adjudication require clarification. Full extracted document inspected. |
| M3 | [Universiti Putra Malaysia: Congkak, Permainan Tradisi Melayu](https://intl.upm.edu.my/artikel/congkak_permainan_tradisi_melayu-43481?L=bm), input 26 September 2018, updated 6 July 2020 | Simultaneous opening followed by alternating play; later refill and close unfillable holes; previous winner starts, inability to fill one hole loses. Mentions five- or seven-hole boards. Full article inspected. |
| P1 | [Philippine National Museum: Games / Hobbies / Recreation](https://www.nationalmuseum.gov.ph/our-collections/ethnology/games-hobbies-recreation/) | Sungka is a two-player game with fourteen small pits and two larger end pits. This museum summary is not a detailed rulebook. |
| P2 | [Dionido and Bautista: Mastering the Game of Sungka from Random Play](https://arxiv.org/html/1905.07102), 2019, sections III-A and III-B | Authors explicitly simplify real-time simultaneous opening and multiround burned-house play to sequential single-round research. Demonstrates why a digital environment's name does not establish equivalence to traditional play. Full relevant sections inspected. |
| D1 | [Donkers, de Voogt and Uiterwijk: Human versus Machine Problem-Solving: Winning Openings in Dakon](https://www.researchgate.net/publication/239522973_Human_versus_Machine_Problem-Solving_Winning_Openings_in_Dakon), Board Games Studies 3, 2000, pp. 79-88 | Author-uploaded paper; Maldives Ohvalhu profile, named Dakon for convenience. Seven-pit first-turn witness; rules and scope below. |
| K1 | [Irving, Donkers and Uiterwijk: Solving Kalah](https://naml.us/paper/irving2000_kalah.pdf), ICGA Journal, September 2000, sections 2.1-2.2 | Kalah ends a sowing without automatic occupied-pit relay, has store bonuses and opposite captures, and sweeps when one side empties. Warns that traditional restrictions on counting differ from exact-count computer play. Full rules section inspected. |

The 2022 paper [The winning percentage in congkak using a randomised strategy](https://sjst.psu.ac.th/journal/44-5/1.pdf), DOI [10.14456/sjst-psu.2022.151](https://doi.org/10.14456/sjst-psu.2022.151), is an additional lead: indexed rule text explicitly distinguishes empty-opposite noncapture. The journal PDF and [author-institution mirror](https://oarep.usim.edu.my/server/api/core/bitstreams/075b9c17-b854-44e1-856c-4de9c5811c6a/content) could not be fully fetched in this session. Do not use its random-play percentages as optimal-play evidence.

## Documented rule differences

Cells identify the evidence above rather than silently selecting a universal rule.

| Decision | Documented alternatives | Consequence for architecture |
| --- | --- | --- |
| Board/setup | I1: 7/7; I6: 5 pits per side; M3: 5 or 7 | Parameters belong in rules identity. |
| Opening | I1/I2: sequential; M1/M3: simultaneous; I4: contradictory descriptions | Sequential search is suitable only after its phase is defined. |
| Direction/store orientation | I1: right store/right next pit; I5: clockwise; I7: counterclockwise | Use explicit next-slot maps and a diagram. |
| Occupied last pit | I1: automatic relay | A stone drop is not normally a player decision. |
| Own-store last seed | I1/I2: another pit choice; I4 historical section differs | Same player may remain to move. |
| Capture geometry | I1: own opposite; I3/I4: additional local agreements | Capture must be a named strategy, not a fixed Kalah assumption. |
| Capture eligibility | M1: circuit; M2: opponent village; I1 omits such a restriction | Gate semantics and reset scope must be specified. |
| Opposite empty | I5/I7: no capture/gain; D1 wording allows collecting landing seed | Changes successor states and exact values. |
| End of round | I1/I7: both sides empty; I2: ambiguous one-side wording; D1: unable to move | Pass/termination/sweep cannot be inferred interchangeably. |
| Match | M3: refill/closed pits; M2: no burned houses; I4: partial protected pit | A match state is richer than a single board. |
| Tie | I3: buk; tournament formats have their own points | Board utility and match utility are separate. |

Congklak, Dakon and Congkak are regional names covering closely related sowing games, with no demonstrated one-to-one mapping to a rules algorithm. Sungka shares the board family but its opening and match practices require a separate profile. Mancala is the wider family. Kalah is a useful comparison implementation, not a substitute simulator. Oware/Awari uses different capture/feeding rules and can have cycles; [the primary solved-games survey](https://web.mit.edu/6.034/wwwbob/GamesSolved.pdf), section 2.2, documents these distinctions. Import search ideas without importing that game’s rules or solution.

## Proposed profiles for rule selection

These are engineering proposals. They must not be labelled official national rules.

### Candidate A Betawi relay rules

Profile: `id-betawi-relay7-v0`.

Use I1 as the target tradition and complete its omissions explicitly: 7 pits per side, 7 initial seeds; sequential start; fixed right-store route; relay at a pit occupied before the final deposit; bonus choice at own store; skip opponent store; own-empty capture of opposite contents plus landing seed only when the opposite is nonempty; unrestricted capture; opponent-empty stop; forced pass when the next player has no seeds; finish only when both rows empty. Stores never re-enter this single round. Final stored majority wins and 49-49 draws.

The capture contents, empty-opposite behavior, unrestricted eligibility, pass mechanism and tie interpretation are engineering completions, not facts all stated in I1. The [first implementation scope](../FIRST_IMPLEMENTATION.md) selects these as a complete experimental contract. Implementing that contract does not assert that the user's familiar convention has been matched.

### Candidate B Sequential sweep rules

Profile: `id-sequential-sweep7-v0`.

Same kernel as A, but after a complete relay action/capture, one empty row ends the round and the other row is swept into its owner’s store. This is a deliberately different analysis contract motivated by I2’s ending ambiguity and common digital play. It must be chosen explicitly; it is not a proven translation of Betawi’s both-empty ending.

### Candidate C Regional Javanese contracts

Start with separate `id-jawa-mbedhil-v0` and `id-jawa-gotongan-v0` drafts. I3/I4 justify preserving local capture agreements, not filling every omission. Before activation, clarify whether mikul/gotongan are identical, whether both adjacent opponent pits must be nonempty, capture at the edge, landing seed ownership, fixed first selections, store handover, refill direction, protected remainder and revival of bera. Retain the documentary disagreement over the three-empty-pits ending.

### Comparison and reproduction profiles

Keep a separate `paper-dakon7-2000` profile for D1’s sequential refill game, with capture even when the opposite is empty and sweep/refill/closed holes. Keep Malaysian circuit-restricted and simultaneous-opening drafts separate. Do not mix their evidence into candidate A. A physical simultaneous opening also needs an interleaving/timing/collision protocol; choosing two opening pits and applying one full action before the other is not automatically equivalent.

## Exact decision state and retrieval keys

The following is original simulator design, not an assertion that the cultural sources specify a software format.

At a stable sequential decision boundary, store:

- Rules profile ID, version and content digest; pit counts, seed total, store counts and player to move.
- Phase: normal decision, mandatory pass, round settlement, match refill, simultaneous opening, terminal.
- Active/closed/protected pit masks and protected remainder where supported.
- Capture eligibility/history flag if a circuit rule carries across bonus choices. Clarify whether its reset is per selected pit, whole turn, round or match.
- Match allocation, next starter and tie policy where those change future play. An administrative round number is useful provenance but need not be in the semantic hash if it cannot change the future.

One action is a pit choice followed by all deterministic sowing and relays until another choice, handover or terminal state. Keep bonus choices as separate search edges; branch count is at most seven for the base board. Return the actual next player explicitly. Negamax/value backups must negate on player change, not on every edge.

Keep an optional event trace for explanations: pickup, individual deposits, relay pickups, store increment, capture, extra choice, pass and settlement. A UI may animate the trace without creating additional engine decisions. Imported mid-sowing positions need a separate transient state with hand count, cursor and active mover; reject them as incomplete if those fields are absent.

Position cache, exact tablebase, evaluation cache and opening book must verify the full semantic key. A hash is an index, not equality proof. Keep provenance and exact/lower/upper/heuristic status with retrieved values. A stored score difference affects final win/loss: omitting both stores from a neural observation or WDL lookup can alias positions with different winners. Compression is possible only with a proved value convention, e.g. future score increment plus separately supplied existing difference.

Sequential symmetry can swap players and rotate the board only if it also transforms stores, routes, opposite map, move labels and metadata. Arbitrary mirroring reverses sowing direction and is generally invalid. Simultaneous and protected-pit profiles require their own symmetry audit.

## Canonical diagram and conformance fixtures

Define A as the bottom player facing upward; B is the top player facing downward. Each row is numbered from that player’s own left to right. This diagram fixes orientation rather than trusting a clockwise label:

```text
          B faces down
SB     B6 B5 B4 B3 B2 B1 B0
       A0 A1 A2 A3 A4 A5 A6     SA
          A faces up

absolute ring: A0 A1 A2 A3 A4 A5 A6 SA B0 B1 B2 B3 B4 B5 B6 SB
opposite: A_i <-> B_(6-i)
A skips SB; B skips SA; source pit is revisited on a sufficiently long lap.
```

Fixtures below are synthetic transition inputs; they are not claimed reachable from the normal opening. Unlisted pits are zero. Pad stores to total 98 as shown. Candidate A expectations apply unless otherwise stated. No actual simulator has been run in this planning phase.

| Test | Input/action | Required expectation |
| --- | --- | --- |
| Initialization | Both rows all 7; SA=SB=0 | Total 98, seven legal choices for starter. |
| Bonus | A0=1,A6=1,B0=1; SA=47,SB=48; A chooses A6 | SA becomes 48; A remains to move; A0 remains legal. |
| Own capture | A0=1,A5=1,B0=4; SA=46,SB=46; choose A5 | Final A6 and opposite B0 become empty; SA=51; A0 retains one. Pass normalization follows, since B is empty. |
| Empty opposite | A0=1,A5=1,B3=1; SA=47,SB=48; choose A5 | A6 retains one, stores unchanged, B moves. Always-capture profile instead stores the landing seed; record distinct outputs. |
| Relay on both sides | A0=1,A4=1,A5=2,B0=1,B1=1; SA=46,SB=46; choose A4 | Relay from A5 with 3, then B0 with 2; stop at B2. End: A0=1,A6=1,B1=2,B2=1,SA=47,SB=46; B moves. |
| Skip opponent store | Exercise a lap crossing SB for A and SA for B | Opponent store does not increment and does not consume a seed. Test both players. |
| Source revisit | Isolated single-lap primitive: A0=16; SA=41,SB=41; choose A0 | Before resolving the final-pit relay, A0=1,A1=2, other small pits=1, SA=42,SB=41. Source is not skipped. |
| Forced pass and revival | A row empty; B6=2; SA=47,SB=49; A to move | A must pass. B chooses B6: deposits in SB and A0; A then has a legal A0 choice. Immediate sweep would produce a different state. |
| No premature finish | Pick up the only own pit while seeds remain in hand | Do not settle merely because pickup temporarily emptied the row. Complete sowing/relay/capture before checking the selected ending rule. |
| Eligibility gate | Reuse own-capture fixture with no completed circuit | Unrestricted profile captures; circuit-restricted profile does not. Repeat with eligibility true and with bonus-choice history. |
| Tie | All pits empty; SA=SB=49 | Terminal draw in single-round score profiles. Match response is profile-specific. |
| Closure/refill | Sweep/refill match fixture with an insufficient store | Verify active mask, ordered refill, protected excess and score conservation against the chosen contract. Never silently drop remainder seeds. |

Additional requirements: reject empty/enemy/closed/protected pit choices; preserve total seeds including hand/protected quantities; capture once; never relay a store; distinguish pre-deposit empty from post-deposit singleton; ensure mandatory pass is unique; perform settlement exactly once; make/unmake must restore every semantic field; player-swapped fixtures must agree after exact coordinate transformation. Golden full traces must be approved for local variants.

## Published first turn construction

D1 reports a seven-pit shortest winning opening of 25 choices (Table 2, p. 85), using holes numbered left-to-right:

```text
1,7,4,1,6,7,4,3,1,1,4,1,4,7,2,1,5,2,3,6,1,5,5,7,1
```

Verify every bonus choice and the strict settlement condition: the opponent owns fewer than seven seeds. Exactly 91 of 98 leaves 7; it does not force match loss. The construction covers the matched opening, not arbitrary positions or maximum margin. The paper's circuit count describes laps, not repeated-state loops.

## Termination and cycle proof obligations

Do not assume every mancala graph is acyclic, and do not use an arbitrary simulation cap as a draw rule.

For the proposed sequential, single-round, fixed-route contracts with permanent stores, a candidate termination argument is available: a complete pit choice either decreases active seeds R through a store/capture, or never reaches the mover’s store and moves seeds forward solely within the mover’s own row. In the latter case, the sum Phi of seed counts times their forward distance to their owner’s store decreases. Thus lexicographic (R,Phi) decreases on effective moves. Normalize forced passes; a double pass is terminal. Separately verify relay termination: a no-store segment advances toward the mover’s store, and store-depositing segments cannot occur indefinitely with a fixed seed total.

This is a proposed engineering proof to check against the frozen transition semantics, including gate/protected-pit behavior. It is not a published theorem cited from Kalah. Multiround refill resets active seeds, so this single-round argument does not establish match termination. Simultaneous scheduling also changes the argument. Until proved, exact solvers must have correct repeated-state/SCC handling and a defined payoff for infinite play; store monotonicity by itself is insufficient evidence of strict progress.

## Uncertainty register and rule selection gate

| ID | Unresolved question | Required resolution |
| --- | --- | --- |
| U1 | Which Indonesian local practice does the user intend? | Select a named profile and approve the board/route diagram. |
| U2 | Opening simultaneous, sequential, or compulsory first-pit exchange? | Explicit decision protocol; physical timing details if simultaneous. |
| U3 | Capture landing seed? Capture it with empty opposite? | Approve the two capture fixtures. |
| U4 | Circuit/opponent-village restriction and reset scope? | Define eligibility flags and trace examples. |
| U5 | One empty row: pass, loss, sweep, or wait for refill? | Approve the forced-pass/revival fixture and settlement timing. |
| U6 | Opponent-empty mikul/gotongan mechanics? | Local rule author/adjudicator plus edge-position examples. |
| U7 | One round versus continued refill match? | Define masks, refill order, reactivation, protected excess, starter and tie. |
| U8 | Human counting restrictions? | Initial engine uses explicit counts; an uncertain human observation is not an exact state. A future belief-state interface needs its own observation rules. |
| U9 | Proof scope and infinite-play treatment? | Prove termination for selected profile or define cycle adjudication. |
| U10 | Solver target? | Position advice first; specify opening proof, maximum score, WDL or all-reachable-state solution separately. |

Implement the complete experimental contract selected in [FIRST_IMPLEMENTATION.md](../FIRST_IMPLEMENTATION.md), with a rules digest and internally reviewed golden traces. Other profiles and local convention matching retain the unresolved questions above. Generate reusable data only after that contract's simulator passes conformance; describe every result by its exact profile, state and utility.

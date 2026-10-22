# Saved capture-Q drift findings

The frozen weighting proposal increases mean H10/H12 **teacher-action-Q drift**, while the H10-best action remains H12-best at all four capture roots. This is a change in a retrospective diagnostic metric, with no newly trained model, new teacher search, or new game. It does not establish weaker play or justify deleting capture data. Existing sampling/loss settings and the allocation HOLD remain unchanged; no proposed weights were applied.

The evidence is [capture-drift-result.json](../build/capture-drift-result.json), covering all four capture roots and all 16 original assessment roots. Root names below are the final components of `midgame-history-pilot/20261023/<name>`. Pits are human 1–7; saved machine pits are 0–6. Q is the fixed-player0 seed-store margin at the saved finite horizon. Actor0 maximizes Q; actor1 minimizes it.

## Where the aggregate change comes from

The original comparator assigns each root 1/16. The frozen proposal preserves each stratum's 1/4 mass and each actor-cell's 1/8 mass. With each root's legal actions averaged first, the global action-Q drift changes from **1.9994047619 to 2.0237621060**, a difference of **20888226409/857574058320 = +0.02435734408**.

| Fixed source group | Global weighted-minus-uniform contribution |
|---|---:|
| Four capture roots | 894149/29924160 = +0.02988050458 |
| Other 12 roots | −607811/110047680 = −0.00552316051 |
| High ordinary stratum | −0.00661900369 |
| Low ordinary stratum | 0 |
| Bonus stratum | +0.00109584318 |

Within capture actor0, conditional mean drift rises from 47/40 to 78/61; within actor1, from 27/14 to 452/219. In both cells the proposal upweights the root with greater later drift. The respective global increments are 253/19520 and 415/24528. These are arithmetic attribution identities, not causal estimates.

| Root | Actor | Proposed root mass | D | Global change from this root |
|---|---:|---:|---:|---:|
| 283 | 0 | 9/122 | 7/4 | +77/3904 |
| 445 | 1 | 185/2628 | 3 | +83/3504 |
| 233 | 0 | 25/488 | 3/5 | −33/4880 |
| 201 | 1 | 287/5256 | 6/7 | −83/12264 |

The capture *root* tag means at least one legal move captures immediately. All four stable H10/H12 best moves instead grant an immediate bonus choice without capturing: pits **4, 6, 7, 7** respectively. Of 21 legal actions, seven capture immediately. Those seven contribute **31807/1282464 = +0.02480147591** to the global excess; the other 14 actions within capture roots contribute **455957/89772480 = +0.00507902867**. Together they give the capture-root excess above. This split preserves root/legal-action denominators; it is not a comparison of two independently normalized action populations. Nonbest H10 actions contribute +0.02142306863 and best actions +0.00845743595 to that same excess.

## All saved action scores and relative gaps

The gap is actor-best score minus actor-action score at that horizon. `Δ0/Δactor` is H12 minus H10: the actor sign reverses for roots 445 and 201. `Ck` means an immediate capture of k seeds, `B` a bonus choice, and `Rk` k relay pickups. These describe the first action only.

| Root | Pit | Q8 → Q10 → Q12 | Δ0 / Δactor | Gap10 → Gap12 | Immediate mechanics |
|---|---:|---|---|---|---|
| 283 | 2 | 5 → 4 → 7 | +3 / +3 | 18 → 14 | C4 |
| 283 | 4 | 20 → 22 → 21 | −1 / −1 | 0 → 0 | B, R1 |
| 283 | 5 | 3 → 1 → 0 | −1 / −1 | 21 → 21 | C2 |
| 283 | 6 | 1 → 0 → −2 | −2 / −2 | 22 → 23 | C2 |
| 445 | 2 | −13 → −10 → −13 | −3 / +3 | 31 → 23 | C8, R1 |
| 445 | 3 | −28 → −28 → −30 | −2 / +2 | 13 → 6 | B, R2 |
| 445 | 4 | −26 → −27 → −30 | −3 / +3 | 14 → 6 | R1 |
| 445 | 6 | −39 → −41 → −36 | +5 / −5 | 0 → 0 | B |
| 445 | 7 | −22 → −27 → −29 | −2 / +2 | 14 → 7 | C2, R3 |
| 233 | 1 | 16 → 23 → 25 | +2 / +2 | 27 → 25 | R1 |
| 233 | 2 | 8 → 3 → 3 | 0 / 0 | 47 → 47 | C2 |
| 233 | 3 | 43 → 40 → 40 | 0 / 0 | 10 → 10 | B, R4 |
| 233 | 4 | 47 → 48 → 49 | +1 / +1 | 2 → 1 | B, R4 |
| 233 | 7 | 47 → 50 → 50 | 0 / 0 | 0 → 0 | B |
| 201 | 1 | 40 → 41 → 40 | −1 / +1 | 14 → 15 | B, R4 |
| 201 | 2 | 26 → 28 → 27 | −1 / +1 | 1 → 2 | B, R1 |
| 201 | 3 | 28 → 29 → 28 | −1 / +1 | 2 → 3 | B, R1 |
| 201 | 4 | 46 → 45 → 46 | +1 / −1 | 18 → 21 | R5 |
| 201 | 5 | 37 → 36 → 36 | 0 / 0 | 9 → 11 | C2, R7 |
| 201 | 6 | 44 → 44 → 44 | 0 / 0 | 17 → 19 | R5 |
| 201 | 7 | 22 → 27 → 25 | −2 / +2 | 0 → 0 | B |

There are **no strict H10/H12 pairwise ranking reversals**. Root445 has two nonbest tie changes: pits4/7 are tied at H10 and separate at H12, while pits3/4 become tied at H12. All four H10/H12 best sets are unchanged singletons. Earlier H8 has the full tie `{4,7}` at root233, narrowed to `{7}` at H10. Stable best sets coexist with numeric gap changes: root445's closest alternative gap shrinks from 13 to 6, root233's from 2 to 1, and root201's expands from 1 to 2. Mean full-action drift includes weak alternatives as well as close choices; it is not best-move regret.

## Translation, cancellation, and dispersion

For legal-action deltas δ = Q12 − Q10, let c = mean δ, D = mean |δ|, C = |c|, S = D − C, and E = mean |δ − c|. **D = C + S**. S measures signed cancellation; it is not ranking error. E measures centered dispersion separately, and **D does not equal C + E** for any of these four roots.

| Root | c, player0 / actor | D | C | S | E |
|---|---|---:|---:|---:|---:|
| 283 | −1/4 / −1/4 | 7/4 | 1/4 | 3/2 | 13/8 |
| 445 | −1 / +1 | 3 | 1 | 2 | 12/5 |
| 233 | +3/5 / +3/5 | 3/5 | 3/5 | 0 | 18/25 |
| 201 | −4/7 / +4/7 | 6/7 | 4/7 | 2/7 | 38/49 |

Mixed signs at 283, 445, and 201 rule out a shared translation as the complete numerical description. Root233 has no signed cancellation, yet unequal action changes still give positive E and different gaps. Neither decomposition by itself identifies the cause of horizon sensitivity.

## What the saved lines show

The H10 and H12 root-best PVs first choose different actions at choices **2, 5, 8, 5** for roots 283, 445, 233, 201. They are different downstream trajectories, not the same H10 line extended by two choices. Tail counts below start after their common choice prefix; capture/store vectors are `[player0, player1]`.

| Root | First differing choice, H10 → H12 | Tail captures, H10 → H12 | Tail store gains, H10 → H12 | Active seeds at leaves, H10 → H12 |
|---|---|---|---|---|
| 283 | choice2: P0 pit7 → pit5 | [8,0] → [8,2] | [14,2] → [15,4] | 12 → 9 |
| 445 | choice5: P1 pit3 → pit1 | [0,14] → [4,15] | [0,21] → [5,21] | 21 → 16 |
| 233 | choice8: P0 pit3 → pit2 | [8,0] → [8,0] | [12,0] → [13,1] | 12 → 10 |
| 201 | choice5: P1 pit1 → pit7 | [3,10] → [3,10] | [7,11] → [7,13] | 7 → 5 |

At root445's first differing choice, the H10 line grants P1 a bonus after five relays; the H12 line instead captures four seeds and passes to P0. Its best-action Q changes −41 → −36, an actor1 decrease of five. At root283's nonbest pit2, Q changes 4 → 7 despite no tail captures on either line: those action PVs first differ at choice10. At root201's best pit7, the post-prefix capture totals are identical but P1's tail store gain rises by two. Captures alone therefore do not describe every saved Q change. These are line facts, not controlled causal decompositions.

All 252 deduplicated lines end **nonterminal**, with 4–47 active seeds across their leaves. The score equals the saved leaf's store margin; zero arithmetic residual does not remove uncertainty from the unplayed remainder. All completed overlapping depths agree in full Q and best sets. The same H12 job supplies a retained depth1–12 trajectory; for example root445's best-value sequence at depths8–12 is −39, −40, −41, −39, −36 while pit6 remains best. This does not establish convergence.

## Coverage and decision

The four roots have 4/5/5/7 legal actions: **63 endpoint action-Q values**, **120 saved stage-history reports**, and **750 root/action PV references**, deduplicated to **252 literal lines / 1,638 replayed choices**. References that repeat a completed prefix are retained with their sources and are not additional independent samples. Every legal action is included, including zero drift and unchanged ranking. Literal replay uses [tests/reference.py](../tests/reference.py); no Native engine, model inference, teacher search, new arena, training, or pruning was performed.

This consumed four-root diagnostic explains the arithmetic location and several saved trajectory mechanisms of the failed weighting screen. It cannot establish H12 game truth, sampling-rule generalization, model strength, or a reason to prune capture positions. Keep the current settings and HOLD. The detailed source-bound protocol is [CAPTURE_FORENSICS.md](CAPTURE_FORENSICS.md); journal/recovery and independent evidence are documented in [CAPTURE_INDEPENDENT_AUDIT.md](CAPTURE_INDEPENDENT_AUDIT.md).

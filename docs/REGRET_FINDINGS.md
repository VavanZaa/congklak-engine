# Saved regret reversal and capture-Q findings

The revised weighting screen remains **HOLD**. Root178 is a real disagreement between saved finite horizons, and the four capture roots explain the capture-category drift increase. This investigation changes no model, policy, label, action, probability or training setting. It performs only saved-data arithmetic and literal PV replay; there are no new searches, model evaluations, games or gradients.

Evidence: [regret-drift-result.json](../build/regret-drift-result.json), the closed `data/regret-drift-v1` journal, and the source-bound [REGRET_FORENSICS.md](REGRET_FORENSICS.md) protocol. All16 assessment roots remain the numerical support; exactly five receive detailed literal replay. Root suffixes below denote `midgame-history-pilot/20261023/<suffix>`. **Pits are human1–7**, with machine indices0–6 retained in the artifacts. All Q values are player0 seed-store margins; actor1 prefers smaller Q.

## Exact attribution, without changing the population

| Saved H10/H12 quantity | Uniform U | Revised W | W−U |
|---|---:|---:|---:|
| Overall mean action-Q drift | 13309/6720 | 23324807/11880960 | −41101/2376192 ≈−0.01729700 |
| Overall earlier-best mean regret | 1/4 | 2/7 | +1/28 |
| Overall earlier-best worst regret | 1/4 | 2/7 | +1/28 |

The capture roots' **global contribution difference is +31/11424 ≈+0.00271359**; the other12 roots contribute **−2797/139776 ≈−0.02001059**. Their sum is the lower overall Q-drift total above. Capture mass is1/4 under both comparators. The five detailed cases have mass5/16 under U and9/28 under W; they must not be silently renormalized and presented as the original16-root mixture. All stratum/actor/cell and remaining-root contributions are retained in the result.

Only root178 has positive H12 regret for its H10-best set. The probabilities changed its diagnostic emphasis, not the saved error: regret4 at U=1/16 contributes1/4; at W=1/14 it contributes2/7. All five requested H10/H12 best sets are singletons, so mean and worst tied-best regret coincide in this observed panel. Full ties at intermediate depths are still preserved.

## Root178: the late pit6/pit7 reversal

Root178 is actor1, with stores `[48,22]` and28 active seeds. All seven alternatives remain in the review:

| Pit | Q8 → Q10 → Q12 | Actor1 gap10 → gap12 | Immediate action |
|---|---|---|---|
| 1 | 31 →31 →32 | 18 →20 | 2 relays; no capture/bonus |
| 2 | 34 →35 →35 | 22 →23 | 2 relays; no capture/bonus |
| 3 | 24 →25 →25 | 12 →13 | Bonus, no relay/capture |
| 4 | 22 →22 →21 | 9 →9 | Bonus, no relay/capture |
| 5 | 30 →30 →30 | 17 →18 | 2 relays; no capture/bonus |
| 6 | **14 →13 →16** | **0 →4** | Bonus, no relay/capture |
| 7 | **16 →14 →12** | **1 →0** | Bonus, no relay/capture |

At H10, pit6 beats pit7 by one seed in actor1 orientation. At H12, pit7 beats pit6 by four: pit6 changes by+3 player0 seeds (−3 for actor1), while pit7 changes by−2 player0 seeds (+2 for actor1). Their relative advantage swings by−5. This is the sole strict H10/H12 pairwise reversal at this root, with no endpoint tie change.

The following trajectory is from **one saved H12 job's completed iterations**, not twelve independent searches or new requested H11 supervision. Its Q/best signatures match the other jobs at overlapping depths.

| Actual depth | Q(pit6) | Q(pit7) | Actor1 advantage of6 over7 | Entire best set |
|---:|---:|---:|---:|---|
| 1 | 25 | 25 | 0 | {1,2,3,4,5,6,7} |
| 2 | 24 | 24 | 0 | {3} |
| 3 | 22 | 22 | 0 | {4,6,7} |
| 4 | 21 | 20 | −1 | {7} |
| 5 | 20 | 20 | 0 | {6,7} |
| 6 | 18 | 18 | 0 | {6,7} |
| 7 | 15 | 17 | +2 | {6} |
| 8 | 14 | 16 | +2 | {6} |
| 9 | 13 | 16 | +3 | {6} |
| 10 | 13 | 14 | +1 | {6} |
| 11 | 15 | 11 | **−4** | {7} |
| 12 | 16 | 12 | −4 | {7} |

The late strict crossover is depth10→11, after earlier shallow ties and a pit7 preference at depth4. Thus endpoints alone would conceal a nonmonotone history. Agreement at depths11/12 is not proof of convergence.

The H10 root-best PV starts with pit6, while H12 starts with pit7: their root-best lines diverge immediately. Comparing the **same fixed initial action** across horizons gives a more precise saved-line description:

| Fixed action / horizon | Leaf stores `[P0,P1]` | Q | Active seeds | Captures `[P0,P1]` | Bonus choices |
|---|---|---:|---:|---|---:|
| Pit6 /H10 | [48,35] | 13 | 15 | [0,5] | 7 |
| Pit6 /H12 | [50,34] | 16 | 14 | [0,4] | 8 |
| Pit7 /H10 | [48,34] | 14 | 16 | [0,2] | 9 |
| Pit7 /H12 | [48,36] | 12 | 14 | [0,3] | 11 |

For pit6, the first seven choices match: six P1 bonuses followed by a two-seed P1 capture and a handoff to P0. At choice8, the H10 line plays P0 pit1 without a store gain; the H12 line plays P0 pit6 and gains a store seed. After that shared prefix, H10 store gains are `[0,4]`, versus `[2,3]` at H12, producing the observed+3 player0 Q difference. Later choices are different trajectories, so this arithmetic is not a counterfactual proof that the first differing move causes the estimate's error.

For pit7, the first four choices match. At choice5 the saved H10 line takes P1 pit3 and H12 takes P1 pit7; both grant another bonus. The H10 line contains nine P1 bonus choices before a two-seed capture on choice10. The H12 line contains eleven P1 bonuses before a three-seed capture on choice12, with no relays anywhere on that H12 line. P0 has no recorded choice on either pit7 line. These bonus chains and different horizon boundaries help describe the saved leaf margins; they do not determine the later game's result. Every leaf in the table is nonterminal.

## All four capture roots and every legal action

`Ck` denotes an immediate capture of k seeds; `B` a bonus; `Rk` k relay pickups. A capture root means **some available action captures**, not necessarily the best action. Root91's best move is `own_empty`, root382's best captures16, and roots92/390 choose immediate bonuses without captures.

| Root / actor | Pit | Q8 → Q10 → Q12 | Actor gap10 → gap12 | Immediate mechanics |
|---|---:|---|---|---|
| 91 /0 | 2 | −10 →−11 →−12 | 0 →0 | Own empty; no capture/bonus/relay |
| 91 /0 | 4 | −35 →−34 →−34 | 23 →22 | No capture/bonus/relay |
| 91 /0 | 5 | −19 →−21 →−17 | 10 →5 | C3, R1 |
| 382 /0 | 2 | −31 →−31 →−34 | 0 →0 | C16 |
| 382 /0 | 3 | −52 →−56 →−57 | 25 →23 | C4 |
| 382 /0 | 5 | −46 →−48 →−50 | 17 →16 | C4 |
| 382 /0 | 7 | −41 →−40 →−40 | 9 →6 | C4, R1 |
| 92 /1 | 1 | −46 →−45 →−45 | 4 →7 | B, R1 |
| 92 /1 | 2 | −34 →−38 →−40 | 11 →12 | C2 |
| 92 /1 | 4 | −23 →−19 →−18 | 30 →34 | R4 |
| 92 /1 | 5 | −47 →−49 →−52 | 0 →0 | B, R1 |
| 92 /1 | 7 | −44 →−44 →−48 | 5 →4 | B, R3 |
| 390 /1 | 1 | 9 →12 →16 | 21 →23 | R1 |
| 390 /1 | 2 | 0 →−1 →−1 | 8 →6 | R2 |
| 390 /1 | 3 | −5 →−3 →−4 | 6 →3 | C2, R1 |
| 390 /1 | 4 | 8 →13 →12 | 22 →19 | R1 |
| 390 /1 | 5 | 12 →12 →10 | 21 →17 | R2 |
| 390 /1 | 7 | −12 →−9 →−7 | 0 →0 | B |

All four H10 best actions remain H12-best, with zero best-action regret. Lower rankings still change: root92 reverses the order of pits1/7; root390 reverses pits1/4 and splits the H10 tie at pits1/5. These are nonbest comparisons. Roots91/382 have no strict reversal or tie change at H10/H12. Across H8→H10, root390 has two strict reversals and one tie change; across H8→H12 it has two strict reversals and no tie change. The other capture roots have no earlier endpoint ranking change except root92's same nonbest reversal across H8/H12.

The capture contribution increase is an exact fixed-mass weighting identity. Root91 contributes+5/336, root382−3/224, root92+1/136 and root390−5/816 to the global weighted-minus-uniform mean-Q drift; these sum to31/11424. Each actor cell upweights its greater-later-mean-drift root. This describes the failed reliability ordering, not an effect of applying weights.

Importantly, immediate-capturing actions contribute **−3/4760 ≈−0.00063025**, while other actions inside capture roots contribute **191/57120 ≈+0.00334384**. Together they give the positive capture-root total. H10-best actions contribute−313/114240, versus+89/16320 from nonbest actions, with the original legal-action denominators retained. Neither a capture root tag nor an immediate capture tag alone explains the aggregate regression or justifies pruning.

## Saved PV tails and residual ambiguity

The four capture root-best H10/H12 PVs share respectively2,6,8,2 choices before choosing different continuations:

| Root | First different choice, H10 →H12 | Post-prefix captures `[P0,P1]` | Post-prefix store gains | Leaf active seeds |
|---|---|---|---|---|
| 91 | choice3: P0 pit6 →pit2 | [6,4] →[6,5] | [11,8] →[11,9] | 17 →16 |
| 382 | choice7: P0 pit5 →pit3 | [2,0] →[5,4] | [4,1] →[5,5] | 7 →2 |
| 92 | choice9: P1 pit7 →pit2 | [0,4] →[0,4] | [0,5] →[0,8] | 19 →16 |
| 390 | choice3: P0 pit2 →pit4 | [3,6] →[11,8] | [7,7] →[14,12] | 21 →9 |

Root382's H12 tail includes two forced passes and ends with two active seeds, still nonterminal. Root92 has identical tail capture totals but three more P1 store seeds in H12, so capture transfer alone does not account for its Q change. Root390's nonbest pit5 is a different pattern: the H10 line is an exact10-choice prefix of its H12 line, despite other actions diverging earlier. All these comparisons preserve actual controllers and bonus/pass events; they do not assume turns alternate or treat later unmatched states as common positions.

For completeness, signed-change decomposition uses D=mean|δ|, C=|meanδ|, S=D−C and E=mean|δ−meanδ| for δ=Q12−Q10:

| Root | Mean δ, P0 / actor | D | C | S | E |
|---|---|---:|---:|---:|---:|
| 178 | +1/7 /−1/7 | 1 | 1/7 | 6/7 | 52/49 |
| 91 | +1 /+1 | 5/3 | 1 | 2/3 | 2 |
| 382 | −3/2 /−3/2 | 3/2 | 3/2 | 0 | 1 |
| 92 | −8/5 /+8/5 | 2 | 8/5 | 2/5 | 42/25 |
| 390 | +1/3 /−1/3 | 5/3 | 1/3 | 4/3 | 16/9 |

Always D=C+S; E is a separate dispersion statistic, not an additive replacement for S or a ranking-error measure. Root382 has no signed cancellation yet changes numerical gaps; root178 has substantial cancellation and an actual best-action crossover. This separates several numerical phenomena without establishing a causal explanation.

## Coverage, limits and next work

The five cases retain **25 legal actions,75 endpoint Q values,150 stage-history references and900 PV references**, deduplicated to **300 literal lines /1,950 choices**. Every line reconciles the saved H_v1 score to its leaf-store margin with zero arithmetic residual. **All300 leaves are nonterminal**, with2–39 active seeds. The55 action pairs yield660 depth rows from the same saved histories, not independent samples. The full16-root ledger includes unchanged scores, all horizon pairs, signed ranks, subset masses and exact U/W contributions.

This closes the requested saved-evidence diagnosis, not the uncertainty of the game, model or later horizons. H12 is not solved truth; PV mechanics cannot establish the cause of learning errors, stronger play or neural stability. The existing models, shared policy, sampler/loss and search settings remain unchanged, with no new weighting fit or application.

The next-work recommendation is the separate held [REGRET_NEURAL_NEXT.md](REGRET_NEURAL_NEXT.md): close these weighting experiments, audit targets/perspective and fixed-feature representability first, then consider controlled work on the existing compact neural value, PUCT and recoverable self-play with independent evaluation. This report supplies no certificate of neural confidence or readiness for larger training. Independent execution/recovery evidence is recorded in [REGRET_INDEPENDENT_AUDIT.md](REGRET_INDEPENDENT_AUDIT.md).

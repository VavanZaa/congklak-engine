# Root178 and new capture Q-drift diagnosis

This review uses saved H8/H10/H12 searches from the consumed qstability panel. Models, policy and active weights stay fixed. No teacher search, model forward, training, self-play, strength game, pruning or promotion runs in this stage. The previous weighting rule remains rejected; the allocation HOLD remains in place.

H8, H10 and H12 mean teacher searches to 8, 10 and 12 pit choices. They are search horizons, not different neural networks or training generations. A bonus choice can keep the same player in control.

## Root178: a small early change conceals a later move reversal

Source family `midgame-history-pilot/20261023/178`, full state `0200000001050330020205040102011601`, is a bonus-stratum position with actor1 to move. Values are player0 seed margins, so actor1 chooses the smaller value. Human pit numbers below start at one.

| Candidate | H8 | H10 | H12 |
| --- | ---: | ---: | ---: |
| Pit6 | 14 | 13 | 16 |
| Pit7 | 16 | 14 | 12 |

H10 prefers pit6 by one seed. H12 prefers pit7 by four. The relative gap changes by five seeds even though the selected root value changes only from13 to12. H10's move therefore has four seeds of regret against the finite H12 reference. This is not an exact game-loss certificate.

Both moves immediately earn a bonus choice, without a capture or relay pickup. The saved continuation and finite leaf stores differ downstream. Their ordering also changes across the saved depth1–12 trajectory, so the last two-depth comparison is not evidence of a converged answer.

The final strict crossover occurs between depths10 and11. Comparing H10/H12 lines for the same fixed first move, pit6 shares seven choices before a different player0 response; pit7 shares four before a different player1 bonus-chain continuation. These are recorded branch changes, not counterfactual proof of why the estimates differ.

The frozen maximum-Q stability rule assigns root178 weight1/14 instead of1/16, because its early maximum change is2 and its cell partner's is4. This single root accounts for the full increase in overall best-action regret: `4*(1/14-1/16)=1/28`, taking the panel mean from1/4 to2/7. The rule changes the diagnostic mixture; it does not change a model or a chosen action. Its absolute-score stability summary does not protect near-tied move ordering.

## Capture roots: unchanged best moves with changing alternative scores

All four roots below use the same source-family prefix. Their H10 best sets remain H12-best, with zero best-action regret.

| Root suffix | Actor | Legal actions | Frozen W | Mean absolute H10/H12 Q drift |
| --- | ---: | ---: | ---: | ---: |
| 91 | 0 | 3 | 1/14 | 5/3 |
| 382 | 0 | 4 | 3/56 | 3/2 |
| 92 | 1 | 5 | 9/136 | 2 |
| 390 | 1 | 6 | 1/17 | 5/3 |

Both two-root actor cells shift probability toward the root with greater subsequent mean Q drift. Conditional capture-stratum drift rises from41/24 to2455/1428, about1.708333 to1.719188 seeds. Every cell retains its original1/8 mass. Capture contributes a positive global drift change, while other roots more than offset it: full-panel mean Q drift still falls from1.980506 to1.963209. The favorable pooled mean therefore conceals the capture and regret failures.

A capture-stratum root has a capture available; this does not identify the mechanism of its best move or of every drifting action. The detailed forensic report retains every legal action, all saved horizon histories, signs, ranks, gap changes, literal captures/relays/bonuses and finite leaf stores. Changed recorded paths describe observed search behavior; they do not isolate the causal source of finite-reference disagreement or expose unrecorded alternatives. H12 remains approximate finite-horizon supervision.

Only root382's stable best move immediately captures; root91's best move ends in an empty own pit, and roots92/390 choose bonus moves. Roots92 and390 each have one strict H10/H12 reversal among nonbest alternatives; root390 also splits one tie. Stable best moves therefore do not imply stable full rankings. The largest score changes occur on a mix of capturing, bonus and ordinary-ending actions; capture-data removal is not supported by these observations.

The four capture roots contribute `+31/11424` to the global weighted-minus-uniform Q drift. Within those roots, immediately capturing actions contribute `-3/4760`, while their other actions contribute `+191/57120`. The excess is therefore carried by other actions inside capture-tagged positions. Both parts use the original root probabilities and all-legal-action denominator.

All16 context roots and all82 legal root actions remain in the exact attribution. Root178 and the four captures are the five detailed cases; their peers and other strata remain in the numerical denominator. No favorable subset replaces the frozen panel.

The five detailed cases retain150 saved iteration reports,75 endpoint action-Q values and900 PV references. Deduplication leaves300 literal lines with1,950 choices; all leaves are nonterminal. All55 unordered legal-action pairs have the full saved depth1–12 trajectory,660 pair-depth rows. Repeated histories and PV references are not independent observations.

## Direction after this review

Close this retrospective weighting branch. Prioritize diagnostics of the neural value path and target consistency, then predeclare one bounded treatment against the current learned control. Do not automatically repeat head-only training or enlarge the model. Preserve policy outputs explicitly, protect consumed assessment data, and require paired equal-time tests against both H_v1 and H8 relay with the exact-endgame guard. The neural route is promising, but its superior stability or strength has not been demonstrated. See REGRET_NEURAL_NEXT.md for the concrete sequence and historical evidence.

## Execution and recovery

All14 current source-bound methods passed on their first formal execution: four producer-owner, four forensic-owner and six independent checks, with zero failures, errors or skips. Two actual process kills immediately before and after the second case commit preserved raw prefixes of one and two cases. Both resumed two-case fixtures matched the fresh run exactly, including its manifest. The guarded production run completed all five cases with zero Native calls, neural forwards, searches, games or optimizer updates.

Completed API reopening, assessment and verification add no case commits or runtime work and preserve watched durable bytes, sizes, modification times and file inventory. Independent checks also cover public CLI reopening, valid missing-export repair and refusal of rehashed corruption, changed identities and missing required sources. Pure arithmetic and literal replay may repeat during validation.

All5,252 prior files remain unchanged. Models, policy, active weights, reservations and the allocation HOLD remain fixed. Closure evidence is in `build/regret-acceptance.json`, `build/regret-rerun.json` and `build/regret-acceptance-rerun.json`; see [detailed findings](REGRET_FINDINGS.md), [independent audit](REGRET_INDEPENDENT_AUDIT.md) and [next neural experiment](REGRET_NEURAL_NEXT.md). Engineering completion does not make the rejected weighting rule pass its scientific safeguards.

The supported resumable public command is:

```powershell
python -m congklak.regret_progress run --tool regret-resume -- python tools/investigate-regret.py all
```

It resumes derived case reports or validates a completed run. Pure arithmetic and literal replay may repeat during semantic validation; no Native, neural inference, search or training is permitted. The dashboard on local port18766 tracks seven current steps and retains earlier completed scopes. Its final verification uses the backend and OS listener, without browser or HTTP inspection.

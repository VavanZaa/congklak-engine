# Existing-data capture Q-drift investigation

This retrospective diagnostic explains the capture-position contribution to the preceding weighting regression. Models, policy, search settings, existing sampling/loss settings, all data ownership reservations and the allocation HOLD stay unchanged. No new teacher search, label, neural inference, training, game, rollout, alternative-weight fitting, pruning or promotion is allowed. No Native runtime call is needed; replay is limited to the literal Python interpreter on saved action sequences. The preceding 4,920 accepted inputs plus three closing proofs are protected in `build/capture-baseline.json`, 4,923 files in total.

## Fixed support and evidence

Study **all four capture roots** from the sealed 16-root within-cell assessment, two per actor cell. Retain **all 16 roots** when attributing the global uniform-versus-weighted change. Do not choose only large drifts, remove an inconvenient action, introduce new roots or treat these consumed positions as a new validation set. All 16 roots remain covered by the existing reservations; this stage creates no new training eligibility or substitute holdout.

Use only the sealed raw roots, original source wrappers, fixed H8/H10/H12 endpoints, full legal-action Q, consecutive completed histories, saved PVs, frozen comparison probabilities and prior reports. Verify raw state/actor/legal support, row hashes, endpoint hashes, manifests, the immutable H8/H10 prefix and rule linkage against the prior accepted bytes. A direct read-only SQLite reader can use those accepted inputs without rebuilding historical eligibility graphs or activating Native. Preserve original source-root versus study-wrapper identities. H12 remains a finite H_v1 reference, not solved truth or a calibrated label.

Freeze diagnostic definitions, source and tests before formal fixtures; independent main/worker freeze before independent fixtures. This is already-observed, retrospective evidence: no blind selection, new assessment, population or learned-strength claim follows. Engineering fixtures use the first two of these already-consumed four cases, under strict build paths; no fresh labels or model calls occur.

## Exact attribution

Keep the previous frozen probabilities U=1/16 and W=(1/8)r/sum_cell(r), with the original H8/H10 raw stability score. Do not refit scales, recompute weights from H12, remove a screen condition or apply any weights. Retain exact Fraction arithmetic and original group denominators.

For every root/action and each recorded horizon, retain player-0 Q, actor-signed Q, best-set membership, action gaps, strict pairwise ranking reversals and tie changes. Compute signed `delta=Q12-Q10` and absolute per-action drift. Report the H8/H10 shifts that produced r beside subsequent H10/H12 shifts. Best-action stability, numeric gap change and absolute value drift are different observations.

For root i with n legal actions and D_i=mean_legal(abs(delta)), the global drift contribution of action a is `(W_i-U_i)*abs(delta_a)/n`. Sum all actions to root, cell, capture, non-capture and all16 totals and reproduce the sealed global delta exactly. Attribute best-set and non-best actions separately without favorable tie selection. The non-capture contribution is retained even if it offsets capture. No within-capture renormalization may be mistaken for the full-panel contribution.

For each two-root actor cell, verify the covariance identity:

    conditional_weighted_D - conditional_uniform_D
      = (r1-r2)*(D1-D2) / (2*(r1+r2))

Multiplying by the fixed cell mass1/8 yields its global contribution. This is a deterministic algebraic identity on observed data, not a causal effect or statistical estimate of generalization.

Report a translation-versus-action-specific diagnostic without claiming a false additive decomposition. With c=mean_legal(delta):

    D = mean(abs(delta))
    C = abs(c)
    S = D-C >= 0
    E = mean(abs(delta-c))

The exact identity is `D=C+S`; S is the signed-cancellation remainder, not a ranking-error component. E is centered dispersion and generally does not satisfy `D=C+E`. Report actor-relative signed shifts separately. For a stable singleton H10 best action b, the change in its actor-relative gap to action a is `actor_sign*(delta_b-delta_a)`. Preserve whole best sets and tied cases in the implementation and synthetic checks. The real capture roots happen to retain the same singleton best set; this does not make other Q entries accurate.

## Saved-line mechanics and horizon evidence

Literally replay the saved root and all-action PVs at the endpoint horizons and their saved completed histories. Validate legal actions, actual controller/bonus turns, seed conservation, capture seeds, relay pickups, deposits, store changes, passes, terminal state and leaf active seeds. Distinguish a root classified as capture because some action captures from an action that actually captures. Retain immediate action mechanisms alongside later PV events.

Compare same-root/action saved paths by their shared state/action prefix and earliest divergence; retain downstream actor and mechanics, leaf stores/active seeds and residual-horizon ambiguity. Use the saved H12 depth1–12 history as a descriptive trajectory; verify overlapping H8/H10 stage Q/best signatures. Duplicate stage histories and root/chosen-action PVs are not independent samples. Different horizons produce different downstream states; longer PVs, capture timing and leaf store swings are observed associations and cannot isolate a cause of heuristic error without an intervention. Do not evaluate new leaves or infer unrecorded search branches.

## Persistence and independent audit

Commit each complete capture case atomically in a FULL/WAL SQLite journal. Retain source-bound identity, ordered case support and hashes, complete marker and validated exports. Cancellation or process death preserves completed cases; uncommitted derived analysis may repeat. A completed reopen validates evidence before reuse and adds no case commits or Native/model/training runtime work. Pure arithmetic and literal replay may repeat for semantic validation; this is not a claim of zero validation computation. A missing final export may be repaired only after validating its committed journal; corrupt state, altered source or input identity must refuse. Required input evidence may not be silently reconstructed.

The independent preflight covers prior bytes/support, literal/exact algebra and synthetic cancellation/ties, and actual pre/post second-case process kills with committed prefixes1/2, raw SQLite copies, flushed markers and deterministic fresh-versus-resumed semantics. Production checks independently rederive all16 attribution and all4/all-action/saved-history evidence. Final checks cover rehashed corruption, missing-export refusal/repair, completed API/CLI no-work, Native/NN/training prohibition and prior bytes/mtime preservation. The planned current coverage is four producer-owner methods, four forensics-owner methods and six independent methods; count only actually executed, source-bound unique methods.

Helpers must precede the entry invocation. Freeze actual executed sources before formal runs, preserve failed reports/raw logs/sources, and never rewrite accepted prior stage proofs. Keep writer API calls and final input-pin capture serialized. No old-stage test is recounted. Final root closure binds partial group proofs, actual sources, cases, all16 context and literal evidence, proves completed no-work/idempotent seal, and verifies the dashboard backend/local listener without browser or HTTP inspection.

The dashboard shows protected history, existing support, exact attribution, literal saved-line mechanics, recovery, production cases and final audit. Completion of this diagnostic permits a clearer decision about future data or training contracts; it applies no weights and starts no training. Existing reservation-aware admission, equal-time strength tests against both baselines and the endgame guard remain requirements for any later learned candidate.

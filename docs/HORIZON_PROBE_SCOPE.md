# Six-disagreement horizon investigation

This probe investigates exactly the six completed natural-study roots whose H8
and H10 best-pit sets were disjoint. Selection is the sorted state-key list in
the sealed stability report, checked against the original complete label maps.
The original wrappers, five training-candidate and one validation ownerships,
and two follow-up/four fresh roles remain unchanged. Every selected root is now
investigation-only: no ingest, training, model expansion, PUCT, strength games
or engine promotion occurs. All 2,199 baseline inputs are preserved, including
2,197 sealed accepted inputs and completed acceptance/rerun evidence.

The quality model remains fixed at SHA256
`bc1160e99f419b3c1052ea87b28145b622639e6e76a49127c388ab4926d67558`.
Original H8 and H10 label/report items and their original-root hashes are copied
unchanged as references; their historical work is not new probe work.

Each of the six roots receives one requested H12 attempt with a ten-million-node
cap. Then every root receives one requested H14 attempt with a 25-million-node
cap, even if its H12 attempt was rejected. The maximum new production work is
210 million nodes. No automatic root, family, horizon or budget expansion is
allowed. Both horizons use the same oracle-free H_v1 native teacher, full-root
action scores, TT16 MiB, PVS, 512 MiB memory and no wall-clock cutoff.

Every requested attempt retains its entire consecutive completed-iteration
history, depths one through its last completed depth. Histories live inside the
atomic attempt body and retain source requested stage and exact report depth.
They are auxiliary reports, not requested-depth labels. An H14 attempt that
stops after H13 contributes an explicitly completed auxiliary H13 report and
an H14 rejection; it does not contribute an H14 label. Likewise an auxiliary
H12 from H14 does not change the status of an independently rejected H12 attempt.
Only full requested-depth reports with all legal action scores become new labels.

Every report is checked for root identity, completed depth, all action scores,
best moves, coverage, TT/PVS options and physical work. Iteration depths must be
the complete prefix, with monotone cumulative physical work bounded by the
attempt total. H8/H10 overlapping action Q and best-set semantics must match
the old references; any shared depth across new requested attempts must agree.
Values and best moves need not be monotone across different depths. H12/H14
action scores have H11/H13 successor horizons and are root policy preferences,
not labels that assign the root horizon to successor states.

Deeper estimates are not exact game truth. Forensics may compare candidate PV
events and same-depth action-Q trajectories without performing new searches.
Requested completions, budget rejections and auxiliary completed depths remain
separate in all counts. Any paired requested-horizon metric uses the complete
intersection and reports the unpaired denominator explicitly.

SQLite FULL/WAL transactions and an exclusive writer commit each requested
root/stage attempt, its complete history and any requested label together.
Actual pre/post commit hooks support subprocess crash tests. An interrupted
uncommitted root is recomputed; completed requested attempts, including budget
rejections, never rerun. Complete-marker validation precedes export publication;
closed readers verify source identities, unchanged references, all histories,
cross-attempt overlap and table hashes. Completed reruns preserve bytes and
timestamps. Author fixtures use prefixes of the same six fixed roots with the
same H12/H14 budgets, and cannot change the production root selection.

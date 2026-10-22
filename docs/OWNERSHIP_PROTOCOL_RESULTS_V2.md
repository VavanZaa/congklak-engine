The prospective design, models, policy, boundary, statistical scope and existing
ownership holds are unchanged. The supported publication entrypoint is now
`python tools/run-ownership-protocol-v2.py run`, with completed verification via
`python tools/run-ownership-protocol-v2.py verify` and final evidence verification
via `python tools/verify-ownership-protocol-v2.py verify`.

An actual interruption before the v1 final catalog exposed a recovery defect.
The inventory unit originally held integer dictionary keys in memory; after
restart JSON loaded those keys as strings. Sorted serialization then ordered
them differently. The saved and resumed summaries had equal parsed values but
different bytes, so immutable publication correctly refused the mismatch.
The first two v1 kill/resume cases passed; the third did not. All v1 sources,
the original complete publication, freezes, logs and failed fixtures remain.

Version 2 converts every computed work unit to its persisted JSON value before
hashing, committing or aggregating it. It also normalizes report values before
publication. Fresh and resumed paths therefore aggregate the same representation.
This is a publication repair, with a separate schema, baseline, production path,
CLI, observer and final closure. The design and projection algorithms are fixed.

Three fresh actual kill/resume cases must pass on v2 before acceptance. These
tests concern the protocol publication only. Future native handoff/reset and
complete footprint tracing remain unimplemented and unverified; data admission,
new starts, new games, model promotion and further training remain unauthorized.

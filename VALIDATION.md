# Validation of this source snapshot

Prepared on 2026-10-22. Checks ran on Windows using existing local toolchains and installed dependencies, with imports directed at this exported copy. This is not a clean-room dependency-install test.

| Check | Result |
| --- | --- |
| native-build | passed |
| n7-independent-proof | passed |
| n64-independent-proof | passed |
| all-opening-certificates | passed |

The C++ native engine built successfully with MSVC. The independent Python checker accepted all 63 supplied opening-win certificates for n=2 through n=64. This establishes the stated sequential single-round opening result, not a solution for every reachable position or every Congklak convention.

The repository contains selected main source, required small configuration/reference inputs and documentation. Installed dependencies, model weights, caches, private records and bulk research outputs are excluded. Reproduce the commands in README.md after installing the declared dependencies.

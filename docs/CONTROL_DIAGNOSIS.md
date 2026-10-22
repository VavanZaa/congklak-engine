# Retained control: data and mistake diagnosis

The improved policy/control remains the learned candidate. H_v1 stays the default. This diagnosis performs no training, changes no model/search settings and generates no new positions or teacher labels.

## Unique data and sampling

The retained control used 7,669 selected roots from 4,576 families; the later value refit used 7,592 roots from 4,546 families. Each run made 384,000 draws with replacement. Repeated optimizer draws are not additional positions. The warm ancestor had already used the original 8,000 roots; current filters do not erase that exposure.

| Stratum | Control roots | Control recorded-history roots | Refit roots | Refit recorded-history roots | Control draws |
| --- | ---: | ---: | ---: | ---: | ---: |
| 25-49/bonus | 727 | 727 | 719 | 719 | 27644 |
| 25-49/capture | 739 | 739 | 732 | 732 | 27371 |
| 25-49/ordinary | 739 | 324 | 737 | 322 | 27460 |
| 25-49/pass | 4 | 4 | 4 | 4 | 27237 |
| 5-24/bonus | 714 | 714 | 708 | 708 | 27480 |
| 5-24/capture | 730 | 730 | 722 | 722 | 27407 |
| 5-24/ordinary | 717 | 617 | 707 | 608 | 27287 |
| 5-24/pass | 646 | 646 | 621 | 621 | 27460 |
| 50-74/bonus | 732 | 732 | 730 | 730 | 27421 |
| 50-74/capture | 723 | 723 | 721 | 721 | 27407 |
| 50-74/ordinary | 612 | 63 | 611 | 62 | 27484 |
| 75-98/bonus | 66 | 66 | 66 | 66 | 27197 |
| 75-98/capture | 515 | 515 | 509 | 509 | 27509 |
| 75-98/ordinary | 5 | 5 | 5 | 5 | 27636 |

The two strata with at most five roots contain 9 positions but receive 54,873 of 384,000 control draws (14.29%). This is a consequence of intentional uniform-stratum weighting. It signals repetition risk, not proof that the weighting caused a particular loss. Constructed examples are valid positions without a recorded game-history reachability witness; they must not be presented as observed game diversity.

## Development errors by source

These are the same 1,000 reused development roots and approximate H8 targets. Counts below expose coverage and provenance; they are not a fresh release test.

| Stratum | Roots | Control MAE, seeds | Policy regret, seeds | Recorded-history roots / MAE | Constructed roots / MAE |
| --- | ---: | ---: | ---: | --- | --- |
| 25-49/bonus | 94 | 6.599 | 2.574 | 94 / 6.599 | 0 / unavailable |
| 25-49/capture | 94 | 5.659 | 2.617 | 94 / 5.659 | 0 / unavailable |
| 25-49/ordinary | 93 | 4.662 | 5.344 | 50 / 4.865 | 43 / 4.426 |
| 25-49/pass | 1 | 3.743 | 0.000 | 1 / 3.743 | 0 / unavailable |
| 5-24/bonus | 93 | 1.929 | 0.484 | 93 / 1.929 | 0 / unavailable |
| 5-24/capture | 93 | 2.479 | 0.656 | 93 / 2.479 | 0 / unavailable |
| 5-24/ordinary | 93 | 2.024 | 1.129 | 85 / 2.081 | 8 / 1.416 |
| 5-24/pass | 93 | 1.266 | 0.247 | 93 / 1.266 | 0 / unavailable |
| 50-74/bonus | 93 | 7.126 | 4.075 | 93 / 7.126 | 0 / unavailable |
| 50-74/capture | 93 | 8.240 | 3.720 | 93 / 8.240 | 0 / unavailable |
| 50-74/ordinary | 73 | 11.071 | 11.356 | 10 / 8.720 | 63 / 11.445 |
| 75-98/bonus | 4 | 4.973 | 1.500 | 4 / 4.973 | 0 / unavailable |
| 75-98/capture | 83 | 9.356 | 4.289 | 83 / 9.356 | 0 / unavailable |

There is no development support for 75-98/ordinary. Five training roots in that stratum do not establish reliable early-game behavior. Active-counter bands are phase proxies, and the root mechanism classifier is stricter than the reason for one chosen move.

## Existing search decisions

| Mechanism / milliseconds | Measurements | Teacher disagreements | Mean regret | Disagreements with <=5% visits on teacher best |
| --- | ---: | ---: | ---: | ---: |
| ordinary/25 | 32 | 10 | 1.062 | 2 |
| ordinary/100 | 32 | 8 | 0.781 | 4 |
| capture/25 | 32 | 2 | 0.219 | 0 |
| capture/100 | 32 | 3 | 0.281 | 2 |
| bonus/25 | 32 | 8 | 1.031 | 1 |
| bonus/100 | 32 | 7 | 1.000 | 5 |

The 192 H8 search measurements cover 96 roots at two budgets. Low visits on the teacher-preferred move are an allocation symptom; approximate H8 disagreements do not prove a correct alternative or establish causality. Worst examples with states, action estimates, priors and visits are retained in the JSON report.

## Existing arena coverage

The control's 96 existing games contain 1,691 control decisions at 1,296 distinct canonical roots. Of 1,259 decisions outside K4, 49 match the existing 9,000-root H8 corpus (48 development and 1 training); 1,210 have no existing H8 root label. The other 432 decisions are covered by the existing exact K4 oracle. Final losses cannot be assigned to individual non-K4 moves from this evidence alone.

| Root stratum | Observed control decisions |
| --- | ---: |
| 0-4/bonus | 88 |
| 0-4/capture | 6 |
| 0-4/ordinary | 64 |
| 0-4/pass | 274 |
| 25-49/bonus | 82 |
| 25-49/capture | 285 |
| 25-49/ordinary | 15 |
| 5-24/bonus | 202 |
| 5-24/capture | 313 |
| 5-24/ordinary | 158 |
| 5-24/pass | 74 |
| 50-74/bonus | 23 |
| 50-74/capture | 87 |
| 50-74/ordinary | 12 |
| 75-98/capture | 8 |

These arenas use selected starting positions: 72 games start from recorded-history roots and 24 from constructed roots. Of the 12 ordinary 50-74 decisions, 10 are initial constructed-root decisions; only two occur in recorded-history-start games. No 75-98/ordinary root occurs in this selected arena sample. These observations do not estimate its frequency in full games from the standard opening. They argue against choosing a large rare-stratum quota merely from five training examples.

## Endgame and next decision

The retained control has zero regret on all 32 existing exact K4 diagnostic roots at both budgets in both campaigns (128 observations, not 128 independent positions). Additionally, read-only verified oracle lookups certify 432 observed arena choices: 0 nonoptimal and 0 total seed regret. No new oracle was built or integrated into search, and these correlated observations are not an unseen release test. This diagnosis leaves the model and endgame code unchanged. Learned low-active values above K4 remain approximate.

Keep the improved policy/control. First audit repetition weights and the non-K4 label gap: 1,210 observed choices lack an existing H8 root label after checking both training and development. A small diagnostic labeling pilot on saved positions can distinguish policy/value/search errors; those protected evaluation trajectories must not silently become training rows. If the errors point to missing experience, collect new independent game-history families for training and separately owned validation, measuring ordinary-state yield before choosing quotas. Retain useful constructed examples, low-active coverage and the exact regression cases. Any later model change still needs equal-time games against both baselines.

No expansion or retraining is performed here. The corpus is small and uneven, so targeted data expansion is a justified hypothesis; the existing measurements do not prove that more data will improve playing strength. Availability/rarity, teacher horizon error, limited representation and search allocation remain distinct hypotheses. New labels or counterfactual searches would be needed to distinguish causes on currently unlabeled game decisions.

Evidence: `build/control-diagnosis.json`, the unchanged accepted quality/value manifests, H8 development roots, and the existing quality/value case and value arena journals. All prior 1,829 accepted input hashes are checked before and after diagnosis.

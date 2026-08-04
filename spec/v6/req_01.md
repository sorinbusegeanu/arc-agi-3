## Codex instructions

Work on `main`.

Do not redesign unrelated code.
Do not change hypothesis thresholds unless explicitly instructed below.
Keep backward-compatible metric names.
Add regression tests for every fix.

### 1. Fix H07 report

File:

```text
src/v6/hypothesis_h07_report.py
```

#### 1.1 Add required-table validation

Immediately after opening `current_state.sqlite`, load all existing table names from `sqlite_master`.

Require these tables:

```text
role_transfer_attempts
role_candidates
role_links
concept_candidates
higher_order_milestones
```

Treat these as optional:

```text
concept_promotion_state
higher_order_milestone_history
```

When a required table is missing:

* Return `INSUFFICIENT_EVIDENCE`.
* Add a message listing all missing tables.
* Include an empty incremental-promotion-validation report.
* Write all normal H07 output files.
* Do not allow SQLite to raise `OperationalError`.

Add a test where `current_state.sqlite` exists but one required table is absent.

#### 1.2 Use durable concept promotion state

Replace the current direct `concept_candidates` query with a query that left-joins:

```text
concept_promotion_state
```

For each concept, return:

```text
effective_is_promoted
effective_promotion_status
effective_validation_status
```

Use:

```sql
COALESCE(
    concept_promotion_state.currently_promoted,
    concept_candidates.is_promoted,
    0
)
```

for `effective_is_promoted`.

A concept is scientifically promoted only when:

```text
effective_is_promoted == 1
```

and neither effective status indicates:

```text
failed
invalid
demoted
rejected
```

Use persistent values when present. Otherwise fall back to candidate values.

If `concept_promotion_state` does not exist, retain legacy behavior using `concept_candidates`.

Use this effective promoted state everywhere in H07:

```text
promoted_rows
promoted_concept_count
promoted_cross_context_count
promoted_cross_game_count
promoted_cross_game_count_max
promoted_cross_context_count_max
promoted_overconcentrated_concept_count
source_role_count_mean
source_carrier_count_mean
max_source_role_count
max_source_family_count
concept_transfer_success_concentration
VALID/PARTIALLY_VALID decision logic
```

Add tests for:

1. Candidate says promoted, persistent state says demoted.
2. Candidate says not promoted, persistent state says currently promoted.
3. Persistent state says promoted but validation is failed.

#### 1.3 Use one canonical source-role identity

In all H07 role-transfer queries, use:

```sql
COALESCE(source_role_signature, role_signature)
```

This includes:

```text
successful_roles
roles_with_transfer_attempts
roles_with_successful_transfers
```

Change `successful_roles` to select the canonical source role:

```sql
SELECT DISTINCT COALESCE(source_role_signature, role_signature)
FROM role_transfer_attempts
WHERE COALESCE(reuse_success, 0) = 1
```

Ignore rows where both role fields are null.

Use this canonical set when calculating:

```text
roles_skipped_missing_transfer_success
roles_used_for_concepts
roles_eligible_for_concept_derivation
```

Add a test where:

```text
role_signature = target-role
source_role_signature = source-role
```

and only `source-role` has the required family/carrier links.

#### 1.4 Stop mixing global and concept-level transfer populations

Do not calculate:

```text
concept_strong_transfer_success_count
```

by summing concept candidate counters if the same transfer attempt can contribute to several concepts.

Build a deduplicated set of relevant transfer attempt IDs.

Preferred approach:

* Use `attempt_id` from `role_transfer_attempts`.
* Count each attempt once.
* Restrict the population to attempts whose canonical source role is linked to at least one concept candidate.
* If an explicit attempt-to-concept link exists, use it instead of role-level inference.

Calculate:

```text
concept_transfer_attempt_count
concept_transfer_success_count
concept_strong_transfer_success_count
```

from this same deduplicated attempt population.

Strong success should use the same strong-transfer definition used by concept derivation. Do not invent a new threshold. Reuse an existing stored boolean or the existing derivation predicate.

Calculate:

```text
transfer_success_rate =
    concept_strong_transfer_success_count /
    concept_transfer_attempt_count
```

Return `None` when the denominator is zero.

Keep the legacy metric:

```text
transfer_attempt_count
```

as the global count for diagnostics, but do not use it as the denominator for concept-level success.

Add assertions:

```text
concept_strong_transfer_success_count <= concept_transfer_success_count
concept_transfer_success_count <= concept_transfer_attempt_count
```

Add a regression test where one successful attempt is linked to two concepts. It must count once.

---

### 2. Fix H08 single-component evidence leakage

File:

```text
src/v6/hypothesis_h08_report.py
```

#### 2.1 Create per-component validation records

For every world-model component, create one record containing all required H08 evidence:

```text
component_signature
effective_currently_coherent
effective_validation_status
has_positive_heldout_gain
cross_context_count
cross_game_count
supported_context_count
concept_link_count
role_link_count
family_link_count
verified_predicted_outcome_count
max_coherence_score or component coherence_score
explanatory_coverage
candidate_only
```

Use `world_model_component_state` when present.

A component must not qualify if persistent state says:

```text
currently_coherent = 0
```

or validation status is:

```text
failed
invalid
demoted
rejected
```

If the state table is absent, fall back to `world_model_components.is_coherent`.

#### 2.2 Define a single qualifying-component predicate

Add a helper such as:

```python
def _component_passes_h08_validity(...)
```

A component qualifies only when that same component satisfies all component-level gates:

```text
currently coherent
positive held-out gain
cross_context_count >= 3 OR cross_game_count >= 2
role_link_count >= 1
family_link_count >= 2
supported_context_count >= 2
verified_predicted_outcome_count >= 1
coherence_score >= 0.45
explanatory_coverage > 0
not candidate_only
```

Do not combine maxima from different components.

Produce:

```text
qualifying_component_count
qualifying_component_signatures
```

Include a bounded sample of qualifying component records in the JSON report.

#### 2.3 Base VALID on a qualifying component

The H08 `VALID` decision must require:

```text
promoted_concept_count >= 1
role_candidate_count >= 1
role_transfer_success_count >= 1
qualifying_component_count >= 1
```

Remove the use of global maxima as independent validity gates.

Global aggregate metrics may remain for diagnostics, but they must not make the decision valid.

#### 2.4 Correct coherent cross-scope counts

Calculate:

```text
coherent_cross_context_component_count
coherent_cross_game_component_count
```

only from components that:

```text
are effectively currently coherent
have positive held-out gain
```

Do not count structural-only components.

Add separate diagnostic counters if needed:

```text
structural_cross_context_component_count
structural_cross_game_component_count
```

#### 2.5 Filter promoted concepts by validation state

Apply the same durable concept-promotion rules described for H07.

A promoted concept must not count when persistent promotion or validation state says:

```text
failed
invalid
demoted
rejected
```

#### 2.6 Add regression tests

Add tests covering:

1. Component A has cross-scope evidence, component B has prediction evidence, component C has held-out gain. Result must not be `VALID`.
2. One component has every required gate. Result may become `VALID`.
3. Structural coherent component without held-out gain does not count as coherent scientific evidence.
4. Persistent component state overrides `is_coherent`.
5. Demoted or failed concept does not count as promoted.

---

### 3. Fix H09 report

File:

```text
src/v6/hypothesis_h09_report.py
```

#### 3.1 Fix structured-effect counter typo

Replace:

```python
int(source_counts.get("structural_effect", 0))
+ int(source_counts.get("structured_effect", 0))
```

with:

```python
int(source_counts.get("structural_effect", 0))
+ int(source_counts.get("structured_effect", 0))
```

Confirm both spellings are intentionally supported because historical data may contain either value.

Add a test with one event of each spelling. Expected total: `2`.

#### 3.2 Build per-motif scientific evidence records

For every motif, build one record with:

```text
motif_signature
motif_type
is_emergent
provenance_status
has_verified_observation
has_verified_cross_game_observation
has_verified_cross_context_observation
verified_event_count
verified_nonzero_option_delta_event_count
classification_sources
unknown_verified_event_count
verified_event_count
```

Only use observations with:

```text
provenance_status == verified
```

for scientific scope and event evidence.

A cross-game observation must require:

```text
source_game_key and target_game_key present
source_game_key != target_game_key
source_game_is_surrogate == 0
target_game_is_surrogate == 0
```

A cross-context observation must require:

```text
both contexts complete
source_context_key != target_context_key
source_context_is_surrogate == 0
target_context_is_surrogate == 0
```

#### 3.3 Define per-motif H09 qualification

A motif qualifies for H09 scientific evidence only when the same motif satisfies:

```text
is_emergent == 1
provenance_status == verified
motif_type != unknown
has_verified_observation
has_verified_cross_game_observation OR has_verified_cross_context_observation
verified_nonzero_option_delta_event_count >= 1
```

Produce:

```text
qualifying_emergent_motif_count
qualifying_emergent_motif_signatures
```

#### 3.4 Require motif-type diversity inside the qualifying population

Calculate type diversity only from qualifying motifs:

```text
qualifying_motif_type_count
qualifying_motif_type_counts
```

For `VALID`, require:

```text
qualifying_emergent_motif_count >= 1
qualifying_motif_type_count >= 2
```

If the intended scientific threshold is only one motif type, preserve the existing threshold instead. The key requirement is that type diversity must be computed from the same qualifying population.

Do not allow:

* one emergent proxy motif,
* one unrelated verified motif,
* one unrelated cross-scope motif,
* and another event with non-zero delta

to jointly produce `VALID`.

#### 3.5 Separate verified-population unknown ratios

Keep existing all-event ratios for backward compatibility.

Add:

```text
verified_event_count
verified_unknown_event_count
verified_unknown_event_ratio
verified_unknown_source_count
verified_unknown_source_ratio
verified_motif_count
verified_unknown_motif_count
verified_unknown_motif_ratio
```

Use verified-population ratios in the scientific `VALID` decision.

Do not allow proxy, missing-provenance, or surrogate-only events to improve or degrade the verified scientific ratios.

#### 3.6 Add regression tests

Add tests covering:

1. Evidence split across unrelated motifs does not produce `VALID`.
2. One motif satisfying the full chain is counted as qualifying.
3. Surrogate scope does not count.
4. Verified cross-context observation counts.
5. Proxy observation does not count.
6. Unknown ratios are calculated separately for all events and verified events.
7. The structured-effect counter counts both historical spellings once.

---

### 4. Strengthen H11 verification

Files:

```text
src/v6/hypothesis_h11_report.py
src/v6/future_options.py
```

#### 4.1 Replace assertion-only verification with explicit predicates

Create a reusable SQL/Python definition for a verified transfer chain.

A fully verified chain must require:

```text
motif_provenance_status == verified
transfer_provenance_status == verified
concept_validation_status == verified
provenance_mode == single_source
source_role_signature is present
concept_signature is present and not "__none__"
source_game_key is present
target_game_key is present
source_context_key is complete
target_context_key is complete
source_game_is_surrogate == 0
target_game_is_surrogate == 0
source_context_is_surrogate == 0
target_context_is_surrogate == 0
source_game_key != target_game_key
    OR
source_context_key != target_context_key
```

Use this complete predicate in:

```text
full_condition
fully_verified_links
fully_verified_emergent_links
verified motif counters
verified attempt/success counters
verified transfer-pair counters
VALID decision
```

Do not rely on `assert` for correctness.

Assertions may remain as internal consistency checks, but removing them via `python -O` must not change report results.

#### 4.2 Validate contexts explicitly

Use the existing complete-context helper, or add an equivalent helper in H11.

Reject values containing:

```text
null
none
[]
{}
empty string
```

Do not treat surrogate context keys as real scope evidence.

#### 4.3 Fix successful-role-transfer count

Change `successful_role_transfer_count` so it requires:

```text
reuse_success == 1
provenance_mode == single_source
provenance_status == verified
source_role_signature present
real cross-game or cross-context scope
no surrogate scope
```

Use the same real-scope predicate as the verified-chain calculation.

#### 4.4 Filter promoted concepts by durable validation

Change `promoted_concept_count` to exclude concepts with effective status:

```text
failed
invalid
demoted
rejected
```

Use persistent promotion state when present.

#### 4.5 Require pair diversity for H11 VALID

Add a named constant near the report defaults:

```python
MIN_H11_VERIFIED_TRANSFER_PAIRS = 2
```

For `VALID`, require:

```text
fully_verified_emergent_chain_count >= 5
verified_motifs_with_strong_transfer_count >= 1
verified_motifs_with_promoted_concept_count >= 1
emergent_motif_strong_transfer_success_rate > 0
verified_transfer_pair_count >= MIN_H11_VERIFIED_TRANSFER_PAIRS
```

The pair identity must include:

```text
source_game
target_game
source_context
target_context
```

Count only fully verified, non-surrogate, real cross-scope pairs.

Add to output:

```text
minimum_verified_transfer_pairs_required
verified_transfer_pair_diversity_gate_passed
```

If all other conditions pass but pair diversity fails, return:

```text
PARTIALLY_VALID
```

and add a specific missing-evidence message.

#### 4.6 Do not crash on malformed old rows

Malformed legacy rows must be classified as:

```text
proxy
missing
unverified
```

They must not raise assertions during normal reporting.

Keep counters for malformed rows:

```text
verified_status_but_invalid_scope_count
verified_status_but_missing_identity_count
verified_status_but_surrogate_scope_count
```

#### 4.7 Add H11 regression tests

Add tests covering:

1. Status fields say verified but `provenance_mode` is not `single_source`.
2. Status fields say verified but context is incomplete.
3. Status fields say verified but scope is surrogate.
4. Same game and same context do not count as transfer.
5. Real cross-context transfer within one game counts.
6. Real cross-game transfer counts.
7. Five verified rows from one pair do not make H11 `VALID`.
8. Five verified rows covering at least two concrete pairs can satisfy pair diversity.
9. Running tests with assertions disabled produces the same report decision.

---

### 5. Run tests

Run the focused tests first:

```bash
PYTHONPATH=src pytest -q \
  tests/test_h07_concept_emergence.py \
  tests/test_h08_world_model_coherence.py \
  tests/test_h09_future_option_motifs.py \
  tests/test_h11_future_option_transfer.py
```

Use the actual existing test filenames if they differ.

Then run all V6 hypothesis and future-option tests:

```bash
PYTHONPATH=src pytest -q tests -k "h07 or h08 or h09 or h11 or future_option or world_model or concept"
```

Then run the complete test suite:

```bash
PYTHONPATH=src pytest -q
```

Also run the relevant tests with assertions disabled:

```bash
PYTHONOPTIMIZE=1 PYTHONPATH=src pytest -q tests -k "h11"
```

### 6. Final response requirements

Report:

```text
files changed
tests added
tests run
test results
remaining known limitations
```

Do not claim completion if any focused test fails.

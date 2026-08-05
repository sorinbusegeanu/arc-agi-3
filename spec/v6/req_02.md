Use this prompt with Ornith-9B. It is deliberately narrow, sequential, and explicit.

You are fixing one broken commit in repository:

sorinbusegeanu/arc-agi-3

Work on branch:

main

The bad commit is:

2fff7374e2bf7507ba93d058b18e0cc13596d239

The previous good commit is:

8c4e3c295fb6a35e1931bf95a096f70c4d621fad

Your task is not to redesign the reports.

Your task is:

1. Restore the four report files from the previous good commit.
2. Reapply only the requested fixes.
3. Preserve all existing report fields, function signatures, artifacts, and behavior unless explicitly changed below.
4. Do not add packaging files, documentation files, summaries, specifications, egg-info files, or unrelated tests.
5. Do not rewrite whole files.
6. Make small local edits.
7. Do not change anything else.

Files allowed to change:

src/v6/hypothesis_h07_report.py
src/v6/hypothesis_h08_report.py
src/v6/hypothesis_h09_report.py
src/v6/hypothesis_h11_report.py
src/v6/tests/test_v6_hypothesis_reports.py

Do not change:

src/v6/future_options.py
pyproject.toml
src/v6/pyproject.toml
docs/
spec/
src/arc_agi_v6.egg-info/
any other file

First remove the unrelated files introduced by the bad commit:

docs/v6/summary.md
pyproject.toml
spec/v6/req_01.md
src/v6/pyproject.toml
src/arc_agi_v6.egg-info/PKG-INFO
src/arc_agi_v6.egg-info/SOURCES.txt
src/arc_agi_v6.egg-info/dependency_links.txt
src/arc_agi_v6.egg-info/top_level.txt
src/v6/tests/test_future_options.py

Do not remove an existing root pyproject.toml if it already existed before commit 2fff7374. Compare with commit 8c4e3c2 first.

Then restore these four files exactly from commit 8c4e3c2 before editing:

src/v6/hypothesis_h07_report.py
src/v6/hypothesis_h08_report.py
src/v6/hypothesis_h09_report.py
src/v6/hypothesis_h11_report.py

After restoration, apply the changes below one file at a time.

==================================================
PART 1: H07
===========

File:

src/v6/hypothesis_h07_report.py

Do not change the function signature of:

evaluate_h07_concept_emergence

It must still accept:

memory_dir
run_dir
output_dir
already_derived
incremental_promotion_validation

Do not move imports inside functions.

Do not remove existing incremental promotion reporting.

---

## H07 CHANGE 1: required tables

Inside evaluate_h07_concept_emergence, immediately after:

conn.row_factory = sqlite3.Row

load all table names:

tables = {
str(row[0])
for row in conn.execute(
"SELECT name FROM sqlite_master WHERE type='table'"
).fetchall()
}

Required tables:

role_transfer_attempts
role_candidates
role_links
concept_candidates
higher_order_milestones

Compute:

missing_tables = [
name
for name in required_tables
if name not in tables
]

If missing_tables is not empty:

* create an H07 result with decision INSUFFICIENT_EVIDENCE
* include one missing_evidence message containing all missing table names
* include the normal incremental_promotion_validation result
* write normal H07 outputs
* return immediately

Do not query missing tables before this check.

Optional tables:

concept_promotion_state
higher_order_milestone_history

Do not require optional tables.

---

## H07 CHANGE 2: durable promotion

Keep the existing concept candidate query fields.

If concept_promotion_state exists, use this query shape:

SELECT
candidate.concept_signature,
candidate.compression_gain,
candidate.promotion_score,
candidate.transfer_success_count,
candidate.strong_transfer_success_count,
candidate.linked_role_count,
candidate.linked_carrier_count,
candidate.linked_family_count,
candidate.cross_context_count,
candidate.cross_game_count,
candidate.is_promoted AS candidate_is_promoted,
candidate.transfer_success_concentration,
candidate.is_overconcentrated,
persistent.currently_promoted AS persistent_currently_promoted,
persistent.promotion_status AS persistent_promotion_status,
persistent.validation_status AS persistent_validation_status
FROM concept_candidates AS candidate
LEFT JOIN concept_promotion_state AS persistent
ON persistent.concept_signature = candidate.concept_signature
ORDER BY candidate.concept_signature ASC

If concept_promotion_state does not exist, use the original concept_candidates query and add Python fallback fields:

persistent_currently_promoted = None
persistent_promotion_status = None
persistent_validation_status = None

Create one helper:

def _effective_concept_is_promoted(row: sqlite3.Row | dict[str, Any]) -> bool:

Its exact logic must be:

1. Read persistent_currently_promoted.
2. If persistent_currently_promoted is not None, use int(persistent_currently_promoted) == 1.
3. Otherwise use int(candidate_is_promoted or is_promoted or 0) == 1.
4. Read persistent_promotion_status and persistent_validation_status.
5. Normalize both with str(... or "").strip().lower().
6. Return False if either status is one of:

failed
invalid
demoted
rejected

7. Otherwise return the effective promoted boolean.

Important:

Do not use:

persistent_value or candidate_value

because persistent zero must override candidate one.

Replace all promoted-row filtering with:

promoted_rows = [
row
for row in concept_rows
if _effective_concept_is_promoted(row)
]

Use promoted_rows consistently for all promoted metrics and decision gates.

---

## H07 CHANGE 3: canonical role

Change successful_roles query to:

SELECT DISTINCT
COALESCE(source_role_signature, role_signature)
AS canonical_source_role
FROM role_transfer_attempts
WHERE COALESCE(reuse_success, 0) = 1
AND COALESCE(source_role_signature, role_signature) IS NOT NULL

Build successful_roles using:

str(row["canonical_source_role"])

Do not access row["role_signature"] for this query.

Use the same expression:

COALESCE(source_role_signature, role_signature)

for:

roles_with_transfer_attempts
roles_with_successful_transfers

Do not change role_links identity. role_links.role_signature must be compared to the canonical source role values.

---

## H07 CHANGE 4: transfer counts

Do not invent a new role-count denominator.

Keep:

transfer_attempt_count

as the total number of rows in role_transfer_attempts.

Create:

deduplicated_transfer_attempt_count
deduplicated_transfer_success_count

Use attempt_id if the column exists.

Check columns using:

PRAGMA table_info(role_transfer_attempts)

If attempt_id exists:

deduplicated_transfer_attempt_count =
COUNT(DISTINCT attempt_id)

deduplicated_transfer_success_count =
COUNT(DISTINCT CASE
WHEN COALESCE(reuse_success, 0) = 1
THEN attempt_id
END)

If attempt_id does not exist:

fall back to row counts.

Do not divide by roles_with_transfer_attempts.

Set:

concept_transfer_success_count =
deduplicated_transfer_success_count

For now preserve the existing concept_strong_transfer_success_count calculation from concept rows.

Do not claim that strong successes are deduplicated.

Set:

transfer_success_rate =
deduplicated_transfer_success_count /
deduplicated_transfer_attempt_count

Return None when denominator is zero.

Do not add assertions that compare concept strong counts against deduplicated attempts because concept strong counts may currently be aggregated differently.

---

## H07 TESTS

Add focused tests to existing file:

src/v6/tests/test_v6_hypothesis_reports.py

Add tests for:

1. Missing required H07 table returns INSUFFICIENT_EVIDENCE.
2. Persistent currently_promoted=0 overrides candidate is_promoted=1.
3. Persistent currently_promoted=1 overrides candidate is_promoted=0.
4. Persistent validation_status=demoted prevents promotion.
5. Canonical source_role_signature is used instead of target role_signature.
6. attempt_id duplicates are counted once.

Do not create a new test file.

Run only H07 tests before moving to H08.

==================================================
PART 2: H08
===========

File:

src/v6/hypothesis_h08_report.py

Restore the original file first.

Do not use sqlite3.Row.get().

Convert rows to dictionaries explicitly when dictionary access is needed:

component_rows = [
dict(row)
for row in conn.execute(...).fetchall()
]

Do not query the database after leaving the with sqlite3.connect block.

---

## H08 CHANGE 1: required tables

Require:

concept_candidates
world_model_components
world_model_links
higher_order_milestones

If missing, return INSUFFICIENT_EVIDENCE before executing report queries.

Optional:

concept_promotion_state
world_model_component_state
role_candidates
role_transfer_attempts

---

## H08 CHANGE 2: durable concepts

Use the same durable promotion logic as H07.

Do not count failed, invalid, demoted, or rejected persistent states.

Do not duplicate this logic inside the component loop.

Compute promoted_concept_count once.

---

## H08 CHANGE 3: component state map

If world_model_component_state exists, load:

SELECT
component_signature,
historically_coherent,
currently_coherent,
validation_status
FROM world_model_component_state

Build:

component_state_by_signature

For each component:

effective_currently_coherent is:

* persistent currently_coherent when present
* otherwise component is_coherent

Reject component qualification if validation_status normalized is one of:

failed
invalid
demoted
rejected

---

## H08 CHANGE 4: per-component evidence

Keep the original world_model_components SELECT.

Do not remove original fields.

Keep original world_model_links loading.

Build the existing link_map:

link_map[component_signature][linked_type] = set(linked_key)

For each component create a dictionary containing:

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
coherence_score
explanatory_coverage
candidate_only

Compute linked counts from link_map for that same component:

supported_context_count =
len(link_map[signature].get("context", set()))

concept_link_count =
len(link_map[signature].get("concept", set()))

role_link_count =
len(link_map[signature].get("role", set()))

family_link_count should use the existing linked_family_count field, or world_model_family_links if the original code already uses it.

Do not assign global promoted concept count to each component.

has_positive_heldout_gain must use the original helper logic:

positive if any one of these is greater than zero:

heldout_prediction_gain
validation_action_selection_lift
validation_transfer_lift
validation_contradiction_resolution
validation_explanatory_gain

Do not require both heldout_prediction_gain and validation_prediction_lift.

verified_predicted_outcome_count for a component is:

predicted_outcome_count

only when:

prediction_evidence_status == "verified"

Otherwise zero.

---

## H08 CHANGE 5: qualifying predicate

Create module-level helper:

def _component_passes_h08_validity(record: dict[str, Any]) -> bool:

Return True only if all are true:

effective_currently_coherent is True

effective_validation_status is not:
failed
invalid
demoted
rejected

has_positive_heldout_gain is True

cross_context_count >= 3
OR
cross_game_count >= 2

role_link_count >= 1

family_link_count >= 2

supported_context_count >= 2

verified_predicted_outcome_count >= 1

coherence_score >= 0.45

explanatory_coverage > 0

candidate_only is False

Do not require concept_link_count unless the original hypothesis explicitly required component-level concept links.

Compute:

qualifying_component_records
qualifying_component_count
qualifying_component_signatures

---

## H08 CHANGE 6: decision

H08 VALID must require:

promoted_concept_count >= 1
role_candidate_count >= 1
role_transfer_success_count >= 1
qualifying_component_count >= 1

Do not add any additional global max or total gates after qualifying_component_count.

Specifically remove from the VALID condition any separate checks using:

max_coherence_score
max_explanatory_coverage
global cross_context maximum
global cross_game maximum
global role link total
global family link total
global verified prediction total

Those may remain diagnostic metrics only.

---

## H08 CHANGE 7: coherent counters

coherent_cross_context_component_count must count components, not sum their context counts.

Use:

sum(
1
for record in component_records
if record["effective_currently_coherent"]
and record["has_positive_heldout_gain"]
and record["cross_context_count"] >= 1
)

Do the equivalent for cross_game.

---

## H08 TESTS

Add tests for:

1. Evidence split across three components does not produce VALID.
2. One component satisfying every component gate can produce VALID.
3. Structural coherent component without positive held-out gain does not qualify.
4. Persistent currently_coherent=0 overrides is_coherent=1.
5. Failed component validation prevents qualification.
6. Demoted concept does not count as promoted.
7. coherent cross-context metric counts components, not scope total.

Run H08 tests before moving to H09.

==================================================
PART 3: H09
===========

File:

src/v6/hypothesis_h09_report.py

Restore the original file first.

Preserve all existing output fields and artifacts.

Do not remove:

core_metrics
missing_evidence
evidence_diagnostics
h09_motif_observations.jsonl
classification counters
derivation summary counters

---

## H09 CHANGE 1: structured counter

Keep source_counts built once from events.

Do not build source_counts inside the motif loop.

Use:

source_counts = Counter(
str(row.get("classification_source") or "unknown")
for row in events
)

Set:

structured_effect_event_count =
int(source_counts.get("structural_effect", 0))
+
int(source_counts.get("structured_effect", 0))

Do not change live_delta_event_count.

live_delta_event_count must remain based on:

live_delta
live_delta_rule

---

## H09 CHANGE 2: index events and observations

Before the motif loop build:

events_by_event_id

observations_by_motif

verified_observations_by_motif

For every observation:

motif_signature = str(observation["motif_signature"])

Append only observations whose provenance_status is verified to verified_observations_by_motif.

For each motif, only inspect observations associated with that motif.

---

## H09 CHANGE 3: verified cross-scope

For a verified observation, cross-game is true only if:

source_game_key is not None or empty
target_game_key is not None or empty
source_game_key != target_game_key
source_game_is_surrogate == 0
target_game_is_surrogate == 0

Cross-context is true only if:

_complete_context_key(source_context_key)
_complete_context_key(target_context_key)
source_context_key != target_context_key
source_context_is_surrogate == 0
target_context_is_surrogate == 0

Do not compare context_key to game_key.

---

## H09 CHANGE 4: motif-specific event evidence

For each motif, collect event IDs from its observations.

Then obtain only those events from events_by_event_id.

Compute for that motif:

verified_event_count

verified_nonzero_option_delta_event_count

unknown_verified_event_count

classification_sources

A motif event is verified when:

classification_provenance_status == "verified"

or, if that field is absent, the associated verified observation proves the event.

Do not use global non-zero option delta events to qualify a motif.

---

## H09 CHANGE 5: motif record

Create one record per motif:

motif_signature
motif_type
is_emergent
provenance_status
has_verified_observation
has_verified_cross_game_observation
has_verified_cross_context_observation
verified_event_count
verified_nonzero_option_delta_event_count
unknown_verified_event_count
classification_sources

A motif qualifies only if the same motif satisfies:

is_emergent == 1
provenance_status == "verified"
motif_type != "unknown"
has_verified_observation
has_verified_cross_game_observation
OR has_verified_cross_context_observation
verified_nonzero_option_delta_event_count >= 1

Compute:

qualifying_emergent_motifs
qualifying_emergent_motif_count
qualifying_emergent_motif_signatures
qualifying_motif_type_counts
qualifying_motif_type_count

---

## H09 CHANGE 6: verified unknown ratios

Compute global verified event metrics directly from events once.

Do not copy global counts into every motif record and sum them.

Use:

verified_events = [
event
for event in events
if classification_provenance_status == "verified"
]

verified_event_count = len(verified_events)

verified_unknown_event_count =
count verified events where motif_type or classification_source is unknown

verified_unknown_event_ratio =
verified_unknown_event_count / verified_event_count

Return None when denominator is zero.

Keep original all-event ratios too.

---

## H09 CHANGE 7: decision

Preserve original non-H09 branches for:

no events
events but no motifs
motifs but no emergent motifs

For VALID require:

qualifying_emergent_motif_count >= 1

qualifying_motif_type_count >= 2

verified_unknown_event_ratio is None
OR
verified_unknown_event_ratio <= 0.20

Do not combine evidence from unrelated motifs.

---

## H09 TESTS

Add tests for:

1. Source counters count events once.
2. Both structural spellings are included in structured_effect_event_count.
3. live_delta_event_count is not modified.
4. Cross-context compares source context to target context.
5. Evidence split across unrelated motifs does not produce VALID.
6. Non-zero delta from another motif does not qualify a motif.
7. Verified event totals are not multiplied by motif count.
8. Surrogate scope does not qualify.
9. Observation artifact is still written.
10. Existing core_metrics and missing_evidence fields remain present.

Run H09 tests before moving to H11.

==================================================
PART 4: H11
===========

File:

src/v6/hypothesis_h11_report.py

Restore the original file first.

Preserve all existing artifacts and report fields.

Add near existing constants:

MIN_H11_VERIFIED_TRANSFER_PAIRS = 2

---

## H11 CHANGE 1: complete context helper

Update the existing context validation helper.

A context is incomplete when normalized text is:

empty
[]
{}

or contains:

null
none

Return None for incomplete contexts.

---

## H11 CHANGE 2: explicit Python predicate

Replace the weak _is_fully_verified helper with exact logic.

It must return False unless all are true:

motif_provenance_status == "verified"

transfer_provenance_status == "verified"

concept_validation_status == "verified"

provenance_mode == "single_source"

source_role_signature or role_signature is present

concept_signature is present

concept_signature != "**none**"

source_game_key is present

target_game_key is present

source_context_key is complete

target_context_key is complete

source_game_is_surrogate == 0

target_game_is_surrogate == 0

source_context_is_surrogate == 0

target_context_is_surrogate == 0

and one real scope difference exists:

source_game_key != target_game_key
OR
source_context_key != target_context_key

Do not use assert for this decision.

---

## H11 CHANGE 3: SQL full condition

Make full_condition match the Python predicate as closely as SQLite allows.

Require:

all three statuses verified

provenance_mode single_source

COALESCE(source_role_signature, role_signature) is not null and not empty

concept_signature is not null
concept_signature != ''
concept_signature != '**none**'

source_game_key and target_game_key present

source_context_key and target_context_key present

all surrogate flags zero

and:

source_game_key != target_game_key
OR
source_context_key != target_context_key

For serialized incomplete context values, filter them again with the Python predicate after row loading.

Do not treat SQL status fields alone as full verification.

---

## H11 CHANGE 4: successful transfer count

Inspect PRAGMA table_info(role_transfer_attempts).

Use only columns that actually exist.

successful_role_transfer_count must require:

reuse_success == 1
provenance_mode == "single_source"
provenance_status == "verified" when that column exists
canonical source role present
all available surrogate flags zero
real cross-game or cross-context difference

If role_transfer_attempts does not have provenance_status:

do not fabricate verification

use the existing transfer provenance field if available

otherwise keep a separate diagnostic count and do not use it as fully verified evidence

---

## H11 CHANGE 5: durable promoted concepts

Use the same exact durable promotion helper as H07.

Persistent zero overrides candidate one.

Exclude:

failed
invalid
demoted
rejected

---

## H11 CHANGE 6: pair diversity

For each row that passes _is_fully_verified, create pair identity from:

source_game_key
target_game_key
source_context_id
target_context_id

Count unique pair IDs.

Set:

verified_transfer_pair_count

minimum_verified_transfer_pairs_required =
MIN_H11_VERIFIED_TRANSFER_PAIRS

verified_transfer_pair_diversity_gate_passed =
verified_transfer_pair_count >= MIN_H11_VERIFIED_TRANSFER_PAIRS

H11 VALID must require:

existing original VALID conditions
AND
verified_transfer_pair_diversity_gate_passed

If every other VALID condition passes but pair diversity fails:

decision = "PARTIALLY_VALID"

Append missing evidence:

"Insufficient diversity of fully verified transfer pairs."

---

## H11 CHANGE 7: malformed verified rows

Add counters:

verified_status_but_invalid_scope_count
verified_status_but_missing_identity_count
verified_status_but_surrogate_scope_count

Definitions:

verified_status_but_invalid_scope_count:
three status fields are verified, but game and context are both unchanged

verified_status_but_missing_identity_count:
three status fields are verified, but source role or concept identity is missing

verified_status_but_surrogate_scope_count:
three status fields are verified, but any surrogate flag is one

Malformed rows must not crash.

Do not assert that malformed verified rows are impossible.

---

## H11 TESTS

Add tests for:

1. Three verified statuses but provenance_mode not single_source.
2. Missing source role.
3. Missing concept.
4. Concept "**none**".
5. Incomplete context "[]".
6. Incomplete context "{}".
7. Context containing null.
8. Surrogate source context.
9. Same game and same context.
10. Same game but different real contexts.
11. Different games.
12. Five rows from one pair cannot produce VALID.
13. Five rows from two pairs can pass pair diversity.
14. Failed promoted concept is excluded.
15. Report decision does not depend on Python assert statements.

==================================================
FINAL CLEANUP
=============

Before finishing, search all four files for:

row.get(

If row is sqlite3.Row, replace it with one of:

dict(row).get(...)

or:

row["column"]

Search for variables that are used but not defined.

Search for database connection objects used outside their with block.

Search for changed public function signatures.

Search for removed output fields.

Search for removed output artifacts.

Search for imports inside report functions.

Search for duplicate pyproject.toml files introduced by this task.

Do not leave explanatory comments such as:

"1.1"
"section 3.2"
"same as H07"
"requested fix"

Use normal concise code comments only where needed.

Run:

python -m py_compile 
src/v6/hypothesis_h07_report.py 
src/v6/hypothesis_h08_report.py 
src/v6/hypothesis_h09_report.py 
src/v6/hypothesis_h11_report.py

Then run:

PYTHONPATH=src pytest -q src/v6/tests/test_v6_hypothesis_reports.py

If the repository uses tests/ instead of src/v6/tests/, use the existing test location.

Then run:

PYTHONPATH=src pytest -q -k "h07 or h08 or h09 or h11"

Do not commit generated files.

Do not change anything else.

# Repository Development Rules

## Architecture
- Keep one authoritative implementation per responsibility.
- Prefer clear modules and explicit composition over runtime patching or installation order.
- Version behavior through Git history and design documents, not production filenames.

## Changes
Before changing code:
1. Read the affected implementation and its callers.
2. Search for existing fixes, overrides, wrappers, duplicate implementations, and relevant regression tests.
3. Identify the current authoritative runtime path.

Modify or refactor that implementation instead of adding another layer.

## New Files
Create a production module only for a genuinely new architectural responsibility.

Do not create new production files merely for:
- bug fixes or behavior corrections;
- performance fixes;
- parameter or lifecycle changes;
- another implementation version;
- fixups, overrides, or compatibility patches.

Avoid names such as `*_fix.py`, `*_fixups.py`, and version-suffixed implementation files when an existing module owns the responsibility.

## Refactoring
When touching code with historical patches, duplicated logic, or multiple implementations:
- consolidate behavior into the owning module;
- remove obsolete layers when safe;
- reduce architectural debt rather than extending it.

Refactoring directly required for a correct change is part of the task.

## Runtime Composition
Do not introduce new monkey patches or installer chains that replace functions, methods, classes, or module globals at runtime.

Runtime behavior should be understandable from normal imports, explicit dependencies, and composition.

## Regression Safety
Every bug fix must include a regression test reproducing the failure.

Preserve regression tests for historical bugs during refactoring. Do not remove tests simply because the implementation changed.

## Duplication
Search before implementing. If equivalent functionality exists, reuse, generalize, or consolidate it instead of creating a parallel implementation.

## Completion Criteria
A change is complete when:
- the changed behavior has one clear owner;
- no unnecessary patch/version module was added;
- relevant regression coverage exists;
- obsolete code introduced by previous fixes is removed when safe;
- existing tests pass;
- runtime behavior does not depend on new patch ordering.

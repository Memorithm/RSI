# RSI × COGNO-1 × SciRust — autonomous engineering evolution program

Status: **authoritative execution plan**

This document is the cross-repository source of truth for turning RSI's current empirical DGM loop into a cumulative, multi-file, cross-repository engineering engine for SciRust and SciAgent, with COGNO-1 as the hard admissibility authority and FLAT-ATTENTION as the first real optimization domain.

The program is intentionally incremental. Every implementation unit is a reviewable pull request with explicit mechanical acceptance criteria. A phase does not start until its predecessor is merged and green on its final head SHA.

## 1. Scope and repositories

Primary repositories:

- `Memorithm/RSI` — orchestration, DGM, COGNO-1 oracle/backend, flywheel and AUTOPILOT.
- `Memorithm/scirust` — canonical `scirust-rsi`, SciAgent, GPU/runtime substrate, resident KV cache and integration surface.
- `Memorithm/FLAT-ATTENTION` — Rust/WGSL fused attention research and qualification target.

Repositories are added to the program only when a concrete dependency is discovered. Cross-repository changes must preserve a known-good compatible set of revisions.

COGNO-1 currently lives in the RSI workspace as `crates/cogno-core` and `crates/cogno-scirust`; it remains an independent hard-gate authority even when consumed from other repositories.

## 2. Non-negotiable operating rules

1. **One implementation PR at a time in dependency order.** No dependent PR starts from an unmerged contract.
2. **Final-head CI rule.** A PR may merge only when all required checks for its exact final head SHA are successful. A stale earlier green run does not count.
3. **No direct main/master mutation.** All code changes go through branches and PRs.
4. **No performance claim without measurement.** Benchmarks must report methodology, shape, device/backend, warm-up and statistic.
5. **Hard constraints are not scalar penalties.** Compilation, required tests, numerical parity, provenance, resource ceilings and forbidden effects are admissibility gates before ranking.
6. **The proposer never controls its judge.** Candidate code cannot modify its own frozen tests, allowlist, budget, evaluator, COGNO gate or benchmark definition within the same task.
7. **Reproducibility first.** Seeds, relevant revisions, features, toolchain/backend and benchmark configuration are recorded with each accepted lineage step.
8. **Cross-repo compatibility is explicit.** Accepted states record exact repository revisions rather than relying on moving branch names.
9. **Safe failure.** An inconclusive evaluator, missing benchmark score, unavailable device gate or malformed proposal cannot become an improvement.
10. **No silent scope expansion.** Discoveries that require another repository are added to the compatibility manifest and receive their own ordered PR.

## 3. Target architecture

```text
human objective / frozen engineering spec
                 |
                 v
            SciAgent proposer
                 |
                 v
      RSI cumulative engineering DGM
      - PatchSet / file operations
      - materialized parent lineage
      - deterministic archive
                 |
                 v
           COGNO hard gate
      - build / required tests
      - numerical oracle / parity
      - provenance / policy
      - resource ceilings
      - deterministic constraints
                 |
          admissible only
                 v
       empirical benchmark ranking
      - latency / throughput
      - VRAM / memory traffic proxies
      - prefill / decode
                 |
                 v
      accepted state / PR / review
                 |
                 v
              flywheel
      - spec + repository state
      - proposal + PatchSet
      - compiler/test/device output
      - benchmark evidence
      - CI and human verdict
                 |
                 v
        SciAgent/world-model update
```

## 4. Program milestones and ordered PR sequence

### P0 — establish source of truth and compatibility discipline

Goal: make the program auditable before changing engine semantics.

#### P0.1 — RSI: publish this program

Repository: `Memorithm/RSI`

Deliverables:

- this document;
- a tracking issue linking the ordered PR sequence;
- explicit final-head CI/merge rule.

Done when:

- CI is green on final PR head;
- PR is merged;
- tracking issue points to the merged plan.

#### P0.2 — RSI: compatibility manifest model

Add a small versioned representation of a cross-repository compatible state:

```text
RepositoryRevision {
  repository,
  revision,
  role,
}
CompatibilitySet {
  revisions[],
  toolchain,
  feature_contract,
}
```

Requirements:

- deterministic serialization;
- reject empty/malformed revisions;
- no network access in the core type;
- unit tests for stable round trips and ordering.

This becomes metadata for all later cross-repo trajectories.

### P1 — make `Memorithm/scirust/scirust-rsi` canonical

Problem: RSI currently carries a local `scirust-rsi` implementation while SciRust contains a richer canonical engine. Two independently evolving implementations create semantic drift.

#### P1.1 — SciRust: freeze canonical public contract

Repository: `Memorithm/scirust`

Tasks:

- inventory the `scirust-rsi` API consumed by RSI;
- add contract tests for `Guard`, `Report::is_monotone`, `SelfRefiner` and deterministic seeded execution;
- add any minimal compatibility façade needed by RSI without regressing the richer SciRust implementation;
- document the canonical ownership rule.

Done when:

- `scirust-rsi` tests pass;
- workspace gates required by SciRust pass;
- no RSI migration has yet occurred.

#### P1.2 — RSI: consume canonical upstream `scirust-rsi`

Repository: `Memorithm/RSI`

Tasks:

- replace the local duplicate implementation with the exact reviewed SciRust revision;
- update `SCIRUST_ACTIVATION.md` to match the actual dependency graph;
- adapt the bridge only where required;
- remove or clearly retire duplicate local code;
- test default and `scirust` feature modes.

Done when:

- RSI `scirust` feature compiles against the pinned canonical SciRust revision;
- canonical monotonicity semantics match SciRust's best-so-far history definition;
- all required RSI CI is green.

### P2 — replace single-file patches with atomic PatchSets

Problem: the current DGM `Patch { target, find, replace }` can express only one substitution in one existing file.

#### P2.1 — RSI: PatchSet core

Introduce deterministic file operations:

```text
FileOperation::ModifyExact
FileOperation::Create
FileOperation::Delete
FileOperation::Rename   (only if implementation can preserve safety simply)
PatchSet { operations[] }
```

Required invariants:

- paths are workspace-relative and normalized;
- no absolute path or `..` escape;
- duplicate/conflicting operations on the same path are rejected unless explicitly legal;
- every PatchSet is atomic in a candidate snapshot;
- `Create` refuses overwrite;
- `Delete` verifies expected content/hash before removal;
- `ModifyExact` remains unique and non-ambiguous;
- stable deterministic PatchSet identity.

Compatibility:

- preserve a conversion path for the old single `Patch` API during migration;
- no change to live-tree promotion semantics in this PR.

Done when:

- focused unit/property tests cover traversal, conflicts, ambiguous match, create/delete and deterministic IDs;
- existing DGM tests remain green.

#### P2.2 — RSI: PatchSet-aware proposer and trajectory schema

Tasks:

- extend the proposer envelope to multiple operations without allowing unbounded edits;
- add explicit operation count and total-change budgets;
- export PatchSets in trajectories;
- keep backward decoding for existing single-patch trajectory data where practical.

Done when:

- malformed multi-file outputs fail closed;
- allowlist applies to every operation;
- flywheel records the complete candidate change.

### P3 — materialize real cumulative lineages

Problem: the archive currently records parent IDs, but candidate evaluation is reconstructed from the live baseline plus only the current patch. A true engineering lineage must evaluate `parent state + child PatchSet`.

#### P3.1 — RSI: immutable materialized candidate state

Add a `CandidateState` abstraction with:

- parent state ID;
- PatchSet ID;
- deterministic state/tree identity;
- materialization into an isolated evaluation workspace;
- bounded storage/cleanup policy.

The implementation may use deterministic snapshot composition or Git worktree/tree mechanics, but the core contract must not depend on wall-clock identifiers.

Done when tests prove:

```text
baseline + A + B != baseline + B
```

and a grandchild sees all accepted ancestor changes.

#### P3.2 — RSI: archive and promotion follow cumulative state

Tasks:

- parent selection operates on materializable accepted states;
- evaluation applies the child PatchSet to the chosen parent state;
- best candidate promotion reproduces the complete accepted state, not one delta from baseline;
- backups/rollback remain explicit;
- archive serialization preserves lineage reproducibly.

Chaos tests must prove:

- rejected descendants do not mutate accepted ancestors;
- promotion cannot omit ancestor edits;
- stale live-tree changes cause a safe conflict rather than silent overwrite.

### P4 — make COGNO-1 the DGM admissibility authority

#### P4.1 — RSI/COGNO-1: engineering admissibility contract

Add typed gate evidence for engineering candidates, separated from ranking metrics:

```text
EngineeringAdmissibility {
  build,
  required_tests,
  numerical_parity,
  provenance,
  deterministic_contract,
  resource_budget,
  policy_checks,
}
```

Rules:

- any required hard check false => inadmissible;
- unknown required check => inadmissible;
- benchmark score cannot compensate for a failed hard check;
- complete breakdown is retained in trajectory/audit output.

`cogno-core` remains the scalar/independent authority; optimized/back-end adapters must cross-validate against it.

#### P4.2 — RSI: COGNO-aware evaluator composition

Split evaluation into:

1. admissibility evidence collection;
2. COGNO decision;
3. ranking measurements only for admissible candidates.

Preserve existing Cargo build/test evaluator as one evidence source rather than the whole policy.

### P5 — cross-repository evaluation substrate

#### P5.1 — RSI: CrossRepoWorkspace

Support a task spanning exact revisions from multiple repositories.

Initial requirements:

- materialize a compatibility set into isolated roots;
- apply PatchSets only to repositories explicitly allowed by the task;
- create temporary dependency overrides without mutating tracked manifests when possible;
- record every effective repository revision and override in the result;
- clean up deterministically/best-effort after evaluation.

No arbitrary remote execution belongs in the core abstraction.

#### P5.2 — RSI: command/evidence pipeline

Introduce a declarative bounded evaluation plan, e.g.:

```text
EvaluationStep {
  repository_role,
  command_kind,
  arguments,
  timeout,
  output_limit,
  evidence_kind,
}
```

Allowed command kinds are host-defined; generated candidates do not get to inject arbitrary shell strings into evaluator policy.

### P6 — FLAT-ATTENTION as the first cross-repo proving ground

Starting baseline must be the current merged FLAT-ATTENTION M11 contract plus SciRust's current resident KV-cache path.

#### P6.1 — FLAT-ATTENTION: finish/normalize M11 branch state

Before new integration work:

- resolve any stale/diverged duplicate PRs against current `main`;
- retain only changes not already merged;
- require all FLAT device/CI qualification on the final head.

No speedup claim is required here; correctness qualification is the goal.

#### P6.2 — SciRust: pin reviewed rectangular FLAT revision

Tasks:

- update the pinned FLAT revision only after FLAT final-head CI is green;
- expose the rectangular decode path behind a controlled integration boundary;
- preserve existing equal-length path until parity and performance evidence justify replacement;
- add SciRust-side parity/integration tests.

#### P6.3 — SciRust + FLAT: decode benchmark contract

Create a benchmark with machine-readable output covering at least:

- `Q=1, KV=N` decode;
- representative GQA/MQA head ratios;
- warm-up count;
- repeated samples and declared statistic;
- resident-cache baseline versus rectangular FLAT candidate;
- adapter/device identity;
- no universal performance conclusion from one adapter.

The benchmark emits a ranking score only after parity passes.

#### P6.4 — RSI: `FlatAttentionEvaluator`

Compose the cross-repo workspace, COGNO hard gate and benchmark contract so RSI can evaluate a FLAT candidate against SciRust without changing either live repository.

Required evidence:

- FLAT tests;
- WGPU qualification when required by the task;
- SciRust GPU/SciAgent integration tests;
- numerical parity;
- benchmark record;
- exact revisions.

### P7 — flywheel v2 for engineering intelligence

#### P7.1 — RSI: versioned engineering trajectory schema

Extend trajectories to include:

- frozen task/spec ID;
- compatibility set;
- parent state ID;
- PatchSet;
- proposer metadata;
- compiler/test/device evidence;
- COGNO admissibility breakdown;
- benchmark samples + summary;
- accepted/rejected verdict and reason;
- later CI/review verdict when available.

Schema is versioned and deterministic.

#### P7.2 — SciAgent: dataset ingestion

Repository: `Memorithm/scirust`

Add a native ingestion path that converts engineering trajectories into training/evaluation examples while preserving provenance and split discipline.

Minimum requirements:

- deterministic train/eval split;
- deduplication by semantic/state identity where feasible;
- negative examples retained;
- no leakage of held-out evaluation targets into training prompts;
- manifest/checksum for the produced dataset.

#### P7.3 — SciAgent: measured specialization loop

Add a bounded specialization experiment whose success criterion is not training loss alone. It must improve held-out engineering metrics such as:

- valid PatchSet rate;
- compile-pass prediction/calibration;
- first-pass gate success;
- accepted-candidate yield per evaluation budget.

A model update is retained only if held-out engineering metrics satisfy the frozen criterion.

### P8 — AUTOPILOT outer engineering loop

Implement the existing `docs/AUTOPILOT.md` design on top of the now-correct inner loop.

#### P8.1 — intake/spec

- repository exploration before questions/spec generation;
- machine-readable frozen acceptance criteria;
- explicit scope and repository allowlist.

#### P8.2 — task DAG

Each task includes:

- repository subset;
- file/operation allowlist;
- hard-gate profile;
- benchmark profile if PERF;
- budget;
- dependencies;
- executable done criterion.

#### P8.3 — FEATURE regime

- tests/specification first;
- frozen test hashes after approval/acceptance;
- implementation task cannot edit frozen tests.

#### P8.4 — PERF regime

- benchmark definition frozen before candidate generation;
- COGNO hard gates precede ranking;
- anti-noise threshold and repeated measurement policy mandatory.

#### P8.5 — PR emitter and review flywheel

- branch/PR only, never direct main/master;
- PR body contains compatibility set and evidence;
- review/CI verdicts are appended to the trajectory;
- no automatic merge policy is embedded in model output or candidate code.

### P9 — synchronization and release discipline

#### P9.1 — cross-repo compatibility lock

Maintain a machine-readable compatibility document/artifact in RSI containing the last fully-qualified set of:

- RSI revision;
- canonical SciRust revision;
- COGNO contract version;
- FLAT-ATTENTION revision;
- required features/toolchain.

Update it only in a dedicated synchronization PR after all component PRs are merged and green.

#### P9.2 — end-to-end qualification

A release candidate of the engineering loop must demonstrate:

1. a multi-file cumulative RSI lineage;
2. COGNO rejection of an intentionally high-score but invalid candidate;
3. cross-repo FLAT + SciRust evaluation;
4. exact revision replay;
5. trajectory export;
6. SciAgent ingestion of that trajectory;
7. no live-tree mutation during dry-run;
8. successful final-head CI across the compatibility set.

## 5. CI matrix expectations

### RSI

At minimum for affected PRs:

- default clippy/tests;
- public feature clippy/tests already required by repository CI;
- focused `scirust` feature coverage once canonical dependency migration lands;
- COGNO oracle/backend cross-validation for COGNO changes;
- DGM chaos tests for any safety boundary change.

### SciRust

Use repository-required gates on the exact final head. For scoped changes, add focused tests rather than weakening workspace gates. GPU/Flat integration must preserve current CPU/non-WGPU builds.

### FLAT-ATTENTION

- fmt;
- clippy all targets/features;
- tests all features;
- required WGPU/lavapipe qualification;
- subgroup-specific gate only where the adapter contract requires it.

## 6. Merge and synchronization protocol

For every implementation PR:

1. branch from the current merged default branch;
2. make one coherent change set;
3. open PR with mechanical done criteria;
4. inspect all required workflow runs on the final head;
5. repair failures on the same branch;
6. re-check final-head status after every repair/rebase;
7. merge only when green and mergeable;
8. fetch the resulting default-branch merge commit;
9. update the program tracking issue;
10. start the next dependent PR from that new merged state.

Cross-repo dependencies additionally record the exact upstream merge SHA before the downstream branch is created.

## 7. Stop/rollback conditions

Execution stops advancing to dependent work when:

- required CI is red or unavailable;
- a PR is not mergeable because its base changed;
- a hard-gate invariant is weakened;
- a benchmark cannot distinguish measurement from noise under its frozen policy;
- a dependency revision cannot be reproduced;
- a proposed change would require silently modifying the plan's hard rules.

The repair remains within the current PR or is replaced by a smaller corrective PR; dependent work does not leapfrog the failure.

## 8. Current starting state (2026-08-10)

Verified before publication of this plan:

- RSI `main` includes COGNO-1 and the current DGM/flywheel implementation.
- SciRust `master` contains the richer canonical `scirust-rsi` implementation and the resident WGPU KV-cache work.
- FLAT-ATTENTION M11 asymmetric/rectangular work is active; merged and stale/diverged branches must be normalized before further integration.
- RSI's documented `SCIRUST_ACTIVATION.md` dependency description and its current local `scirust-rsi` path dependency are not aligned and are scheduled for P1.

## 9. First execution tranche

Immediately after this plan merges, execute in this exact order:

1. **P0.2** compatibility manifest in RSI.
2. **P1.1** canonical `scirust-rsi` contract tests/documentation in SciRust.
3. **P1.2** RSI migration to the exact merged SciRust revision.
4. **P2.1** PatchSet core in RSI.
5. **P2.2** PatchSet proposer/trajectory support.
6. **P3.1/P3.2** cumulative materialized lineages and promotion.
7. **P4.1/P4.2** COGNO engineering hard gate and evaluator composition.
8. **P5.1/P5.2** cross-repository workspace/evidence pipeline.
9. Normalize FLAT M11 and proceed through **P6**.
10. Build flywheel v2 and SciAgent ingestion (**P7**), then AUTOPILOT (**P8**).

The ordering is deliberate: cross-repo autonomous engineering is not enabled until the candidate representation, cumulative lineage and hard admissibility semantics are correct.
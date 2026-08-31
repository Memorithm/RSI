# RSI Agent Bootstrap Contract

Before autonomous coding, recursive-loop changes, safety/adoption changes, backend integration, self-improvement promotion, PR creation, or merge decisions, read both:

```bash
git fetch origin agent/ecosystem-roadmap && \
git show origin/agent/ecosystem-roadmap:.agent/RSI_ECOSYSTEM_ROADMAP.yaml && \
git show origin/main:docs/ROADMAP.md
```

`docs/ROADMAP.md` is the published engineering plan. The off-main roadmap is authoritative for persistent ecosystem boundaries, backend lineage and promotion state.

If either roadmap cannot be read, fail closed for major self-improvement-promotion, safety, backend-lineage, cross-repository, or merge decisions. Read-only diagnosis is allowed.

## Core rule

LLM proposals are candidates. The deterministic RSI engine validates safety, evaluates candidates, and decides adoption. Dry-run, candidate generation, or a higher predicted score is not promotion.

Preserve loop stop conditions, checkpoint/replay, observer/veto semantics and explicit safety overrides. Human veto and high-impact approval gates must not be bypassed by autonomous modes.

Embedded Forge, CCOS, OctaSoma or SciRust-related backends must not silently diverge from their owning repositories. Audit lineage before modifying their semantics.

Required CI must be green on the exact PR head before merge.

Reread both roadmaps at every session start, before major loop/safety/backend work, after promotion/rejection state changes, and before PR/merge decisions.

Do not merge the off-main roadmap itself into `main` unless the user explicitly requests it.

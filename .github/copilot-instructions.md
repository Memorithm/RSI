# RSI repository agent instructions

Before repository changes, read both the persistent off-main ecosystem roadmap and the published engineering roadmap:

```bash
git fetch origin agent/ecosystem-roadmap && \
git show origin/agent/ecosystem-roadmap:.agent/RSI_ECOSYSTEM_ROADMAP.yaml && \
git show origin/main:docs/ROADMAP.md
```

Treat root `AGENTS.md` as mandatory bootstrap policy. If either roadmap is unavailable, fail closed for major self-improvement-promotion, safety, backend-lineage, cross-repository, or merge decisions.

Preserve deterministic validation/adoption authority over LLM proposals, loop safety/veto/replay guarantees, and explicit lineage for embedded ecosystem backends.

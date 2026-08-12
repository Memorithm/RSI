# PAPERS scientific contract in RSI

RSI consumes PAPERS through the versioned `memorithm.science/bundle-v1` JSON contract. It does **not** link `papers_core` into the RSI core.

## Trust boundary

A PAPERS claim is an input to candidate generation, not proof. `ScientificBundle::directive_goals` preserves the claim id and explicitly requires the existing empirical DGM gate (`build + tests + benchmark`) before promotion.

Claims marked `contradicted` or `not_applicable` are not converted into directive goals.

## Runtime bridge

`rsi::paper_science::ScientificPapersRunner` drives two external binaries:

1. `papers analyze` produces `analysis.json`;
2. `papers-contract` converts that report to `scientific_bundle.json`.

Binary discovery:

- `RSI_PAPERS_BIN` — defaults to `papers` on `PATH`;
- `RSI_PAPERS_CONTRACT_BIN` — defaults to `papers-contract` beside an explicit PAPERS binary, otherwise `papers-contract` on `PATH`.

Both subprocesses are bounded by timeout and output caps and are invoked without a shell.

## Model provenance

For model-backed analysis, use `PaperAnalysisMode::Model { provider, model }`.

The runner sets `PAPERS_LLM__PROVIDER=<provider>` **on the PAPERS analysis subprocess itself**. PAPERS loads `PAPERS_` environment overrides after its defaults/config file, so the provider later written into the scientific bundle is the provider actually selected for analysis. The model is forced by PAPERS' `--model` CLI argument.

Endpoint/API-key configuration remains PAPERS configuration, for example through its `PAPERS_LLM__...` variables. RSI does not copy secrets into the bundle.

## Example

```rust
use std::path::Path;
use rsi::paper_science::{PaperAnalysisMode, ScientificPapersRunner};

let runner = ScientificPapersRunner::from_environment();
let bundle = runner.analyze_bundle(
    "2401.01234",
    Path::new(".rsi_science/paper"),
    &PaperAnalysisMode::Model {
        provider: "ollama".into(),
        model: "local-model".into(),
    },
)?;

let goals = bundle.directive_goals("src/kernel.rs", 3);
```

The returned goals are still hypotheses to test. They are not automatically promoted.

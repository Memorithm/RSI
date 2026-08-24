//! Démonstration **bout-en-bout de la pile autopilot P5→P8.5** — le
//! consommateur exécutable qui manquait à cette couche produit.
//!
//! Le pipeline construit un plan de pull-request complet et auditable à partir
//! d'un objectif gelé, sans aucun accès réseau ni écriture disque :
//!
//! 1. **P5 intake** : exploration dépôt → objectif → questionnaire résolu →
//!    `FrozenAutopilotSpec` (hash-pinné) ;
//! 2. **DAG** : une tâche `perf` (régime gelé, budget, gates stricts) ;
//! 3. **trajectoire engineering** : patch candidat avec tous les gates au vert ;
//! 4. **P8.4 perf** : profil de benchmark gelé + mesures baseline/candidat →
//!    `VerifiedPerfPromotion` (preuve de promotion non falsifiable) ;
//! 5. **P8.5 PR plan** : liaison tâche↔trajectoire↔promotion → plan
//!    `create_branch` puis `open_pull_request` — **jamais de merge**.
//!
//! ```bash
//! cargo run --release --example autopilot_pipeline
//! ```

use rsi::{
    AcceptanceCheck, AcceptanceCriterion, AntiNoisePolicy, AutopilotPullRequestPlan,
    AutopilotSpecDraft, AutopilotTask, AutopilotTaskDag, AutopilotTaskDraft, BenchmarkCase,
    BenchmarkCaseSpec, BenchmarkClass, BenchmarkEnvironment, CompatibilitySet,
    EngineeringTrajectory, EngineeringVerdict, ExplorationObservation, ExplorationSource,
    ExploredObjective, FileOperation, FrozenAutopilotSpec, FrozenBenchmarkArtifact,
    FrozenPerfBenchmark, GateStatus, HardGateProfile, MetricDirection, PatchSet,
    PerfBenchmarkApproval, PerfBenchmarkDraft, PerfMeasurementBatch, ProposerMetadata,
    PullRequestPlanDraft, RepositoryExploration, RepositoryRevision, RepositoryScope,
    SpecBudget, TaskBudget, TaskDagPolicy, TaskEditAllowance,
    TaskOperation, TaskRegime,
};
// types non ré-exportés à la racine :
use rsi::autopilot_pr::TaskBoundEngineeringTrajectory;
use rsi::autopilot_pr::VerifiedPerfPromotion;
use rsi::engineering_trajectory::AdmissibilityBreakdown as Gates;

fn hex64(c: char) -> String {
    c.to_string().repeat(64)
}

fn rev40(c: char) -> String {
    c.to_string().repeat(40)
}

/// P5 — spécification d'intake gelée.
fn spec() -> FrozenAutopilotSpec {
    let exploration = RepositoryExploration::new(
        "Memorithm/RSI",
        rev40('a'),
        vec![ExplorationObservation::new(
            "code",
            ExplorationSource::repository_file("src/lib.rs", hex64('b')).unwrap(),
            "inspected the implementation and CI boundary",
        )
        .unwrap()],
    )
    .unwrap();
    ExploredObjective::new(
        "emit an auditable engineering pull request",
        "candidate already passed the frozen inner-loop contract",
        vec![exploration],
    )
    .unwrap()
    .questionnaire(Vec::new())
    .unwrap()
    .resolve(Vec::new())
    .unwrap()
    .freeze(AutopilotSpecDraft::new(
        vec![AcceptanceCriterion::new(
            "tests-pass",
            "all required tests pass",
            AcceptanceCheck::command("rsi", "cargo_test", Vec::new()).unwrap(),
        )
        .unwrap()],
        vec!["automatic merge".to_string()],
        SpecBudget::new(20, 20_000, 20_000).unwrap(),
        vec![RepositoryScope::new(
            "rsi",
            "Memorithm/RSI",
            rev40('a'),
            vec!["benches".to_string(), "src".to_string()],
        )
        .unwrap()],
    ))
    .unwrap()
}

/// Tâche du DAG : régime perf gelé sur la même spéc.
fn task(spec: &FrozenAutopilotSpec) -> AutopilotTaskDag {
    let draft = AutopilotTaskDraft {
        id: "perf-task".to_string(),
        description: "accelerate the hot loop without regressing tests".to_string(),
        regime: TaskRegime::perf("decode-v1").unwrap(),
        repository_roles: vec!["rsi".to_string()],
        edit_allowances: vec![TaskEditAllowance::new(
            "rsi",
            vec!["src/kernels.rs".to_string()],
            vec![TaskOperation::ModifyExact],
        )
        .unwrap()],
        hard_gate_profile: HardGateProfile::engineering_strict(),
        budget: TaskBudget::new(4, 8, 8_000, 8_000).unwrap(),
        dependencies: Vec::new(),
        done_criterion_id: "tests-pass".to_string(),
    };
    let t = AutopilotTask::new(draft).unwrap();
    AutopilotTaskDag::new(spec, vec![t], TaskDagPolicy::new(8, 8, 16).unwrap()).unwrap()
}

/// Trajectoire engineering candidate (gates tout-au-vert).
fn trajectory(spec: &FrozenAutopilotSpec) -> EngineeringTrajectory {
    EngineeringTrajectory {
        task_spec_id: spec.spec_sha256().to_string(),
        compatibility: CompatibilitySet::new(
            vec![RepositoryRevision::new("Memorithm/RSI", rev40('a'), "rsi").unwrap()],
            "stable",
            vec!["default".to_string()],
        )
        .unwrap(),
        parent_state_id: hex64('c'),
        patch_set: PatchSet::new(vec![FileOperation::modify_exact(
            "src/kernels.rs",
            "old implementation",
            "faster implementation",
        )])
        .unwrap(),
        proposer: ProposerMetadata::new("sciagent", "engineering", "heldout-v1").unwrap(),
        compiler_test_device_evidence: vec!["cargo test: pass".to_string()],
        admissibility: Gates {
            build: GateStatus::Pass,
            required_tests: GateStatus::Pass,
            numerical_parity: GateStatus::Pass,
            provenance: GateStatus::Pass,
            deterministic_contract: GateStatus::Pass,
            resource_budget: GateStatus::Pass,
            policy_checks: GateStatus::Pass,
        },
        benchmarks: Vec::new(),
        verdict: EngineeringVerdict::Accepted,
        verdict_reason: "frozen acceptance criteria satisfied".to_string(),
        later_verdicts: Vec::new(),
    }
}

/// P8.4 — profil de benchmark gelé + mesures → preuve de promotion vérifiée.
fn promotion(
    spec: &FrozenAutopilotSpec,
    dag: &AutopilotTaskDag,
    traj: &EngineeringTrajectory,
) -> VerifiedPerfPromotion {
    let profile = FrozenPerfBenchmark::freeze(
        spec,
        dag,
        "perf-task",
        PerfBenchmarkDraft {
            approval: PerfBenchmarkApproval::new("human-review", hex64('d')).unwrap(),
            environment: BenchmarkEnvironment::new(hex64('e'), hex64('f')).unwrap(),
            policy: AntiNoisePolicy::new(5, 3, 20_000, 20_000, 3).unwrap(),
            cases: vec![BenchmarkCase::new(BenchmarkCaseSpec {
                id: "e2e-latency".to_string(),
                repository_role: "rsi".to_string(),
                command_kind: "bench_e2e".to_string(),
                arguments: Vec::new(),
                metric: "latency".to_string(),
                unit: "ns".to_string(),
                direction: MetricDirection::Minimize,
                class: BenchmarkClass::EndToEnd,
                promotion_gate: true,
            })
            .unwrap()],
            artifacts: vec![
                FrozenBenchmarkArtifact::new("rsi", "benches/perf.rs", hex64('1')).unwrap()
            ],
        },
    )
    .unwrap();

    let mut baseline = Vec::new();
    let mut candidate = Vec::new();
    for run in ["a", "b", "c"] {
        baseline.push(
            PerfMeasurementBatch::new(
                "e2e-latency",
                run,
                profile.environment_fingerprint(),
                vec![100.0, 100.5, 99.5, 100.2, 99.8],
            )
            .unwrap(),
        );
        candidate.push(
            PerfMeasurementBatch::new(
                "e2e-latency",
                run,
                profile.environment_fingerprint(),
                vec![90.0, 90.4, 89.6, 90.2, 89.8],
            )
            .unwrap(),
        );
    }
    // ~10 % plus rapide ET moins bruyant que le seuil anti-noise → promotable
    VerifiedPerfPromotion::evaluate(&profile, traj, &baseline, &candidate).unwrap()
}

fn main() {
    println!("{}", "═".repeat(78));
    println!("  AUTOPILOT PIPELINE — intake → DAG → trajectoire → perf → PR plan");
    println!("{}", "═".repeat(78));

    // 1-2. spec gelée + DAG de tâches
    let spec = spec();
    let dag = task(&spec);
    println!("\n[1] FrozenAutopilotSpec  sha256={}…", &spec.spec_sha256()[..16]);
    println!("    objectif : {}", spec.objective());
    println!(
        "[2] DAG                  {} tâche(s), politique gelée",
        dag.tasks().len()
    );

    // 3. trajectoire candidate acceptée par l'inner loop
    let traj = trajectory(&spec);
    println!(
        "[3] Trajectoire          verdict={:?}, {} opération(s) de patch",
        traj.verdict,
        traj.patch_set.operations().len(),
    );

    // 4. promotion perf VÉRIFIÉE (le champ est privé : impossible à fabriquer)
    let promo = promotion(&spec, &dag, &traj);
    let report = promo.report();
    println!(
        "[4] Promotion perf       promotable={} (profil {}…)",
        report.promotable,
        &promo.profile_sha256()[..16]
    );

    // 5. liaison + plan de PR (create_branch + open_pull_request, jamais merge)
    let bound = TaskBoundEngineeringTrajectory::new(
        &spec,
        &dag,
        "perf-task",
        "rsi",
        traj,
        Some(promo),
    )
    .unwrap();
    let plan = AutopilotPullRequestPlan::new(
        &spec,
        &dag,
        &bound,
        PullRequestPlanDraft::new("main", "feat(autopilot): apply verified perf candidate"),
    )
    .unwrap();

    println!("[5] Plan de PR           branche « {} »", plan.branch_name());
    for a in plan.hosting_actions() {
        println!("                         action : {:?}", a);
    }
    assert_ne!(plan.branch_name(), plan.default_branch());
    assert!(plan.body().contains("does not request or encode automatic merge"));

    println!("\nPlan JSON ({} octets) — émission = revue humaine puis CI :", plan.to_json_string().len());
    println!("{}", "─".repeat(78));
    println!("base: {} ← head: {}", plan.default_branch(), plan.branch_name());

    println!("\n✓ pipeline complet exécuté hors-ligne : aucune écriture, aucun merge.");
}

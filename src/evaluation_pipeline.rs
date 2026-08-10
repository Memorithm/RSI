//! Bounded declarative command/evidence pipeline for P5.2.
//!
//! Candidate plans name host-defined command kinds. They never provide an
//! executable or shell command. A trusted [`EvaluationCommandHost`] resolves
//! each kind and validates its arguments before a process can be spawned.

use crate::cross_repo_workspace::CrossRepoWorkspace;
use std::collections::BTreeMap;
use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum EvidenceKind {
    Build,
    RequiredTests,
    NumericalParity,
    Provenance,
    Determinism,
    ResourceBudget,
    Policy,
    Benchmark,
    Named(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluationStep {
    pub repository_role: String,
    pub command_kind: String,
    pub arguments: Vec<String>,
    pub timeout_ms: u64,
    pub output_limit: usize,
    pub evidence_kind: EvidenceKind,
}

impl EvaluationStep {
    pub fn new(
        repository_role: impl Into<String>,
        command_kind: impl Into<String>,
        arguments: Vec<String>,
        timeout_ms: u64,
        output_limit: usize,
        evidence_kind: EvidenceKind,
    ) -> Result<Self, EvaluationPipelineError> {
        let repository_role = checked_text("repository_role", repository_role.into())?;
        let command_kind = checked_text("command_kind", command_kind.into())?;
        Ok(Self {
            repository_role,
            command_kind,
            arguments,
            timeout_ms,
            output_limit,
            evidence_kind,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvaluationPlanPolicy {
    pub max_steps: usize,
    pub max_timeout_ms: u64,
    pub max_output_limit: usize,
    pub max_arguments: usize,
    pub max_argument_bytes: usize,
}

impl EvaluationPlanPolicy {
    pub fn new(
        max_steps: usize,
        max_timeout_ms: u64,
        max_output_limit: usize,
        max_arguments: usize,
        max_argument_bytes: usize,
    ) -> Result<Self, EvaluationPipelineError> {
        if max_steps == 0
            || max_timeout_ms == 0
            || max_output_limit == 0
            || max_argument_bytes == 0
        {
            return Err(EvaluationPipelineError::InvalidPlan(
                "step, timeout, output and argument-byte limits must be positive".into(),
            ));
        }
        Ok(Self {
            max_steps,
            max_timeout_ms,
            max_output_limit,
            max_arguments,
            max_argument_bytes,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluationPlan {
    steps: Vec<EvaluationStep>,
    policy: EvaluationPlanPolicy,
}

impl EvaluationPlan {
    pub fn new(
        steps: Vec<EvaluationStep>,
        policy: EvaluationPlanPolicy,
    ) -> Result<Self, EvaluationPipelineError> {
        if steps.is_empty() {
            return Err(EvaluationPipelineError::InvalidPlan(
                "evaluation plan must contain at least one step".into(),
            ));
        }
        if steps.len() > policy.max_steps {
            return Err(EvaluationPipelineError::InvalidPlan(format!(
                "step limit exceeded: {} > {}",
                steps.len(), policy.max_steps
            )));
        }
        for (index, step) in steps.iter().enumerate() {
            if step.timeout_ms == 0 || step.timeout_ms > policy.max_timeout_ms {
                return Err(EvaluationPipelineError::InvalidPlan(format!(
                    "step {index} timeout {} outside 1..={}",
                    step.timeout_ms, policy.max_timeout_ms
                )));
            }
            if step.output_limit == 0 || step.output_limit > policy.max_output_limit {
                return Err(EvaluationPipelineError::InvalidPlan(format!(
                    "step {index} output limit {} outside 1..={}",
                    step.output_limit, policy.max_output_limit
                )));
            }
            if step.arguments.len() > policy.max_arguments {
                return Err(EvaluationPipelineError::InvalidPlan(format!(
                    "step {index} argument-count limit exceeded"
                )));
            }
            let argument_bytes = step.arguments.iter().try_fold(0usize, |total, argument| {
                if argument.contains('\0') {
                    return Err(EvaluationPipelineError::InvalidPlan(format!(
                        "step {index} contains a NUL argument"
                    )));
                }
                total.checked_add(argument.len()).ok_or_else(|| {
                    EvaluationPipelineError::InvalidPlan(format!(
                        "step {index} argument-byte count overflow"
                    ))
                })
            })?;
            if argument_bytes > policy.max_argument_bytes {
                return Err(EvaluationPipelineError::InvalidPlan(format!(
                    "step {index} argument-byte limit exceeded: {argument_bytes} > {}",
                    policy.max_argument_bytes
                )));
            }
        }
        Ok(Self { steps, policy })
    }

    pub fn steps(&self) -> &[EvaluationStep] {
        &self.steps
    }

    pub fn policy(&self) -> EvaluationPlanPolicy {
        self.policy
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedCommand {
    pub program: PathBuf,
    pub arguments: Vec<String>,
    pub environment: BTreeMap<String, String>,
}

impl ResolvedCommand {
    pub fn new(program: impl Into<PathBuf>, arguments: Vec<String>) -> Self {
        Self {
            program: program.into(),
            arguments,
            environment: BTreeMap::new(),
        }
    }

    pub fn with_environment(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.environment.insert(key.into(), value.into());
        self
    }
}

/// Trusted host policy for resolving declarative command kinds.
///
/// The candidate controls only `command_kind` and bounded `arguments`. The host
/// must reject kinds/arguments outside its policy and returns the actual process
/// invocation. The pipeline never invokes a shell implicitly.
pub trait EvaluationCommandHost {
    fn resolve(
        &self,
        command_kind: &str,
        candidate_arguments: &[String],
        repository_root: &Path,
        cargo_override_config: Option<&Path>,
    ) -> Result<ResolvedCommand, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepEvidence {
    pub repository_role: String,
    pub command_kind: String,
    pub evidence_kind: EvidenceKind,
    pub exit_code: Option<i32>,
    pub success: bool,
    pub timed_out: bool,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub output_truncated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluationEvidence {
    pub steps: Vec<StepEvidence>,
}

pub struct BoundedEvidencePipeline;

impl BoundedEvidencePipeline {
    pub fn run<H: EvaluationCommandHost>(
        workspace: &CrossRepoWorkspace,
        plan: &EvaluationPlan,
        host: &H,
    ) -> Result<EvaluationEvidence, EvaluationPipelineError> {
        let mut evidence = Vec::with_capacity(plan.steps.len());
        for step in &plan.steps {
            let repository_root = workspace
                .root_for_role(&step.repository_role)
                .ok_or_else(|| EvaluationPipelineError::UnknownRole(step.repository_role.clone()))?;
            let resolved = host
                .resolve(
                    &step.command_kind,
                    &step.arguments,
                    repository_root,
                    workspace.cargo_override_config(&step.repository_role),
                )
                .map_err(|message| EvaluationPipelineError::HostPolicy {
                    command_kind: step.command_kind.clone(),
                    message,
                })?;
            validate_resolved(&resolved)?;
            evidence.push(execute_step(step, repository_root, resolved)?);
        }
        Ok(EvaluationEvidence { steps: evidence })
    }
}

#[derive(Debug)]
pub enum EvaluationPipelineError {
    InvalidPlan(String),
    UnknownRole(String),
    HostPolicy { command_kind: String, message: String },
    InvalidResolvedCommand(String),
    Spawn(String),
    Wait(String),
    CaptureThreadPanic,
}

impl fmt::Display for EvaluationPipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPlan(message) => write!(f, "invalid evaluation plan: {message}"),
            Self::UnknownRole(role) => write!(f, "unknown evaluation repository role: {role}"),
            Self::HostPolicy {
                command_kind,
                message,
            } => write!(f, "host rejected command kind {command_kind}: {message}"),
            Self::InvalidResolvedCommand(message) => {
                write!(f, "invalid host-resolved command: {message}")
            }
            Self::Spawn(message) => write!(f, "evaluation process spawn failed: {message}"),
            Self::Wait(message) => write!(f, "evaluation process wait failed: {message}"),
            Self::CaptureThreadPanic => write!(f, "evaluation output capture thread panicked"),
        }
    }
}

impl std::error::Error for EvaluationPipelineError {}

fn execute_step(
    step: &EvaluationStep,
    repository_root: &Path,
    resolved: ResolvedCommand,
) -> Result<StepEvidence, EvaluationPipelineError> {
    let mut command = Command::new(&resolved.program);
    command
        .args(&resolved.arguments)
        .current_dir(repository_root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (key, value) in &resolved.environment {
        command.env(key, value);
    }
    let mut child = command
        .spawn()
        .map_err(|error| EvaluationPipelineError::Spawn(error.to_string()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| EvaluationPipelineError::Spawn("stdout pipe unavailable".into()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| EvaluationPipelineError::Spawn("stderr pipe unavailable".into()))?;
    let per_stream_cap = step.output_limit;
    let stdout_thread = thread::spawn(move || capture_bounded(stdout, per_stream_cap));
    let stderr_thread = thread::spawn(move || capture_bounded(stderr, per_stream_cap));

    let deadline = Duration::from_millis(step.timeout_ms);
    let start = Instant::now();
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if start.elapsed() >= deadline => {
                timed_out = true;
                let _ = child.kill();
                break child
                    .wait()
                    .map_err(|error| EvaluationPipelineError::Wait(error.to_string()))?;
            }
            Ok(None) => thread::sleep(Duration::from_millis(5)),
            Err(error) => return Err(EvaluationPipelineError::Wait(error.to_string())),
        }
    };

    let stdout = stdout_thread
        .join()
        .map_err(|_| EvaluationPipelineError::CaptureThreadPanic)?;
    let stderr = stderr_thread
        .join()
        .map_err(|_| EvaluationPipelineError::CaptureThreadPanic)?;
    let (stdout_bytes, stderr_bytes, output_truncated) =
        enforce_combined_limit(stdout, stderr, step.output_limit);

    Ok(StepEvidence {
        repository_role: step.repository_role.clone(),
        command_kind: step.command_kind.clone(),
        evidence_kind: step.evidence_kind.clone(),
        exit_code: exit_code(status),
        success: status.success() && !timed_out,
        timed_out,
        stdout: stdout_bytes,
        stderr: stderr_bytes,
        output_truncated,
    })
}

struct BoundedCapture {
    bytes: Vec<u8>,
    discarded: bool,
}

fn capture_bounded<R: Read>(mut reader: R, limit: usize) -> BoundedCapture {
    let mut captured = Vec::with_capacity(limit.min(16 * 1024));
    let mut discarded = false;
    let mut buffer = [0u8; 8192];
    loop {
        let count = match reader.read(&mut buffer) {
            Ok(0) | Err(_) => break,
            Ok(count) => count,
        };
        let remaining = limit.saturating_sub(captured.len());
        let keep = remaining.min(count);
        captured.extend_from_slice(&buffer[..keep]);
        if keep < count {
            discarded = true;
        }
    }
    BoundedCapture {
        bytes: captured,
        discarded,
    }
}

fn enforce_combined_limit(
    mut stdout: BoundedCapture,
    mut stderr: BoundedCapture,
    limit: usize,
) -> (Vec<u8>, Vec<u8>, bool) {
    let mut truncated = stdout.discarded || stderr.discarded;
    if stdout.bytes.len() > limit {
        stdout.bytes.truncate(limit);
        truncated = true;
    }
    let remaining = limit.saturating_sub(stdout.bytes.len());
    if stderr.bytes.len() > remaining {
        stderr.bytes.truncate(remaining);
        truncated = true;
    }
    (stdout.bytes, stderr.bytes, truncated)
}

fn validate_resolved(resolved: &ResolvedCommand) -> Result<(), EvaluationPipelineError> {
    if resolved.program.as_os_str().is_empty() {
        return Err(EvaluationPipelineError::InvalidResolvedCommand(
            "program must not be empty".into(),
        ));
    }
    for argument in &resolved.arguments {
        if argument.contains('\0') {
            return Err(EvaluationPipelineError::InvalidResolvedCommand(
                "host argument contains NUL".into(),
            ));
        }
    }
    for (key, value) in &resolved.environment {
        if key.is_empty() || key.contains('=') || key.contains('\0') || value.contains('\0') {
            return Err(EvaluationPipelineError::InvalidResolvedCommand(
                "invalid environment entry".into(),
            ));
        }
    }
    Ok(())
}

fn exit_code(status: ExitStatus) -> Option<i32> {
    status.code()
}

fn checked_text(
    field: &'static str,
    value: String,
) -> Result<String, EvaluationPipelineError> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(EvaluationPipelineError::InvalidPlan(format!(
            "invalid {field}"
        )));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> EvaluationPlanPolicy {
        EvaluationPlanPolicy::new(4, 2_000, 128, 4, 256).unwrap()
    }

    #[test]
    fn plan_rejects_unbounded_timeout_output_and_arguments() {
        let too_slow = EvaluationStep::new(
            "rsi",
            "test",
            vec![],
            2_001,
            16,
            EvidenceKind::RequiredTests,
        )
        .unwrap();
        assert!(EvaluationPlan::new(vec![too_slow], policy()).is_err());

        let too_large = EvaluationStep::new(
            "rsi",
            "test",
            vec![],
            100,
            129,
            EvidenceKind::RequiredTests,
        )
        .unwrap();
        assert!(EvaluationPlan::new(vec![too_large], policy()).is_err());

        let too_many_args = EvaluationStep::new(
            "rsi",
            "test",
            vec!["a".into(), "b".into(), "c".into(), "d".into(), "e".into()],
            100,
            16,
            EvidenceKind::RequiredTests,
        )
        .unwrap();
        assert!(EvaluationPlan::new(vec![too_many_args], policy()).is_err());
    }

    #[test]
    fn resolved_command_rejects_nul_without_spawning() {
        let resolved = ResolvedCommand::new("cargo", vec!["bad\0arg".into()]);
        assert!(matches!(
            validate_resolved(&resolved),
            Err(EvaluationPipelineError::InvalidResolvedCommand(_))
        ));
    }

    #[cfg(unix)]
    mod unix {
        use super::*;
        use crate::candidate_state::CandidateStoragePolicy;
        use crate::compatibility::{CompatibilitySet, RepositoryRevision};
        use crate::cross_repo_workspace::{
            CrossRepoWorkspacePolicy, LocalRepositorySource,
        };
        use std::sync::atomic::{AtomicU64, Ordering};

        static TEST_SEQ: AtomicU64 = AtomicU64::new(0);

        struct Host;

        impl EvaluationCommandHost for Host {
            fn resolve(
                &self,
                command_kind: &str,
                candidate_arguments: &[String],
                _repository_root: &Path,
                _cargo_override_config: Option<&Path>,
            ) -> Result<ResolvedCommand, String> {
                if !candidate_arguments.is_empty() {
                    return Err("dynamic arguments are not allowed in this test host".into());
                }
                match command_kind {
                    "emit" => Ok(ResolvedCommand::new(
                        "sh",
                        vec!["-c".into(), "printf 1234567890; printf abcdefghij >&2".into()],
                    )),
                    "sleep" => Ok(ResolvedCommand::new(
                        "sh",
                        vec!["-c".into(), "sleep 1".into()],
                    )),
                    "pass" => Ok(ResolvedCommand::new("sh", vec!["-c".into(), "exit 0".into()])),
                    _ => Err("unknown host command kind".into()),
                }
            }
        }

        fn git(root: &Path, args: &[&str]) -> String {
            let output = Command::new("git")
                .arg("-C")
                .arg(root)
                .args(args)
                .output()
                .unwrap();
            assert!(output.status.success());
            String::from_utf8(output.stdout).unwrap().trim().to_string()
        }

        fn workspace() -> (CrossRepoWorkspace, PathBuf) {
            let root = std::env::temp_dir().join(format!(
                "rsi-evidence-test-{}-{}",
                std::process::id(),
                TEST_SEQ.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(root.join("src")).unwrap();
            git(&root, &["init", "-q"]);
            git(&root, &["config", "user.email", "rsi-test@example.invalid"]);
            git(&root, &["config", "user.name", "RSI Test"]);
            std::fs::write(
                root.join("Cargo.toml"),
                "[package]\nname = \"fixture\"\nversion = \"0.1.0\"\n",
            )
            .unwrap();
            std::fs::write(root.join("src/lib.rs"), "pub fn fixture() {}\n").unwrap();
            git(&root, &["add", "."]);
            git(&root, &["commit", "-qm", "fixture"]);
            let revision = git(&root, &["rev-parse", "HEAD"]);
            let set = CompatibilitySet::new(
                vec![RepositoryRevision::new("Memorithm/RSI", &revision, "rsi").unwrap()],
                "rustc stable",
                vec![],
            )
            .unwrap();
            let workspace = CrossRepoWorkspace::materialize(
                set,
                vec![LocalRepositorySource::new("Memorithm/RSI", "rsi", &revision, &root).unwrap()],
                vec![],
                vec![],
                CrossRepoWorkspacePolicy::new(
                    1,
                    100_000,
                    CandidateStoragePolicy::new(20, 100_000).unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
            (workspace, root)
        }

        #[test]
        fn host_defined_command_runs_without_candidate_shell_control() {
            let (workspace, root) = workspace();
            let plan = EvaluationPlan::new(
                vec![EvaluationStep::new(
                    "rsi",
                    "pass",
                    vec![],
                    500,
                    64,
                    EvidenceKind::Build,
                )
                .unwrap()],
                policy(),
            )
            .unwrap();
            let evidence = BoundedEvidencePipeline::run(&workspace, &plan, &Host).unwrap();
            assert!(evidence.steps[0].success);

            let rejected = EvaluationPlan::new(
                vec![EvaluationStep::new(
                    "rsi",
                    "sh",
                    vec!["-c".into(), "rm -rf /".into()],
                    500,
                    64,
                    EvidenceKind::Policy,
                )
                .unwrap()],
                policy(),
            )
            .unwrap();
            assert!(matches!(
                BoundedEvidencePipeline::run(&workspace, &rejected, &Host),
                Err(EvaluationPipelineError::HostPolicy { .. })
            ));
            let _ = std::fs::remove_dir_all(root);
        }

        #[test]
        fn output_is_bounded_and_timeout_is_enforced() {
            let (workspace, root) = workspace();
            let plan = EvaluationPlan::new(
                vec![
                    EvaluationStep::new(
                        "rsi",
                        "emit",
                        vec![],
                        500,
                        12,
                        EvidenceKind::RequiredTests,
                    )
                    .unwrap(),
                    EvaluationStep::new(
                        "rsi",
                        "sleep",
                        vec![],
                        20,
                        32,
                        EvidenceKind::ResourceBudget,
                    )
                    .unwrap(),
                ],
                policy(),
            )
            .unwrap();
            let evidence = BoundedEvidencePipeline::run(&workspace, &plan, &Host).unwrap();
            assert_eq!(evidence.steps[0].stdout.len() + evidence.steps[0].stderr.len(), 12);
            assert!(evidence.steps[0].output_truncated);
            assert!(evidence.steps[1].timed_out);
            assert!(!evidence.steps[1].success);
            let _ = std::fs::remove_dir_all(root);
        }
    }
}
use std::process::Command;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/scientific_bundle_v1.json"
);

#[test]
fn rsi_science_emits_traceable_goals_without_promotion() {
    let output = Command::new(env!("CARGO_BIN_EXE_rsi-science"))
        .args([
            "--bundle",
            FIXTURE,
            "--target",
            "src/kernel.rs",
            "--max-goals",
            "3",
        ])
        .output()
        .expect("run rsi-science");

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("paper_id=fixture-paper-1"));
    assert!(stdout.contains("actionable_goals=1"));
    assert!(stdout.contains("method-fixture-1"));
    assert!(stdout.contains("build+tests+benchmark"));
    assert!(stdout.contains("ne promeus que si"));
}

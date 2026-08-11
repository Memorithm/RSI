use rsi::run_local_release_qualification;
use std::path::PathBuf;

fn main() {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/p9-release-qualification"));
    let artifacts = run_local_release_qualification().unwrap_or_else(|error| {
        eprintln!("P9 release qualification failed: {error}");
        std::process::exit(1);
    });
    artifacts.write_to(&output).unwrap_or_else(|error| {
        eprintln!("P9 artifact write failed: {error}");
        std::process::exit(1);
    });
    println!("p9_local_contract=pass");
    println!(
        "compatibility_fingerprint={}",
        artifacts.report.compatibility_fingerprint
    );
    println!("trajectory={}", output.join("engineering-trajectory.json").display());
    println!(
        "report={}",
        output.join("qualification-report.json").display()
    );
}

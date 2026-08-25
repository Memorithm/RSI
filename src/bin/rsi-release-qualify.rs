use rsi::run_local_release_qualification;
use std::path::PathBuf;

fn main() {
    // audit a22 : `-h` était traité comme nom de dossier de sortie
    let argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.iter().any(|a| a == "-h" || a == "--help") {
        eprintln!(
            "rsi-release-qualify — qualification locale hard-gate\n\n\
             USAGE:\n  rsi-release-qualify [OUT_DIR]   (défaut: ./qualification)"
        );
        std::process::exit(0);
    }
    let output = argv
        .first()
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

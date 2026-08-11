use rsi::release_compatibility::ReleaseCompatibilityLock;

const P10_LOCK_JSON: &str =
    include_str!("../compatibility/SCIRUST_RSI_P10_COMPATIBILITY_LOCK.json");

const RSI_P9_2: &str = "530b5e3b7c86e01c51b05927f6fad44d3c455cdc";
const SCIRUST_P10: &str = "4c4e71ffc73c00138a34c648b6ad954b24316184";
const SCIRUST_RSI_CANONICAL: &str = "8af0801b8bc0c69630797db82bb2dd3416cc8f0a";
const FLAT_M24: &str = "311f6b89e001d69f53cddcd2f9ba396a6f80c746";

fn revision<'a>(lock: &'a ReleaseCompatibilityLock, role: &str) -> &'a str {
    &lock
        .locked_revision(role)
        .unwrap_or_else(|| panic!("missing P10 repository role {role}"))
        .revision
}

#[test]
fn p10_lock_is_canonical_replayable_and_exact() {
    let lock = ReleaseCompatibilityLock::from_json_str(P10_LOCK_JSON.trim()).unwrap();

    assert_eq!(revision(&lock, "rsi"), RSI_P9_2);
    assert_eq!(revision(&lock, "scirust"), SCIRUST_P10);
    assert_eq!(revision(&lock, "scirust-rsi"), SCIRUST_RSI_CANONICAL);
    assert_eq!(revision(&lock, "flat"), FLAT_M24);
    assert_eq!(lock.cogno_contract_version(), "cogno-core@0.1.0");
    assert_eq!(
        lock.compatibility().toolchain(),
        "rustc 1.97.1 (qualification); MSRV 1.89"
    );
    assert_eq!(lock.to_json_string(), P10_LOCK_JSON.trim());
    assert_eq!(lock.fingerprint().len(), 64);
}

#[test]
fn p10_lock_carries_every_hard_compatibility_contract() {
    let lock = ReleaseCompatibilityLock::from_json_str(P10_LOCK_JSON.trim()).unwrap();
    assert_eq!(
        lock.compatibility().feature_contract(),
        &[
            "cogno:hard-gates",
            "flat-attention:m24-caller-owned-grouped-forward",
            "flat-attention:wgpu",
            "rsi:public-features",
            "rsi:scirust",
            "scirust-gpu:flat-attention",
            "scirust-gpu:p10.3-resident-grouped-training",
            "scirust-sciagent:flat-attention",
            "scirust-sciagent:id-opt-shared-rule-table",
            "scirust-sgemm:elastic-measured-evidence",
            "scirust-sgemm:no-unqualified-winner",
            "tokenizer:canonical-parity",
        ]
    );
}

#[test]
fn p10_evidence_records_all_slices_without_promoting_an_sgemm_winner() {
    let lock = ReleaseCompatibilityLock::from_json_str(P10_LOCK_JSON.trim()).unwrap();
    let evidence = lock.qualification_evidence();

    for slice in ["p10.1", "p10.2", "p10.3", "p10.4", "p10.5"] {
        assert!(
            evidence
                .iter()
                .any(|record| record.starts_with(&format!("scirust:{slice}:"))),
            "missing qualification evidence for {slice}"
        );
    }

    let sgemm = evidence
        .iter()
        .find(|record| record.starts_with("scirust:p10.5:"))
        .expect("missing P10.5 SGEMM evidence");
    assert!(sgemmm_has_no_promoted_winner(sgemm));

    assert!(
        evidence.iter().any(|record| record.starts_with("flat:m24:")),
        "missing exact FLAT M24 evidence"
    );
    assert!(
        evidence.iter().any(|record| record.starts_with("rsi:p9.2:")),
        "missing frozen RSI predecessor evidence"
    );
}

fn sgemmm_has_no_promoted_winner(record: &str) -> bool {
    record.split(':').any(|field| field == "winner=none")
}

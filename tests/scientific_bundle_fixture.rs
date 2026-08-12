use rsi::paper_science::ScientificBundle;

const FIXTURE: &str = include_str!("fixtures/scientific_bundle_v1.json");

#[test]
fn canonical_bundle_v1_is_accepted_by_rsi() {
    let bundle = ScientificBundle::parse(FIXTURE).expect("canonical scientific bundle v1");
    assert_eq!(bundle.paper_id, "fixture-paper-1");
    assert_eq!(bundle.claims.len(), 2);
    assert_eq!(bundle.provenance.generator, "fixture");

    let goals = bundle.directive_goals("src/kernel.rs", 3);
    assert_eq!(goals.len(), 1);
    assert!(goals[0].contains("method-fixture-1"));
    assert!(goals[0].contains("build+tests+benchmark"));
}

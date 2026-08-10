# Bounded PatchSet proposal contract

This document freezes the P2.2 boundary between an engineering proposer and later RSI evaluation stages.

## Proposal envelope

A multi-file proposer emits strict JSON with an ordered `operations` array and an optional `rationale`. Supported operation kinds are `modify_exact`, `create`, and `delete`.

```json
{
  "operations": [
    {
      "kind": "modify_exact",
      "path": "src/lib.rs",
      "expected": "old exact text",
      "replacement": "new text"
    },
    {
      "kind": "create",
      "path": "src/new.rs",
      "content": "new file contents"
    },
    {
      "kind": "delete",
      "path": "obsolete.txt",
      "expected_sha256": "<64 lowercase hex>"
    }
  ],
  "rationale": "one coherent engineering change"
}
```

The envelope is not trusted. `BoundedProposal` accepts it only after the P2.1 `PatchSet` invariants and the P2.2 proposal budgets pass.

## Hard budgets

`ProposalBudget` has two independent limits:

- `max_operations`: maximum number of operations in one candidate;
- `max_touched_bytes`: maximum bytes the candidate may consume/replace/create/delete.

Touched bytes are measured against the actual current workspace. A `ModifyExact` operation is charged for both matched bytes and replacement bytes. A `Create` is charged for its content. A `Delete` is charged for the actual target file size, not the small SHA-256 declaration. This prevents deletion from bypassing the total-change budget.

Every operation must also target an exact path in the caller-provided allowlist. Unknown operation kinds and malformed envelopes fail closed.

## Trajectory schema v2

`PatchSetTrajectory` records the exact ordered operations plus the deterministic `PatchSet::identity()`, rationale, compile verdict, test counts, measured score, and optional evaluator output. Decoding recomputes and verifies the identity before accepting a record.

The existing single-file `Trajectory` type remains supported and converts directly to a one-operation v2 record. Its historical JSONL export was intentionally optimized for SFT and contains only `prompt`, `completion`, and `score`; because it omitted raw patch fields, P2.2 does not guess an exact patch back out of natural-language prompt text.

## Phase boundary

P2.2 does **not** change DGM parent materialization, archive ancestry, or live-tree promotion. Those are P3 responsibilities. This keeps proposal representation/data migration independently testable before cumulative candidate states are introduced.

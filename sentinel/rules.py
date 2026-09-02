# One integration change — rules produce the new Detection type:
# In run_rules(), construct Detection(
#     sentence_text=match.group(0), evidence_id=rule["id"],
#     evidence_fact=rule["correction"], source="deterministic rule",
#     contradiction_prob=1.0, severity=rule["severity"],
#     layer="rule", verdict="IMPOSSIBLE_VALUE" or "PROHIBITED_CLAIM")

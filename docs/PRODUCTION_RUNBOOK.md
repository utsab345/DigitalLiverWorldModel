# Production runbook

1. Use only approved, de-identified clinical data. Keep raw identifiers out of
   logs and manifests; store access controls and retention rules with the data
   owner.
2. Pin a model version and data manifest for every deployment.
3. Run `/health`, constraint checks, and a held-out regression suite before
   traffic is enabled.
4. Monitor `/metrics`, latency, error rate, drift, and constraint violations.
5. Route high-uncertainty or out-of-distribution cases to human review. This
   research system is not a medical device or autonomous clinical decision.

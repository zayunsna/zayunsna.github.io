AI code review vs unit tests — reproducible demonstration

helpers.py is INTENTIONALLY BROKEN. Do not use it in an application.
requirements.txt is the exact behavioral contract, NOT pip dependencies.
helpers_fixed.py satisfies the tested cases; passing tests are not a proof.

Download all .py files in this directory into one folder. With Python 3.12:
  python3.12 -m venv .venv
  source .venv/bin/activate
  python -m pip install pytest==9.2.0
  python run_checks.py --implementation broken --suite baseline
  python run_checks.py --implementation broken --suite all
  python run_checks.py --implementation fixed --suite all
  python verify_findings.py

The first two commands intentionally exit 1 (test failures):
  baseline on broken helpers: 3 failed, 5 passed
  expanded on broken helpers: 6 failed, 5 passed
  expanded on fixed helpers: 11 passed

Reviewed on 2026-10-30. Eight synthetic functions, six seeded defects, two
correct controls. The fixture, contract, baseline suite, regression suite and
reference implementation were written before either review result arrived.
This is a constructed demonstration, not a representative benchmark or a
test of AI-generated test quality. Both reviewers saw only the frozen fixture
and contract, not tests, solutions, the number of defects or each other's report.
Reviewers were separate fresh Claude Code sessions on subscription logins;
both self-reported claude-opus-5-5. The exact model was not independently
verified. Codex did not run a review because its startup update prompt blocked
readiness. No model comparison can be drawn from this demonstration.

review_a.txt and review_b.txt preserve the verbatim delivered reports, without
private runtime metadata. Counterexamples were executed by the coordinator
after review. No out-of-scope invalid input handling was graded. Both reviewers
identified the same six buggy functions and did not flag the two controls.
No per-review token usage, monetary cost or reliable review latency was measured.
Subscription capacity was used; this is not a claim of cost-free AI.

Frozen SHA256:
  helpers.py: b2be22a584c289e0e1e71f014f38053a0cdfc60452ca1d9256d0d14225241ab2
  requirements.txt: ced09f5e29141a3caaf7690e5c9b3b0e9fdce08be7c20c929228348700ff1819

render_cover.py recreates the cover from the saved test output and verified
review findings. Optional rendering packages: matplotlib==3.11.0, pillow==13.0.0.
Tested: Python 3.12.13, pytest 9.2.0, macOS, Apple M5, CPU.

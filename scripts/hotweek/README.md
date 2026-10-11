# October 12–16 article experiments

The user approved all five articles on 2026-10-12. They are in their category `_posts` directories,
with publication dates in the filenames. No existing post or schedule has been replaced.
GitHub Pages still requires the user to push and rebuild on or after each publication date.

| Intended date | Post file | Experiment |
|---|---|---|
| 2026-10-12 | blog/_posts/2026-10-12-agent_log_trust.md | harmless HTML escaping / stored-record checks |
| 2026-10-13 | tips/_posts/2026-10-13-rag_document_trust.md | cached local Qwen, lexical retrieval, eight generations |
| 2026-10-14 | blog/_posts/2026-10-14-free_threaded_numpy.md | standard Python 3.15 / GIL enabled / independent tasks |
| 2026-10-15 | ds/_posts/2026-10-15-ai_productivity_selection.md | synthetic selection bias, not real worker data |
| 2026-10-16 | blog/_posts/2026-10-16-tabpfn_lightgbm_evaluation.md | one LightGBM model, two splits; no TabPFN execution |

Canonical executable sources are here. Exact public download copies are in `assets/code/hotweek/` because
the site excludes `scripts/`. When editing either copy, update the other and check byte equality.

Recorded results are in `assets/code/hotweek/results/`. Rebuild the figures from those exact inputs:

```bash
python scripts/hotweek/render_figures.py \
  --results assets/code/hotweek/results --assets assets/img
```

This requires matplotlib. Output covers are 1200 by 630 pixels. No external illustration or screenshot is used.
The log figure is an illustration of the source records, not a screenshot or a reproduction of the Inspect exploit.

To repeat experiments, use a temporary directory and separate virtual environments. Do not replace the recorded
benchmark JSON with a new run without updating the corresponding article tables and figures. Install commands in
the articles are reproduction instructions; the author run reused cached packages and model weights.

Remaining validation as of 2026-10-12:

- Full Hydejack build: blocked by DNS resolution for codeload.github.com.
- Browser-rendered log demo and themed article/list screenshots: not verified in this sandbox.
- Free-threaded build comparison: not run; interpreter download blocked by DNS. No such comparison is claimed.
- TabPFN model and output use: not authorized under our revenue-oriented use case; no model executed.
- Clean installation from public package indexes: not tested; the local runs used cached dependencies.

Commit approval was granted, but this session cannot create `.git/index.lock` because Git metadata is read-only.
No commits or pushes were made by this session. The user must commit from a writable terminal and then push.

2026-10-12 additions (Claude cross-check): `results/threads_compare/` holds an interleaved standard vs free-threaded
Python 3.15.0 run (NumPy 2.5.3, two runs each) used in the free-threaded post. The log post's screenshot
`assets/img/blog/agent_log_trust/viewer_chrome.png` is a real Chrome 154 rendering of `viewer.html`.

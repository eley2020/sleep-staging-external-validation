# Sleep staging: external validation and subject-disjoint adaptation

Research software by Jian Zhang, School of Information Science and Engineering,
East China University of Science and Technology, Shanghai, China.
Contact: jianzhang@ecust.edu.cn; ORCID: 0009-0001-8651-0524.

This repository contains code, methodology, limitations and audited aggregate
results for a four-channel quality-gated extension of AttnSleep. It does not
claim a wholly new backbone or clinically ready sleep scoring. Manuscript
drafts, raw PSG, labels, individual predictions and trained weights are excluded.

## License and attribution

Original contributions: [MIT License](LICENSE), copyright 2026 Jian Zhang.
MIT allows reuse, modification and commercial use while requiring retention of
the license/copyright notice, and provides no warranty. Upstream AttnSleep MIT
notices are preserved; see [third-party notices](THIRD_PARTY_NOTICES.md).
Software licensing does not override dataset-access or redistribution conditions.

Cite Eldele et al., IEEE TNSRE 29 (2021):809–818,
doi:10.1109/TNSRE.2021.3076234, as well as this software's `CITATION.cff`.
No publication DOI is claimed for the ongoing project manuscript.

## Installation and aggregate figure reproduction

Experiments used Python 3.12 and PyTorch 2.5.1+cu121 on NVIDIA TITAN V GPUs.
Install CUDA-compatible PyTorch for neural training and inference. The aggregate
export below does not execute a neural network or read restricted individual data.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/export_sleep_medicine_publication.py --output-dir outputs/reproduced_publication
```

The output directory must be new. Tables and figures are regenerated from
`publication/publication_metrics.json`; this re-exports audited aggregate
bootstrap intervals, not individual-level bootstrap draws. Prepared figures
and tables are in `publication/generated/`.

## Methods and valid results

See [methods](docs/sleep_medicine_complete_methods.md),
[limitations](docs/sleep_medicine_limitations.md), and
[dataset access conditions](docs/data_access_and_licenses.md).

- Source Sleep-EDF-20 accuracy 0.8419 is an internal held-out-subject result,
  not external-cohort accuracy; source and external summaries have different weighting.
- The corrected original-input ST frozen baseline is accuracy 0.7441,
  Macro-F1 0.6905 and kappa 0.6389. Seven defective input caches were corrected;
  this is a data-integrity correction, not a model improvement.
- Legacy ST adaptation/context conclusions using defective caches are excluded.
- Preprocessing sensitivity did not yield a universally passing modification.
  ST envelope gains did not generalize to ISRUC; SHHS sensitivity samples are
  not whole-cohort modified-input evaluations.
- SHHS calibration budgets require an additional 20 labeled validation subjects.
  Single-source-model adaptation gains cannot be compared directly with the
  frozen 20-model ensemble. ISRUC matched controls must be interpreted separately.

## Model-level reproduction boundaries

Acquire authorized datasets from their official providers. Start with:

```bash
.venv/bin/python scripts/prepare_sleepedf20_multichannel.py --help
.venv/bin/python scripts/train_attnsleep_multichannel_baseline.py --help
.venv/bin/python scripts/evaluate_attnsleep_external_ensemble.py --help
.venv/bin/python scripts/evaluate_attnsleep_isruc.py --help
```

Use your own dataset roots and CUDA for neural runs. Preparation rejects invalid
or wholly zero channels and refuses overwriting prepared outputs. Historical
study wrappers depend on local checkpoint/protocol inventories not distributed
here. A new model-level reproduction needs a new explicit protocol and authorized
data; exact bitwise retraining is not promised. Old statistics wrapper defaults
are historical, not current publication regeneration commands.

`CODE_EXPORT_PROVENANCE.json` records release-only path substitutions from the
locked workspace. `PUBLIC_RELEASE_PROVENANCE.json` records public-release curation;
`MANIFEST.json` hashes the curated snapshot excluding Git metadata and itself.
Scientific Python files and aggregate CSV results are unchanged from the audited v4 bundle.

## Privacy and research status

Do not commit credentials, datasets, caches, individual results or checkpoints.
The only distributed `.npy` is the fixed upstream fold-permutation helper.
These outputs are retrospective research, not expert-equivalence evidence or
an approved medical device. Institutional ethics/consent wording, current journal
requirements and final author manuscript review remain separate submission tasks.

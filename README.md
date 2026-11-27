# Agricultural coverage PPO

Train a single agricultural robot to cover a field through interaction with a physically constrained Gymnasium environment. Stable-Baselines3 PPO uses local and global maps, vehicle state and obstacle observations. The four robot configurations and procedural curriculum are bundled. The simulator counts treatment over executed motion and distinguishes vehicle-body clearance from the effective agricultural working width.

**New to the project?** Follow the [step-by-step training guide](docs/TRAINING_GUIDE.md) to install the package, train and evaluate a first policy, resume a checkpoint, or prepare the real EU field cohort.

## Quick start on Linux

Use Python 3.12 and a C++17 compiler (g++ for the accelerated backend). Run from the repository root.

```bash
git clone https://github.com/wushuaixue-boop/agricultural-coverage-ppo.git
cd agricultural-coverage-ppo
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python verify_data.py
python train.py --steps 512 --out runs/smoke
```

For the exact tested Linux CPU dependencies, install the Torch version in [the lockfile](requirements-cpu.lock) from the CPU wheel index, then run `python -m pip install -r requirements-cpu.lock`. The default V1.0 backend compiles a local shared library on first use. Choose `--backend reference` to use the Python simulator. The training commands also work from a Linux notebook terminal.

## Train, evaluate and continue

Stage 1 learns on small generated rectangles. Stage 6 mixes the training split of the 12,800-field dataset shipped here with newly generated A–D fields, including concave fields and holes. Stage 7 also samples different robots. Each command below is a separate example.

```bash
python train.py --stage 6 --workers 4 --steps 100000 --interval 25000 --out runs/mixed
python train.py --resume runs/mixed/step_100352 --steps 100000 --out runs/mixed-next
python -m ppo_coverage_planning_v1.evaluate_cli --checkpoint runs/mixed/step_100352 --dataset-manifest data/fields.json --stage 6 --n-episodes 2 --output-json runs/mixed/eval.json
python train.py --version 1.2 --stage 3 --workers 2 --steps 10240 --out runs/v12
```

Each rollout contains 512 aggregate steps. Requests round up to a complete rollout. Use the checkpoint directory named in `latest.json` or `best.json`. Supply a fresh output directory and the same version and dataset when resuming. The runner restores policy weights, optimizer, reward normalization, curriculum state and cumulative timesteps. Episodes and rollout buffers restart. Validation runs at the configured interval and at the end. TensorBoard logs are under each run directory.

## Versions

| Launcher choice | Policy interface | Backend |
|---|---|---|
| Default V1.0 | Target speed and curvature, two actions, `simple_v1` reward | Kaggle-benchmarked compiled backend or original Python |
| `--version 1.2` | Curvature only, global untreated-area cue, `v1_2_anti_circle` reward | V1.2 Python simulator |

The two versions use different observation and action contracts. Use the version that trained your checkpoint. The default V1.2 example starts a fresh policy with the registered small-map configuration. The working V1.3 experiment is outside this release. [Source revisions](provenance.json)

## Datasets

[data/fields.json](data/fields.json) contains **12,800 synthetic metric field polygons** used by the default launcher. Each family contributes 2,560 training, 320 validation and 320 held-out test fields. In total there are **10,240 training, 1,280 validation and 1,280 test fields**. Family A has a convex exterior, B a concave exterior, C a convex exterior with holes, and D a concave exterior with holes. Each record includes polygon WKB, ID, geometry hash, split, area and obstacle count. All splits use the same procedural shape generators. The test split measures transfer to other generated examples within those shape families.

The original eight fields retain their IDs, geometry and training or validation assignments. To resume a run bound to the original dataset hash, pass `--data data/smoke8/fields.json`. For new mixed-field runs use `--stage 6`. Stage 1 generates small rectangles independently of the manifest. The training budget counts environment steps, so a larger field pool does not require visiting every field in one training run.

Run `python verify_data.py` to check geometry, family classification, unique IDs and hashes, split counts, and the original eight records. To regenerate with the pinned dependencies, run `python generate_fields.py` and compare the resulting checksum with the committed [checksum](data/checksums.json). Generation uses seed 20261127 and separate random streams per family and split.

### Full real EU field dataset

Planet and Sinergise published the original European field boundaries as a GeoParquet file containing 51,398,853 records (15.5 GB). Download it from [Zenodo](https://zenodo.org/records/14229033) or [Source Cooperative](https://source.coop/planet/eu-field-boundaries/field_boundaries.parquet). The [publisher's README](https://data.source.coop/planet/eu-field-boundaries/README.md) specifies [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Credit the authors, link the license and identify changes when sharing adapted data.

```bash
curl -fL --retry 3 -C - -o field_boundaries.parquet 'https://zenodo.org/records/14229033/files/field_boundaries.parquet?download=1'
echo '9275e8b3817db07bb3c96fa72f607bb7  field_boundaries.parquet' | md5sum -c -
```

The repository's 12,800 synthetic fields are ready for PPO training. The full download contains source polygons in geographic coordinates. Our A–D classification, local metric projections and PPO splits are downstream preparation steps. The trainer expects a metric JSON field manifest and checksums. For the existing 94-field prepared package, use `--data /path/to/fields/real_fields_splits.json` on training and resume. Its split is 47 training, 17 validation, 18 test and 12 historical evaluation fields.

To build the registered real-field manifest directly from the full download, follow [the EU preparation walkthrough](docs/TRAINING_GUIDE.md#5-download-real-fields-and-make-a-training-manifest).

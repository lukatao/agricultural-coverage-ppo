# Agricultural coverage PPO

Train a single agricultural robot to cover a field through interaction with a physically constrained Gymnasium environment. Stable-Baselines3 PPO uses local and global maps, vehicle state and obstacle observations. The four robot configurations and procedural curriculum are bundled. The simulator counts treatment over executed motion and distinguishes vehicle-body clearance from the effective agricultural working width.

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

Stage 1 learns on small generated rectangles. Stage 6 mixes the fixed fields shipped here with newly generated A–D fields, including concave fields and holes. Stage 7 also samples different robots. Each command below is a separate example.

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

[data/fields.json](data/fields.json) contains **eight project-generated polygons**, one training and one validation geometry per family. A is convex, B is concave, C has a convex exterior with an obstacle, and D has a concave exterior with multiple obstacles. Coordinates are in local metres. Original WKB, IDs, areas, hole counts, geometry hashes, generation seeds and split membership are preserved. The two packaging changes to the Kaggle file are the loader key `fields` and split spelling `val`. Its upstream SHA-256 is recorded in the file. The fixed set is intended for reproducible workloads. The procedural curriculum supplies further fields during training. There is no independent test cohort in the eight-field dataset.

The 94-field real EU dataset comes from Planet Labs’ proprietary `planet_afb` product. Its recorded permission covers local use. External hosting and redistribution to teammates have not been established. A collaborator who already has an authorized copy of the corrected field package can keep it outside Git and pass `--data /path/to/fields/real_fields_splits.json` on both training and resume. Keep its checksums, metadata and splits together. The registered split is 47 training, 17 validation, 18 test and 12 historical evaluation fields. Training samples the training split. Conventional planners and expert trajectories are outside this package.

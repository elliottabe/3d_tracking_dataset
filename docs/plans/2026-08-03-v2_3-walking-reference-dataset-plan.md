# v2.3 Walking Reference Dataset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run IK over all 387 free-running walking bouts with the `fruitfly_v2.3` anatomy and pack them into a single batched HDF5 matching the format of `Fruitfly_v1_walk_1000hz_interp_padded.h5`.

**Architecture:** Make `v2.3` a usable anatomy (fix its MJX incompatibility and add the 50 STAC tracking sites), then drive the existing four-stage batch pipeline (preprocess → STAC IK → postprocess → combine) under `anatomy=v2_3`, and finally replace a notebook repack step with a committed script. The 3D keypoints already exist on disk; no vision-pipeline rerun is involved.

**Tech Stack:** Python 3.12, MuJoCo + MJX, JAX, Hydra/OmegaConf, h5py, pytest, SLURM (Hyak).

**Spec:** `docs/specs/2026-08-03-v2_3-walking-reference-dataset-design.md`

## Global Constraints

- Repo root: `/mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset`. Run all commands from there.
- Conda env: `micromamba activate 3d_tracking`.
- **Never run GPU/MJX work on the Hyak login node.** Use a compute node (`sbatch`, or `salloc` then run directly). If you are already on a GPU node, run directly — do not sbatch and idle.
- JAX env for any GPU stage:
  ```bash
  module load cuda/12.9.1
  export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
  unset LD_LIBRARY_PATH
  unset JAX_PLATFORMS
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
  ```
  When invoking a launcher by hand, prefix with `env -u JAX_PLATFORMS`.
- CPU-only checks (model compile, h5 inspection, unit tests) may set `JAX_PLATFORMS=cpu` and run anywhere.
- Anatomy name is exactly `v2_3` (underscore, matching the `configs/anatomy/<name>.yaml` + `anatomy.name` convention). Dataset name is exactly `free_running`.
- Model source of truth: `fruitfly_v2.3/fruitfly_muscles_warp.xml`. Expected compiled shape: `nq=101, nv=100, nbody=74, njnt=95`, and after Stage 0 `nu=264` with 50 tracking sites.
- Tests: plain `pytest` from repo root, files at `tests/test_<thing>.py`, plain `def test_*()` functions, `OmegaConf.create` for config inputs, `tmp_path` for file I/O. No pytest config file exists; do not add one.
- Commit after every task. Do not push.

---

## File Structure

**Create**
- `scripts/models/add_v2_3_tracking_sites.py` — inserts the 50 `tracking[...]` sites into the v2.3 XML from the anatomy config. Idempotent.
- `configs/anatomy/v2_3.yaml` — the new anatomy (copy of `v2_muscles.yaml` with 5 changes).
- `scripts/export/pack_reference_clips.py` — combined per-bout h5 → batched reference-clip h5.
- `tests/test_v2_3_anatomy.py` — model + config invariants.
- `tests/test_pack_reference_clips.py` — padding/stacking invariants.
- `tests/test_bout_summary_discovery.py` — bout-summary candidate resolution.
- `models/fruitfly_v2.3` — symlink into the body-model tree.

**Modify**
- `<body_models>/fruitfly_v2.3/fruitfly_muscles_warp.xml` — cross-repo: comment out 8 adhesion actuators, add 50 sites.
- `utils/fly_detection.py:733-740` — bout-summary candidate list.
- `scripts/batch_process_predictions.py:201` — `--dataset` choices.
- `scripts/batch_run_stac.py:223,442` — `rglob`, `--dataset` choices.
- `scripts/batch_postprocess_predictions.py:217` — `--dataset` choices.
- `configs/postprocessing/v2_3.yaml` (create) — v2.3 floor-alignment end effectors.
- `docs/running_the_pipeline.md:149-179` — replace the "v2_muscles does NOT work" section.

**Path shorthand used below**
```
REPO=/mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
BODY=/gscratch/portia/eabe/Research/MyRepos/fly_neuromech/fruitfly_body_models
DATA=/gscratch/portia/eabe/data/Johnson_lab/free_running
DATASETS=/gscratch/portia/eabe/fly_neuromech/data/datasets
```

---

## Task 1: Anatomy config `v2_3.yaml`

Authored first because Task 2's site generator reads it.

**Files:**
- Create: `configs/anatomy/v2_3.yaml`
- Test: `tests/test_v2_3_anatomy.py`

**Interfaces:**
- Produces: Hydra group `anatomy=v2_3`, resolving `anatomy.name == "v2_3"`, `anatomy.mjcf_path`, `anatomy.arena_path`, and `anatomy.model.*` (bridged to `cfg.model` by `configs/config.yaml:24`). Task 2 reads `anatomy.model.KEYPOINT_MODEL_PAIRS` and `anatomy.model.KEYPOINT_INITIAL_OFFSETS`. Tasks 5-8 pass `anatomy=v2_3` on the command line.

- [ ] **Step 1: Copy the base config**

```bash
cd $REPO
cp configs/anatomy/v2_muscles.yaml configs/anatomy/v2_3.yaml
```

- [ ] **Step 2: Apply exactly five edits to `configs/anatomy/v2_3.yaml`**

Edit A — line 3:
```yaml
name: v2_3
```

Edit B — lines 10-11 (note: the originals have a trailing space after the closing quote; drop it):
```yaml
mjcf_path: "${paths.body_model_dir}/fruitfly_v2.3/fruitfly_muscles_warp.xml"
arena_path: "${paths.body_model_dir}/fruitfly_v2.3/floor.xml"
```

Edit C — in `joint_names:`, append the two new v2.3 joints at the end of the wing/abdomen/leg list (after `tarsus1_T3_right`), with a comment:
```yaml
  # Haltere joints (new in v2.3)
  - haltere_left
  - haltere_right
```

Edit D — `model.name` (inside the `model:` block):
```yaml
  name: 'fruitfly_V2_3'
```

Edit E — `model.KEYPOINT_INITIAL_OFFSETS.T3L_TaTip`, currently `0 -0.005 0`. This is a left/right mirror bug: `T1L_TaTip` and `T2L_TaTip` are both `0 0.005 0`, and every `*R_TaTip` is `0 -0.005 0`. Correct it:
```yaml
    T3L_TaTip: 0 0.005 0
```

Leave everything else byte-identical — in particular `KP_NAMES` order, `KEYPOINT_MODEL_PAIRS` key order, `SITES_TO_REGULARIZE`, and all solver parameters.

- [ ] **Step 3: Write the failing test**

Create `tests/test_v2_3_anatomy.py`:

```python
"""Invariants for the v2_3 anatomy config and its compiled MuJoCo model.

These guard the three things that silently break a non-v1 anatomy run:
keypoint column order (STAC matches columns positionally), the MJX-incompatible
adhesion actuators, and the missing tracking[...] sites.
"""
from __future__ import annotations

from pathlib import Path

import mujoco as mj
import pytest
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
BODY = Path('/gscratch/portia/eabe/Research/MyRepos/fly_neuromech/fruitfly_body_models')
V2_3_XML = BODY / 'fruitfly_v2.3' / 'fruitfly_muscles_warp.xml'


def _cfg(name):
    return OmegaConf.load(REPO / 'configs' / 'anatomy' / f'{name}.yaml')


def _model():
    return mj.MjModel.from_xml_path(str(V2_3_XML))


def test_name_and_paths():
    c = _cfg('v2_3')
    assert c.name == 'v2_3'
    assert c.mjcf_path.endswith('fruitfly_v2.3/fruitfly_muscles_warp.xml')
    assert c.arena_path.endswith('fruitfly_v2.3/floor.xml')


def test_kp_names_order_identical_to_v1():
    """STAC matches keypoint columns POSITIONALLY once prune_model_to_available
    early-returns on an exact name-set match, so order is load-bearing."""
    assert list(_cfg('v2_3').model.KP_NAMES) == list(_cfg('v1').model.KP_NAMES)


def test_keypoint_model_pairs_key_order_identical_to_v2_muscles():
    assert (list(_cfg('v2_3').model.KEYPOINT_MODEL_PAIRS)
            == list(_cfg('v2_muscles').model.KEYPOINT_MODEL_PAIRS))


def test_tatip_offsets_are_left_right_mirrored():
    """v2_muscles.yaml has T3L_TaTip mirrored onto the right side; v2_3 fixes it."""
    offs = _cfg('v2_3').model.KEYPOINT_INITIAL_OFFSETS
    for leg in ('T1', 'T2', 'T3'):
        left = [float(x) for x in str(offs[f'{leg}L_TaTip']).split()]
        right = [float(x) for x in str(offs[f'{leg}R_TaTip']).split()]
        assert left == [0.0, 0.005, 0.0], f'{leg}L_TaTip should be +y'
        assert right == [0.0, -0.005, 0.0], f'{leg}R_TaTip should be -y'


def test_every_mapped_body_exists_in_model():
    m = _model()
    bodies = {mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)}
    missing = {k: v for k, v in _cfg('v2_3').model.KEYPOINT_MODEL_PAIRS.items()
               if v not in bodies}
    assert not missing, f'bodies absent from v2.3: {missing}'


def test_declared_joint_names_exist_in_model():
    m = _model()
    joints = {mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)}
    missing = [j for j in _cfg('v2_3').joint_names if j not in joints]
    assert not missing, f'joint_names absent from v2.3: {missing}'


def test_haltere_joints_declared():
    names = list(_cfg('v2_3').joint_names)
    assert 'haltere_left' in names and 'haltere_right' in names


def test_model_kinematic_shape():
    m = _model()
    assert (m.nq, m.nv, m.nbody, m.njnt) == (101, 100, 74, 95)
```

- [ ] **Step 4: Run the test — expect the model-shape tests to pass and nothing to fail**

```bash
cd $REPO && JAX_PLATFORMS=cpu python -m pytest tests/test_v2_3_anatomy.py -v
```
Expected: all 8 PASS. The adhesion/tracking-site assertions live in Task 2's test, so this file must be green already.

If `test_declared_joint_names_exist_in_model` fails, you mistyped a joint name in Edit C. If `test_kp_names_order_identical_to_v1` fails, you accidentally reordered `KP_NAMES`.

- [ ] **Step 5: Commit**

```bash
cd $REPO
git add configs/anatomy/v2_3.yaml tests/test_v2_3_anatomy.py
git commit -m "feat(anatomy): v2_3 config (v2.3 model, haltere joints, T3L_TaTip mirror fix)"
```

---

## Task 2: Make the v2.3 model MJX-loadable and STAC-trackable

Two edits to one XML in the **`fly_neuromech` repo**, plus a symlink so `${paths.body_model_dir}` resolves.

**Files:**
- Create: `scripts/models/add_v2_3_tracking_sites.py`
- Create: `models/fruitfly_v2.3` (symlink)
- Modify: `$BODY/fruitfly_v2.3/fruitfly_muscles_warp.xml` (cross-repo)
- Test: `tests/test_v2_3_anatomy.py` (extend)

**Interfaces:**
- Consumes: `configs/anatomy/v2_3.yaml` from Task 1 — specifically `anatomy.model.KEYPOINT_MODEL_PAIRS` (keypoint → parent body) and `anatomy.model.KEYPOINT_INITIAL_OFFSETS` (keypoint → `"x y z"` string).
- Produces: a compiled v2.3 model with `nu=264`, zero `mjTRN_BODY` actuators, and 50 sites named `tracking[<KP_NAME>]`. Every later stage depends on this.

- [ ] **Step 1: Create the symlink**

```bash
cd $REPO
ln -s $BODY/fruitfly_v2.3 models/fruitfly_v2.3
ls -l models/fruitfly_v2.3/fruitfly_muscles_warp.xml   # must resolve
```

- [ ] **Step 2: Comment out the 8 adhesion actuators**

These are `mjTRN_BODY` transmissions; `mjx.put_model` raises `NotImplementedError: [<mjtTrn.mjTRN_BODY: 5>] not supported`, which would hard-fail `postprocess_stac_data.py:818`. Comment rather than delete, matching `fruitfly_v2.1_muscles.xml:2293-2300`, so the muscle model keeps them for non-MJX dynamics work.

In `$BODY/fruitfly_v2.3/fruitfly_muscles_warp.xml` there are four 2-line blocks — at lines 2392-2393, 2539-2540, 2628-2629, 2723-2724. Wrap each pair. For example the first becomes:

```xml
    <!-- Adhesion actuators are mjTRN_BODY, which MJX does not support
         (mjx.put_model raises NotImplementedError). Commented out so the
         model can be used for IK/FK; restore for non-MJX dynamics work.
    <adhesion name="adhere_labrum_left" class="adhesion_labrum" body="labrum_left"/>
    <adhesion name="adhere_labrum_right" class="adhesion_labrum" body="labrum_right"/>
    -->
```

Apply the same treatment to the `tarsal_claw_T1_*`, `T2_*` and `T3_*` pairs. Do **not** touch the `<default class="adhesion*">` blocks (they define no actuators) or the `adhesion-collision` geom class (needed for FK geometry).

Verify:
```bash
cd $REPO && JAX_PLATFORMS=cpu python -c "
import mujoco as mj
m=mj.MjModel.from_xml_path('models/fruitfly_v2.3/fruitfly_muscles_warp.xml')
n=int((m.actuator_trntype==mj.mjtTrn.mjTRN_BODY).sum())
print('nu',m.nu,'body_trn',n,'nq',m.nq,'nbody',m.nbody)
assert (m.nu,n,m.nq,m.nbody)==(264,0,101,74), 'unexpected'
print('OK')"
```
Expected: `nu 264 body_trn 0 nq 101 nbody 74` then `OK`.

- [ ] **Step 3: Write the site-insertion script**

Create `scripts/models/add_v2_3_tracking_sites.py`:

```python
"""Insert the 50 STAC ``tracking[...]`` sites into the v2.3 MJCF.

STAC creates its own marker sites at runtime (stac.py _create_body_sites), so IK
does not need these. Two other stages read them out of the XML and DO:

  * preprocess_keypoints_for_ik.py:256 selects sites via ``'tracking[' in name``
    to fix the keypoint column order and the rest-pose Procrustes reference.
    With zero tracking sites it does not raise -- it reports every node
    unmatched and produces a degenerate reorder.
  * postprocess_stac_data.py:826 selects via ``'tracking' in name`` for the
    egocentric outputs, silently yielding empty (T, 0, 3) arrays.

Positions come from the anatomy config's KEYPOINT_INITIAL_OFFSETS rather than
from the v2.1 XML, so the XML sites and STAC's runtime sites cannot drift apart.

Idempotent: re-running replaces any existing tracking[...] sites.

Usage:
    python scripts/models/add_v2_3_tracking_sites.py \
        --xml models/fruitfly_v2.3/fruitfly_muscles_warp.xml \
        --anatomy configs/anatomy/v2_3.yaml
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from omegaconf import OmegaConf

SITE_RE = re.compile(r'[ \t]*<site name="tracking\[[^\]]*\]"[^>]*/>[ \t]*\n')


def build_site_xml(kp_name: str, pos: str) -> str:
    return f'      <site name="tracking[{kp_name}]" pos="{pos}" size="0.01" group="3"/>\n'


def strip_existing(xml: str) -> tuple[str, int]:
    """Remove any tracking[...] sites so the script is idempotent."""
    n = len(SITE_RE.findall(xml))
    return SITE_RE.sub('', xml), n


def insert_sites(xml: str, pairs: dict, offsets: dict) -> tuple[str, int]:
    """Insert one <site> per keypoint as the first child of its parent body.

    Matches ``<body name="X" ...>`` and inserts immediately after that tag.
    """
    by_body: dict[str, list[str]] = {}
    for kp, body in pairs.items():
        by_body.setdefault(body, []).append(
            build_site_xml(kp, str(offsets[kp]).strip()))

    inserted = 0
    for body, site_lines in by_body.items():
        pat = re.compile(r'(<body name="' + re.escape(body) + r'"[^>]*>[ \t]*\n)')
        m = pat.search(xml)
        if not m:
            raise SystemExit(f'ERROR: body "{body}" not found in XML')
        xml = xml[:m.end()] + ''.join(site_lines) + xml[m.end():]
        inserted += len(site_lines)
    return xml, inserted


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--xml', required=True, type=Path)
    ap.add_argument('--anatomy', required=True, type=Path)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.anatomy)
    pairs = OmegaConf.to_container(cfg.model.KEYPOINT_MODEL_PAIRS)
    offsets = OmegaConf.to_container(cfg.model.KEYPOINT_INITIAL_OFFSETS)

    missing = [k for k in pairs if k not in offsets]
    if missing:
        raise SystemExit(f'ERROR: no KEYPOINT_INITIAL_OFFSETS for {missing}')

    xml = args.xml.read_text()
    xml, removed = strip_existing(xml)
    xml, inserted = insert_sites(xml, pairs, offsets)
    args.xml.write_text(xml)

    print(f'removed {removed} pre-existing tracking sites; inserted {inserted}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 4: Run it**

```bash
cd $REPO
mkdir -p scripts/models
python scripts/models/add_v2_3_tracking_sites.py \
    --xml models/fruitfly_v2.3/fruitfly_muscles_warp.xml \
    --anatomy configs/anatomy/v2_3.yaml
```
Expected: `removed 0 pre-existing tracking sites; inserted 50`

Run it a second time to confirm idempotence. Expected: `removed 50 pre-existing tracking sites; inserted 50`.

- [ ] **Step 5: Add the model-state tests**

Append to `tests/test_v2_3_anatomy.py`:

```python
def test_no_body_transmission_actuators():
    """mjTRN_BODY (adhesion) actuators make mjx.put_model raise."""
    m = _model()
    assert int((m.actuator_trntype == mj.mjtTrn.mjTRN_BODY).sum()) == 0
    assert m.nu == 264


def test_has_50_tracking_sites_matching_config():
    m = _model()
    names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_SITE, i) for i in range(m.nsite)]
    tracking = [n for n in names if n and n.startswith('tracking[')]
    assert len(tracking) == 50
    got = {n[len('tracking['):-1] for n in tracking}
    assert got == set(_cfg('v2_3').model.KP_NAMES)


def test_tracking_sites_attached_to_mapped_bodies():
    m = _model()
    pairs = _cfg('v2_3').model.KEYPOINT_MODEL_PAIRS
    for kp, body in pairs.items():
        sid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, f'tracking[{kp}]')
        assert sid >= 0, f'missing site for {kp}'
        parent = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, m.site_bodyid[sid])
        assert parent == body, f'{kp} on {parent}, expected {body}'


def test_model_loads_into_mjx():
    """The end-to-end guard: postprocess_stac_data.py calls mjx.put_model."""
    from mujoco import mjx
    mjx.put_model(_model())
```

- [ ] **Step 6: Run the full test file**

```bash
cd $REPO && JAX_PLATFORMS=cpu python -m pytest tests/test_v2_3_anatomy.py -v
```
Expected: 12 PASS.

- [ ] **Step 7: Commit both repos**

```bash
cd $REPO
git add scripts/models/add_v2_3_tracking_sites.py tests/test_v2_3_anatomy.py models/fruitfly_v2.3
git commit -m "feat(models): v2.3 IK model - MJX-compatible, 50 STAC tracking sites"

cd $BODY/..
git add fruitfly_body_models/fruitfly_v2.3/fruitfly_muscles_warp.xml
git commit -m "fix(v2.3): comment out mjTRN_BODY adhesion actuators; add STAC tracking sites

Adhesion actuators are body-transmission, which MJX does not implement, so
mjx.put_model raised NotImplementedError. Commented (not deleted) matching
fruitfly_v2.1_muscles.xml. Tracking sites generated from the v2_3 anatomy
config by 3d_tracking_dataset/scripts/models/add_v2_3_tracking_sites.py."
```

---

## Task 3: Bout-summary discovery

`utils/fly_detection.py:739` hardcodes `f'{dataset}_bouts_summary.csv'` → `free_running_bouts_summary.csv`, which matches **1 of 23** dirs. The batch driver builds its own command line, so no Hydra override can fix this from outside.

**Files:**
- Modify: `utils/fly_detection.py:733-740`
- Test: `tests/test_bout_summary_discovery.py`

**Interfaces:**
- Produces: `find_bouts_csv(folder: Path, dataset: str) -> str` returning the **basename** of the first existing candidate; raises `FileNotFoundError` listing all candidates if none exist. `detect_flies` uses it to populate `fly_info['bouts_csv']`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_bout_summary_discovery.py`:

```python
"""Bout-summary filename resolution across the free_running session groups.

On disk the same dataset uses three different names:
  session1_7 / session10      -> free_walking_bouts_summary.csv
  session11 (6 dirs)          -> free_running_bout_summary.csv   (singular)
  session11/Predictions_3D_34292437 -> free_running_bouts_summary.csv
"""
from __future__ import annotations

import pytest

from utils.fly_detection import find_bouts_csv


def test_prefers_canonical_plural_name(tmp_path):
    (tmp_path / 'free_running_bouts_summary.csv').write_text('x')
    assert find_bouts_csv(tmp_path, 'free_running') == 'free_running_bouts_summary.csv'


def test_falls_back_to_singular_bout(tmp_path):
    (tmp_path / 'free_running_bout_summary.csv').write_text('x')
    assert find_bouts_csv(tmp_path, 'free_running') == 'free_running_bout_summary.csv'


def test_falls_back_to_legacy_free_walking_name(tmp_path):
    (tmp_path / 'free_walking_bouts_summary.csv').write_text('x')
    assert find_bouts_csv(tmp_path, 'free_running') == 'free_walking_bouts_summary.csv'


def test_canonical_wins_when_several_present(tmp_path):
    (tmp_path / 'free_running_bouts_summary.csv').write_text('x')
    (tmp_path / 'free_walking_bouts_summary.csv').write_text('x')
    assert find_bouts_csv(tmp_path, 'free_running') == 'free_running_bouts_summary.csv'


def test_ignores_the_OLD_prefixed_file(tmp_path):
    """session10 dirs carry OLD_walking_bouts_summary.csv alongside the real one."""
    (tmp_path / 'OLD_walking_bouts_summary.csv').write_text('x')
    with pytest.raises(FileNotFoundError):
        find_bouts_csv(tmp_path, 'free_running')


def test_raises_listing_candidates_when_none_found(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        find_bouts_csv(tmp_path, 'free_running')
    assert 'free_running_bouts_summary.csv' in str(e.value)


def test_other_datasets_still_use_canonical_name(tmp_path):
    (tmp_path / 'courtship_bouts_summary.csv').write_text('x')
    assert find_bouts_csv(tmp_path, 'courtship') == 'courtship_bouts_summary.csv'
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd $REPO && python -m pytest tests/test_bout_summary_discovery.py -v
```
Expected: FAIL — `ImportError: cannot import name 'find_bouts_csv'`.

- [ ] **Step 3: Implement**

Add to `utils/fly_detection.py`, above `detect_flies`:

```python
def find_bouts_csv(folder, dataset: str) -> str:
    """Return the basename of the bout-summary CSV present in ``folder``.

    The free_running dirs are inconsistent: session1_7/session10 kept the
    pre-rename ``free_walking_bouts_summary.csv``, six session11 dirs use the
    singular ``free_running_bout_summary.csv``, and one uses the canonical
    plural. Try the canonical name first so other datasets are unaffected.

    Raises FileNotFoundError (listing the candidates) rather than returning a
    name that does not exist -- a silent miss would skip the folder entirely.
    """
    from pathlib import Path

    folder = Path(folder)
    candidates = [
        f'{dataset}_bouts_summary.csv',
        f'{dataset}_bout_summary.csv',
        'free_walking_bouts_summary.csv',
    ]
    for name in candidates:
        if (folder / name).is_file():
            return name
    raise FileNotFoundError(
        f'No bout-summary CSV in {folder}. Tried: {", ".join(candidates)}')
```

Then change the single-fly branch at lines 733-740 to use it:

```python
    else:
        # Single-fly layout: data3D.csv
        flies.append({
            'fly_id': None,
            'suffix': '',
            'csv': 'data3D.csv',
            'bouts_csv': find_bouts_csv(folder, dataset),
        })
```

If `folder` is not already a parameter in scope at that point, read the enclosing `detect_flies` signature and use whatever it names the directory argument.

- [ ] **Step 4: Run to verify it passes**

```bash
cd $REPO && python -m pytest tests/test_bout_summary_discovery.py -v
```
Expected: 7 PASS.

- [ ] **Step 5: Confirm all 23 real dirs now resolve**

```bash
cd $REPO && python -c "
from pathlib import Path
from utils.fly_detection import find_bouts_csv
R=Path('$DATA')
n=0
for s in ['session1_7','session10','session11']:
    for d in sorted((R/s).glob('Predictions_3D_*')):
        print(f'{s}/{d.name}: {find_bouts_csv(d, \"free_running\")}'); n+=1
print('total', n)"
```
Expected: 23 lines, no exception, `total 23`.

- [ ] **Step 6: Commit**

```bash
cd $REPO
git add utils/fly_detection.py tests/test_bout_summary_discovery.py
git commit -m "fix(fly_detection): resolve bout-summary CSV across free_running naming variants"
```

---

## Task 4: Unblock the batch drivers for `free_running`

Three argparse `choices` lists exclude `free_running` (argparse exits 2), and `batch_run_stac.py` uses non-recursive `glob` where its siblings use `rglob`.

**Files:**
- Modify: `scripts/batch_process_predictions.py:201`
- Modify: `scripts/batch_run_stac.py:223`, `:442`
- Modify: `scripts/batch_postprocess_predictions.py:217`

**Interfaces:**
- Produces: `--dataset free_running` accepted by all three drivers; `batch_run_stac.py --base-dir $DATA` discovers all 23 `Predictions_3D_*` dirs.

- [ ] **Step 1: Widen the three choices lists**

`scripts/batch_process_predictions.py:201` and `scripts/batch_postprocess_predictions.py:217` currently read:
```python
        choices=['', 'courtship', 'amputation'],
```
Change both to:
```python
        choices=['', 'courtship', 'amputation', 'stationary', 'free_running', 'muscle_imaging'],
```

`scripts/batch_run_stac.py:442` currently reads:
```python
        choices=['', 'courtship', 'stationary', 'amputation'],
```
Change to:
```python
        choices=['', 'courtship', 'stationary', 'amputation', 'free_running', 'muscle_imaging'],
```

The added names are exactly the files present in `configs/dataset/` (minus `BDN2_headless`, which has no batch path).

- [ ] **Step 2: Make stac discovery recursive**

`scripts/batch_run_stac.py:223`:
```python
        candidate_folders = sorted(base_dir.glob("Predictions_3D_*"))
```
becomes:
```python
        # rglob (not glob) to match batch_process_predictions.py:55 and
        # batch_postprocess_predictions.py:46 -- the prediction dirs live one
        # level below the dataset root (free_running/session11/Predictions_3D_*).
        candidate_folders = sorted(base_dir.rglob("Predictions_3D_*"))
```

- [ ] **Step 3: Verify discovery, without running any work**

```bash
cd $REPO
python scripts/batch_process_predictions.py --dataset free_running --anatomy v2_3 \
    --paths hyak --base-dir $DATA --dry-run
python scripts/batch_run_stac.py --dataset free_running --anatomy v2_3 \
    --paths hyak --base-dir $DATA --dry-run
python scripts/batch_postprocess_predictions.py --dataset free_running --anatomy v2_3 \
    --paths hyak --base-dir $DATA --dry-run
```
Expected: no argparse error from any of them, and each reports **23** `Predictions_3D_*` folders. `batch_run_stac.py` will report 0 processable items (no `preprocessed_bout_v2_3_free_running.h5` yet) — that is correct at this point; what matters is that it found 23 folders rather than 0.

- [ ] **Step 4: Commit**

```bash
cd $REPO
git add scripts/batch_process_predictions.py scripts/batch_run_stac.py scripts/batch_postprocess_predictions.py
git commit -m "fix(batch): allow --dataset free_running; make batch_run_stac discovery recursive"
```

---

## Task 5: Postprocessing config for v2.3

**Files:**
- Create: `configs/postprocessing/v2_3.yaml`

**Interfaces:**
- Produces: Hydra group `postprocessing=v2_3`, identical to `default` except the floor-alignment end-effector names.

- [ ] **Step 1: Create the config**

`configs/dataset/free_running.yaml` pulls in `/postprocessing: default`. Rather than edit the shared default, add a v2.3 variant selected on the command line.

Create `configs/postprocessing/v2_3.yaml`:

```yaml
# @package _global_.postprocessing

# v2.3 postprocessing: identical to `default` except the floor-alignment end
# effectors, which v1 calls claw_* and v2.x calls tarsal_claw_*.
#
# The v1 names happen to still work under v2.3 because
# postprocess_stac_data.py:590-593 does a SUBSTRING match and
# 'claw_T1_left' in 'tarsal_claw_T1_left' is True -- but depending on that
# accident is how a silent floor-height bug gets introduced later.
defaults:
  - default
  - _self_

floor_alignment:
  end_effector_names:
    - tarsal_claw_T1_left
    - tarsal_claw_T1_right
    - tarsal_claw_T2_left
    - tarsal_claw_T2_right
    - tarsal_claw_T3_left
    - tarsal_claw_T3_right
```

- [ ] **Step 2: Verify it resolves**

```bash
cd $REPO && python test_configs.py paths=hyak dataset=free_running anatomy=v2_3 \
    postprocessing=v2_3 2>&1 | grep -A8 floor_alignment
```
Expected: the six `tarsal_claw_*` names, and `source_hz: 800.0` / `target_hz: 1000.0` / `method: cubic` still inherited from `default`.

- [ ] **Step 3: Commit**

```bash
cd $REPO
git add configs/postprocessing/v2_3.yaml
git commit -m "feat(configs): v2_3 postprocessing with tarsal_claw floor-alignment end effectors"
```

---

## Task 6: Run preprocessing over all 23 dirs

First stage that touches real data. CPU-heavy but not GPU-bound; `alignment_render` needs EGL, which the compute nodes have.

**Files:** none modified — this is an execution task with a hard verification gate.

**Interfaces:**
- Consumes: Tasks 1-5.
- Produces: `preprocessing/preprocessed_bout_v2_3_free_running.h5` in each of 23 dirs, each containing `bout_NNN/{keypoints,kp_names,orig_keypoints,skeleton_edges,alignment_info}` plus `info/{clip_lengths,fly_ids,source_flies}`.

- [ ] **Step 1: Dry run**

```bash
cd $REPO
python scripts/batch_process_predictions.py --dataset free_running --anatomy v2_3 \
    --paths hyak --base-dir $DATA --dry-run
```
Expected: 23 folders listed, none reporting `missing_inputs`. If any does, Task 3 is incomplete — fix before continuing.

- [ ] **Step 2: Run it on a compute node**

If not already on one: `salloc --account=portia --partition=gpu-l40s --gpus=1 --cpus-per-task=8 --mem=64G --time=4:00:00`

```bash
cd $REPO
micromamba activate 3d_tracking
module load cuda/12.9.1
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
unset LD_LIBRARY_PATH; unset JAX_PLATFORMS
python scripts/batch_process_predictions.py --dataset free_running --anatomy v2_3 \
    --paths hyak --base-dir $DATA 2>&1 | tee /tmp/preprocess_v2_3.log
```
Per-folder timeout is 600 s (`batch_process_predictions.py:174`). If a folder times out, run that one folder alone with `--base-dir <that dir>` and inspect.

- [ ] **Step 3: HARD GATE — verify per-dir bout counts**

```bash
cd $REPO && python -c "
import h5py
from pathlib import Path
EXPECT = {
 'session1_7/Predictions_3D_20260202-171900':56,'session1_7/Predictions_3D_20260203-103416':4,
 'session1_7/Predictions_3D_20260203-164328':55,'session1_7/Predictions_3D_20260204-062628':22,
 'session1_7/Predictions_3D_20260205-111851':4,'session1_7/Predictions_3D_20260209-094844':11,
 'session1_7/Predictions_3D_20260209-142412':10,
 'session10/Predictions_3D_20260309-002459':14,'session10/Predictions_3D_20260331-145742':13,
 'session10/Predictions_3D_20260331-200523':26,'session10/Predictions_3D_20260401-012417':17,
 'session10/Predictions_3D_20260401-062914':18,'session10/Predictions_3D_20260401-115616':14,
 'session10/Predictions_3D_20260401-172928':8,'session10/Predictions_3D_20260401-231909':33,
 'session10/Predictions_3D_20260402-090606':14,
 'session11/Predictions_3D_34292434':14,'session11/Predictions_3D_34292436':10,
 'session11/Predictions_3D_34292438':7,'session11/Predictions_3D_34292439':1,
 'session11/Predictions_3D_34292440':13,'session11/Predictions_3D_34292441':8,
 'session11/Predictions_3D_34292437':15,
}
R=Path('$DATA'); bad=[]; tot=0
for rel,exp in EXPECT.items():
    p=R/rel/'preprocessing'/'preprocessed_bout_v2_3_free_running.h5'
    if not p.exists(): bad.append((rel,'MISSING',exp)); continue
    with h5py.File(p) as f: n=len([k for k in f if k.startswith('bout')])
    tot+=n
    if n!=exp: bad.append((rel,n,exp))
print('total bouts:',tot,'(expected 387)')
for b in bad: print('MISMATCH',b)
assert not bad and tot==387, 'GATE FAILED'
print('GATE PASSED')"
```
Expected: `total bouts: 387` then `GATE PASSED`.

**If the gate fails:** do not continue. A per-dir mismatch means bout selection changed under `anatomy=v2_3`. Most likely cause is a different `bouts_csv` being picked (check Task 3's resolution for that dir) or filtering dropping bouts (compare `filtering` settings). Reconcile before Stage 4 — a changed bout set breaks comparability with the v1 and v2_muscles datasets.

- [ ] **Step 4: Spot-check one file's shape**

```bash
cd $REPO && python -c "
import h5py
p='$DATA/session1_7/Predictions_3D_20260202-171900/preprocessing/preprocessed_bout_v2_3_free_running.h5'
with h5py.File(p) as f:
    b=f['bout_000']
    print('keypoints', b['keypoints'].shape)
    print('info keys', list(f['info']))
    assert b['keypoints'].shape[1:]==(50,3)
print('OK')"
```
Expected: `keypoints (T, 50, 3)`, `info keys ['clip_lengths', 'fly_ids', 'source_flies']`, `OK`.

- [ ] **Step 5: Commit the log only (data lives outside the repo)**

```bash
cd $REPO
git commit --allow-empty -m "chore: preprocessing complete for anatomy=v2_3 (387 bouts across 23 dirs)"
```

---

## Task 7: STAC IK + postprocess + combine

The long compute stage. GPU required; 6 h timeout per dir for STAC.

**Files:** none modified.

**Interfaces:**
- Consumes: Task 6's 23 preprocessed files.
- Produces: `<dir>/stac/Fruitfly_ik_v2_3_free_running.h5` (per dir), `<dir>/postprocessing/ik_output_v2_3_free_running.h5` (per dir), and `$DATA/v2_3/ik_output_combined_v2_3_free_running{,_interpolated}.h5`. Task 8 consumes the `_interpolated` file.

- [ ] **Step 1: Dry-run STAC discovery**

```bash
cd $REPO
python scripts/batch_run_stac.py --dataset free_running --anatomy v2_3 \
    --paths hyak --base-dir $DATA --dry-run
```
Expected: 23 items queued (it now finds `preprocessed_bout_v2_3_free_running.h5` in each).

- [ ] **Step 2: Run STAC IK**

```bash
cd $REPO
micromamba activate 3d_tracking
module load cuda/12.9.1
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
unset LD_LIBRARY_PATH; unset JAX_PLATFORMS
python scripts/batch_run_stac.py --dataset free_running --anatomy v2_3 \
    --paths hyak --base-dir $DATA --gpu-mem-fraction 0.9 2>&1 | tee /tmp/stac_v2_3.log
```

For a SLURM-parallel run instead (one job per dir + an `afterok` combine job):
```bash
python ./scripts/slurm_run.py --dataset free_running --anatomy v2_3 --paths hyak \
    --base-dir $DATA --steps stac,postprocess --partition gpu-l40s --dry-run
```
Drop `--dry-run` once the printed jobs look right.

- [ ] **Step 3: Verify STAC output shape — this is the v2.3 DOF check**

```bash
cd $REPO && python -c "
import h5py, glob
g=sorted(glob.glob('$DATA/*/Predictions_3D_*/stac/Fruitfly_ik_v2_3_free_running.h5'))
print('files:',len(g))
with h5py.File(g[0]) as f:
    print({k: f[k].shape for k in ['qpos','qvel','xpos','xquat','kp_data']})
    assert f['qpos'].shape[1]==101, f\"qpos {f['qpos'].shape}\"
    assert f['xpos'].shape[1:]==(74,3)
    assert f['names_qpos'].shape[0]==101
assert len(g)==23, g
print('OK')"
```
Expected: 23 files; `qpos (T,101)`, `qvel (T,100)`, `xpos (T,74,3)`, `xquat (T,74,4)`, `kp_data (T,150)`; `OK`.

- [ ] **Step 4: Run postprocess**

```bash
cd $REPO
python scripts/batch_postprocess_predictions.py --dataset free_running --anatomy v2_3 \
    --paths hyak --base-dir $DATA 2>&1 | tee /tmp/postproc_v2_3.log
```

Note: `batch_postprocess_predictions.py` does not forward a `postprocessing=` group override. If the six `tarsal_claw_*` names from Task 5 are not being applied, add `postprocessing=v2_3` to the `cmd` list at `batch_postprocess_predictions.py:148-155`, or run `postprocess_stac_data.py` per folder with `postprocessing=v2_3` on the command line.

- [ ] **Step 5: Verify egocentric arrays are NOT empty — the §2.2 silent-garbage gate**

```bash
cd $REPO && python -c "
import h5py, glob
g=sorted(glob.glob('$DATA/*/Predictions_3D_*/postprocessing/ik_output_v2_3_free_running.h5'))
print('files:',len(g))
with h5py.File(g[0]) as f:
    b=[k for k in f if k.startswith('bout')][0]
    ego=f[b]['xpos_egocentric']; print('xpos_egocentric', ego.shape)
    assert ego.shape[1]==50, 'EMPTY -> tracking sites missing from the model (Task 2)'
    assert f[b]['qpos'].shape[1]==101
assert len(g)==23
print('OK')"
```
Expected: `xpos_egocentric (T, 50, 3)` and `OK`. A shape of `(T, 0, 3)` means Task 2's tracking sites did not land — go back and fix, do not proceed.

- [ ] **Step 6: Combine**

```bash
cd $REPO
python scripts/combine_data.py paths=hyak dataset=free_running anatomy=v2_3 \
    +base_dir=$DATA
```
Note the leading `+` on `base_dir` — it is not in `config.yaml`, so a bare `base_dir=` errors.

Expected console: `Found 23 files`. Outputs land in `$DATA/v2_3/`.

- [ ] **Step 7: Verify the combined file**

```bash
cd $REPO && python -c "
import h5py
p='$DATA/v2_3/ik_output_combined_v2_3_free_running_interpolated.h5'
with h5py.File(p) as f:
    bouts=[k for k in f if k.startswith('bout')]
    print('bouts',len(bouts))
    print('info', list(f['info']))
    b=f[bouts[0]]
    print({k: b[k].shape for k in ['qpos','qvel','xpos','xquat','kp_data']})
    assert len(bouts)==387, len(bouts)
    assert b['qpos'].shape[1]==101 and b['xpos'].shape[1]==74
    assert len(f['info']['names_qpos'])==101
print('OK')"
```
Expected: `bouts 387`, `qpos (T,101)`, `xpos (T,74,3)`, `OK`.

- [ ] **Step 8: Commit**

```bash
cd $REPO
git commit --allow-empty -m "chore: v2_3 STAC IK + postprocess + combine complete (387 bouts)"
```

---

## Task 8: The packing script

Replaces cells 75-77 of `fly_neuromech/fly_mimic/notebooks/Walking_data_cleaning.ipynb`.

**Files:**
- Create: `scripts/export/pack_reference_clips.py`
- Test: `tests/test_pack_reference_clips.py`

**Interfaces:**
- Consumes: a combined interpolated h5 with `bout_NNN/{qpos,qvel,xpos,xquat,kp_data}` groups and `info/names_qpos`.
- Produces: `pack_clips(bouts: list[dict], names_qpos: list[str]) -> dict` returning a dict with keys `qpos, qvel, xpos, xquat, kp_data` (each stacked `(N, T_max, ...)`), `clip_lengths` (`(N,)` int32, **true** lengths), and `qpos_names` (list[str]). CLI: `--input <h5> --output <h5>`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_pack_reference_clips.py`:

```python
"""Padding/stacking invariants for the reference-clip packer.

Two conventions matter and are easy to get wrong:
  * padding repeats the FINAL frame (np.tile of arr[-1:]), not zeros
  * clip_lengths holds the TRUE unpadded length. The v1 reference file stores
    the padded length for every clip, which fly_mimic works around by sniffing
    repeated trailing frames; we do not reproduce that bug.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.export.pack_reference_clips import pack_clips

NAMES = [f'j{i}' for i in range(5)]


def _bout(T, seed):
    rng = np.random.default_rng(seed)
    return {
        'qpos': rng.normal(size=(T, 5)).astype(np.float32),
        'qvel': rng.normal(size=(T, 4)).astype(np.float32),
        'xpos': rng.normal(size=(T, 3, 3)).astype(np.float32),
        'xquat': rng.normal(size=(T, 3, 4)).astype(np.float32),
        'kp_data': rng.normal(size=(T, 6)).astype(np.float32),
    }


def test_shapes_are_stacked_to_max_length():
    out = pack_clips([_bout(10, 0), _bout(25, 1), _bout(17, 2)], NAMES)
    assert out['qpos'].shape == (3, 25, 5)
    assert out['qvel'].shape == (3, 25, 4)
    assert out['xpos'].shape == (3, 25, 3, 3)
    assert out['xquat'].shape == (3, 25, 3, 4)
    assert out['kp_data'].shape == (3, 25, 6)


def test_clip_lengths_are_true_unpadded_lengths():
    out = pack_clips([_bout(10, 0), _bout(25, 1), _bout(17, 2)], NAMES)
    np.testing.assert_array_equal(out['clip_lengths'], [10, 25, 17])
    assert out['clip_lengths'].dtype == np.int32


def test_padding_repeats_the_final_frame():
    bouts = [_bout(10, 0), _bout(25, 1)]
    out = pack_clips(bouts, NAMES)
    last = bouts[0]['qpos'][-1]
    for t in range(10, 25):
        np.testing.assert_allclose(out['qpos'][0, t], last)


def test_real_frames_are_untouched():
    bouts = [_bout(10, 0), _bout(25, 1)]
    out = pack_clips(bouts, NAMES)
    np.testing.assert_allclose(out['qpos'][0, :10], bouts[0]['qpos'])
    np.testing.assert_allclose(out['xquat'][1], bouts[1]['xquat'])


def test_longest_clip_is_not_padded():
    bouts = [_bout(10, 0), _bout(25, 1)]
    out = pack_clips(bouts, NAMES)
    np.testing.assert_allclose(out['qpos'][1], bouts[1]['qpos'])


def test_qpos_names_passed_through():
    out = pack_clips([_bout(4, 0)], NAMES)
    assert out['qpos_names'] == NAMES


def test_rejects_names_length_mismatch():
    """qpos_names must be per-COLUMN; fly_mimic keys its root-offset logic on it."""
    with pytest.raises(ValueError, match='qpos_names'):
        pack_clips([_bout(4, 0)], NAMES[:3])


def test_single_clip_needs_no_padding():
    out = pack_clips([_bout(7, 0)], NAMES)
    assert out['qpos'].shape == (1, 7, 5)
    np.testing.assert_array_equal(out['clip_lengths'], [7])
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd $REPO && python -m pytest tests/test_pack_reference_clips.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.export.pack_reference_clips'`.

- [ ] **Step 3: Implement**

`scripts/export/` already exists (it holds `export_free_walking_raw.py` and
`export_courtship_raw.py` — read the former for house style). Neither
`scripts/__init__.py` nor `scripts/export/__init__.py` exists, and the test
imports `from scripts.export.pack_reference_clips import pack_clips`, so both
are needed:

```bash
cd $REPO && touch scripts/__init__.py scripts/export/__init__.py
```

Then confirm the new package files did not break direct script execution — the
existing scripts are run as files (`python scripts/combine_data.py ...`), and
Hydra resolves `config_path="../configs"` relative to the file, so this should
be a no-op. Verify rather than assume:

```bash
cd $REPO && python test_configs.py paths=hyak dataset=free_running anatomy=v2_3 >/dev/null \
    && echo "hydra entry still resolves OK"
```
Expected: `hydra entry still resolves OK`. If it fails, drop the two
`__init__.py` files and instead load the module in the test with
`importlib.util.spec_from_file_location`.

Create `scripts/export/pack_reference_clips.py`:

```python
"""Pack a combined per-bout IK h5 into a batched reference-clip dataset.

Replaces cells 75-77 of fly_mimic/notebooks/Walking_data_cleaning.ipynb.

Output layout (consumed by fly_mimic's ReferenceClips / HDF5ReferenceClips):
    qpos         (N, T_max, nq)
    qvel         (N, T_max, nv)
    xpos         (N, T_max, nbody, 3)
    xquat        (N, T_max, nbody, 4)
    kp_data      (N, T_max, 150)
    clip_lengths (N,) int32   -- TRUE unpadded lengths (see below)
    qpos_names   group of nq scalar strings, one per COLUMN

Padding repeats the final frame, matching the reference file.

Deliberate deviation: the v1 reference file stores the PADDED length in
clip_lengths for every clip, and fly_mimic compensates with
unpadded_clip_lengths() sniffing repeated trailing frames. We store the true
lengths instead; the file still loads, and use_unpadded_clip_length becomes
unnecessary. The root attr clip_lengths_are_true records this.

Usage:
    python scripts/export/pack_reference_clips.py \
        --input  .../ik_output_combined_v2_3_free_running_interpolated.h5 \
        --output .../Fruitfly_v2_3_walk_1000hz_interp_padded.h5 \
        --anatomy v2_3
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

ARRAY_KEYS = ['qpos', 'qvel', 'xpos', 'xquat', 'kp_data']


def pack_clips(bouts: list[dict], names_qpos: list[str]) -> dict:
    """Stack per-bout arrays into (N, T_max, ...) with final-frame padding."""
    if not bouts:
        raise ValueError('no bouts to pack')

    nq = bouts[0]['qpos'].shape[1]
    if len(names_qpos) != nq:
        raise ValueError(
            f'qpos_names has {len(names_qpos)} entries but qpos has {nq} '
            f'columns; fly_mimic requires one name per column')

    lengths = [int(b['qpos'].shape[0]) for b in bouts]
    t_max = max(lengths)

    out: dict = {}
    for key in ARRAY_KEYS:
        stacked = []
        for b in bouts:
            arr = np.asarray(b[key])
            n_pad = t_max - arr.shape[0]
            if n_pad:
                pad = np.tile(arr[-1:], (n_pad,) + (1,) * (arr.ndim - 1))
                arr = np.concatenate([arr, pad], axis=0)
            stacked.append(arr)
        out[key] = np.stack(stacked, axis=0).astype(np.float32)

    out['clip_lengths'] = np.asarray(lengths, dtype=np.int32)
    out['qpos_names'] = list(names_qpos)
    return out


def _decode(values) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def load_bouts(path: Path) -> tuple[list[dict], list[str]]:
    """Read bout_NNN groups in sorted order, plus info/names_qpos."""
    with h5py.File(path, 'r') as f:
        keys = sorted(k for k in f if k.startswith('bout'))
        bouts = [{k: np.asarray(f[bk][k]) for k in ARRAY_KEYS} for bk in keys]
        info = f['info']
        raw = info['names_qpos']
        if isinstance(raw, h5py.Group):
            names = _decode([raw[str(i)][()] for i in range(len(raw))])
        else:
            names = _decode(np.asarray(raw))
    return bouts, names


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parents[2],
            text=True).strip()
    except Exception:
        return 'unknown'


def write_h5(out: dict, path: Path, attrs: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as f:
        for key in ARRAY_KEYS + ['clip_lengths']:
            f.create_dataset(key, data=out[key], compression='gzip',
                             compression_opts=5)
        grp = f.create_group('qpos_names')
        for i, name in enumerate(out['qpos_names']):
            grp[str(i)] = name
        for k, v in attrs.items():
            f.attrs[k] = v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--anatomy', default='v2_3')
    ap.add_argument('--source-hz', type=float, default=800.0)
    ap.add_argument('--target-hz', type=float, default=1000.0)
    args = ap.parse_args()

    bouts, names = load_bouts(args.input)
    out = pack_clips(bouts, names)

    write_h5(out, args.output, {
        'source_file': str(args.input),
        'anatomy': args.anatomy,
        'source_hz': args.source_hz,
        'target_hz': args.target_hz,
        'git_sha': _git_sha(),
        'clip_lengths_are_true': True,
    })

    print(f'wrote {args.output}')
    print(f'  clips={out["qpos"].shape[0]} T_max={out["qpos"].shape[1]} '
          f'nq={out["qpos"].shape[2]} nbody={out["xpos"].shape[2]}')
    print(f'  clip_lengths min={out["clip_lengths"].min()} '
          f'max={out["clip_lengths"].max()}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 4: Run to verify it passes**

```bash
cd $REPO && python -m pytest tests/test_pack_reference_clips.py -v
```
Expected: 8 PASS.

- [ ] **Step 5: Commit**

```bash
cd $REPO
git add scripts/__init__.py scripts/export/__init__.py scripts/export/pack_reference_clips.py tests/test_pack_reference_clips.py
git commit -m "feat(export): pack_reference_clips.py replacing the notebook repack step"
```

---

## Task 9: Produce the dataset and validate it end-to-end

**Files:** none modified.

**Interfaces:**
- Consumes: Task 7's combined interpolated h5, Task 8's script.
- Produces: `$DATASETS/Fruitfly_v2_3_walk_1000hz_interp_padded.h5`.

- [ ] **Step 1: Pack**

```bash
cd $REPO
python scripts/export/pack_reference_clips.py \
    --input  $DATA/v2_3/ik_output_combined_v2_3_free_running_interpolated.h5 \
    --output $DATASETS/Fruitfly_v2_3_walk_1000hz_interp_padded.h5 \
    --anatomy v2_3
```
Expected: `clips=387 T_max=<something> nq=101 nbody=74`.

- [ ] **Step 2: Verify the file against the reference format**

```bash
cd $REPO && python -c "
import h5py, numpy as np
p='$DATASETS/Fruitfly_v2_3_walk_1000hz_interp_padded.h5'
with h5py.File(p) as f:
    for k in f: print(k, getattr(f[k],'shape','GROUP(%d)'%len(f[k])))
    N,T = f['qpos'].shape[:2]
    assert f['qpos'].shape==(N,T,101)
    assert f['qvel'].shape==(N,T,100)
    assert f['xpos'].shape==(N,T,74,3)
    assert f['xquat'].shape==(N,T,74,4)
    assert f['kp_data'].shape==(N,T,150)
    assert len(f['qpos_names'])==101
    cl=np.asarray(f['clip_lengths'])
    assert cl.shape==(N,) and len(np.unique(cl))>1, 'clip_lengths should be TRUE lengths, not uniform'
    assert N==387
    print('attrs', dict(f.attrs))
print('OK')"
```
Expected: shapes as asserted, `clip_lengths` **not** uniform, `OK`.

- [ ] **Step 3: Verify padding really repeats the final real frame**

```bash
cd $REPO && python -c "
import h5py, numpy as np
p='$DATASETS/Fruitfly_v2_3_walk_1000hz_interp_padded.h5'
with h5py.File(p) as f:
    cl=np.asarray(f['clip_lengths']); i=int(np.argmin(cl)); L=int(cl[i])
    q=np.asarray(f['qpos'][i])
    if L < q.shape[0]:
        np.testing.assert_allclose(q[L:], np.tile(q[L-1:L],(q.shape[0]-L,1)))
        print(f'clip {i}: {q.shape[0]-L} padded frames all equal frame {L-1}')
print('OK')"
```
Expected: a nonzero padded-frame count and `OK`.

- [ ] **Step 4: Verify it loads through fly_mimic**

```bash
cd /gscratch/portia/eabe/Research/MyRepos/fly_neuromech && python -c "
from fly_mimic.utils.data_utils import HDF5ReferenceClips
c=HDF5ReferenceClips('$DATASETS/Fruitfly_v2_3_walk_1000hz_interp_padded.h5')
print('clip_lengths', c.clip_lengths[:5])
print('n qpos_names', len(c.qpos_names))
print('OK')"
```
Expected: loads without error, 101 names. If `HDF5ReferenceClips` takes different constructor args, read `fly_mimic/utils/data_utils.py:576` and adapt — the point is that the file opens through the real loader.

- [ ] **Step 5: Sanity-compare against v1 on shared bouts**

```bash
cd $REPO && python -c "
import h5py, numpy as np
v1='$DATASETS/Fruitfly_v1_walk_1000hz_interp_padded.h5'
v23='$DATASETS/Fruitfly_v2_3_walk_1000hz_interp_padded.h5'
with h5py.File(v1) as a, h5py.File(v23) as b:
    print('v1 ', a['qpos'].shape, 'v2_3', b['qpos'].shape)
    for n,f in (('v1',a),('v2_3',b)):
        q=np.asarray(f['qpos'][0]); print(n,'root xyz range', q[:,:3].min(0).round(3), q[:,:3].max(0).round(3))
"
```
Expected: root translations in a comparable range. A wildly different scale points at the Stage 3 rest-pose rescale — investigate before trusting the dataset.

- [ ] **Step 6: Commit**

```bash
cd $REPO
git commit --allow-empty -m "chore: Fruitfly_v2_3_walk_1000hz_interp_padded.h5 produced and validated (387 clips)"
```

---

## Task 10: Documentation

**Files:**
- Modify: `docs/running_the_pipeline.md:149-179`

- [ ] **Step 1: Replace the stale anatomy section**

The section headed `## Anatomy: `anatomy=v1` (default) works; `anatomy=v2_muscles` does NOT yet` describes blockers that this work resolved for v2.3. Replace it with:

```markdown
## Anatomy: `v1` (default) and `v2_3`

`anatomy=v1` remains the default for the per-bout SAM3 pipeline.

`anatomy=v2_3` is supported on the **old batch route** (`batch_process_predictions`
→ `batch_run_stac` → `batch_postprocess_predictions` → `combine_data`) and was
used to build `Fruitfly_v2_3_walk_1000hz_interp_padded.h5`. Two model
prerequisites, both already applied to
`models/fruitfly_v2.3/fruitfly_muscles_warp.xml`:

1. The 8 `<adhesion>` actuators are commented out. They are `mjTRN_BODY`
   transmissions, which MJX does not implement, so `mjx.put_model` (used by
   `postprocess_stac_data.py`) raises without this.
2. The 50 `tracking[<KP_NAME>]` sites are present, generated by
   `scripts/models/add_v2_3_tracking_sites.py` from `configs/anatomy/v2_3.yaml`.
   STAC creates its own marker sites at runtime, but preprocessing reads these
   from the XML to fix keypoint column order and the rest-pose reference, and
   postprocess reads them for the egocentric outputs. Without them neither
   stage errors — they silently produce degenerate output.

`anatomy=v2_muscles` still does NOT work on the per-bout pipeline: it needs a v2
CSE silhouette mesh (only `fly_v1_*` meshes exist in `models/fruitfly_cse/`) and
`configs/silhouette/default.yaml` hardcodes the v1 XML. The batch route used for
v2_3 has no silhouette stage, which is why it is unaffected.

See `docs/specs/2026-08-03-v2_3-walking-reference-dataset-design.md`.
```

- [ ] **Step 2: Run the whole test suite**

```bash
cd $REPO && JAX_PLATFORMS=cpu python -m pytest tests/ -v
```
Expected: all pass, including the 4 pre-existing test files.

- [ ] **Step 3: Commit**

```bash
cd $REPO
git add docs/running_the_pipeline.md
git commit -m "docs: v2_3 anatomy is supported on the batch route"
```

---

## Self-Review Notes

**Spec coverage:** §2.1 adhesion → Task 2 Step 2. §2.2 tracking sites → Task 2 Steps 3-5. §2.3 models/ path → Task 2 Step 1. §2.4 non-blockers → Task 5 (floor alignment), Task 1 test (keypoint order); segment calibration stays off by inheriting the default, no task needed. §3 Stages 0-7 → Tasks 1-2, 1, 5, 6, 7, 7, 7, 8-9. §4 testing → per-task tests plus Task 9 gates. §6 deliverables → all covered; the three `--dataset` choices patches and the `rglob` fix are Task 4.

**Known soft spot:** Task 7 Step 4 flags that `batch_postprocess_predictions.py` may not forward the `postprocessing=v2_3` group. The fallback (edit the `cmd` list, or run `postprocess_stac_data.py` per folder) is written into the step. This is the one place the plan cannot be fully deterministic without running it, because the v1 substring accident means the wrong config still *works* — the step therefore verifies the config took effect rather than assuming it.

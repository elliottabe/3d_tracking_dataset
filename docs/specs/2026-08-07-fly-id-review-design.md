# Fly ID Review GUI — Design

**Date:** 2026-08-07
**Status:** Approved by user (brainstorming session)
**Data root:** `/gscratch/portia/eabe/data/Johnson_lab/processed/courtship`

## Problem

Courtship processing is complete, but the fly0/fly1 identity assignment (convention:
fly0 = female, fly1 = male) has not been verified for every bout. Session0
(1 recording, 30 bouts) has per-bout `sex.json` from a manual montage review
(2026-07-17). Session1 (10 recordings, ~130 bouts) has no `sex.json` at all.
We need a fast, keyboard-driven review tool to confirm or correct identities per
bout, and a deliberate apply step that makes the on-disk data uniformly consistent
with the reviewed IDs.

## Verified facts about the data

- Each bout dir `Session*/<rec>/pose/bouts/bout_NNNNN/` contains `fly0/` and
  `fly1/`, each with `sidebyside.mp4` (~1 MB, 10 s @ 30 fps, 1248x512) and
  `sidebyside_still.png`.
- Videos are H.264 with the `moov` atom before `mdat` (faststart) in both
  sessions — browser-native and instantly seekable once byte ranges are served.
- Existing `sex.json` schema (Session0):
  `{male_fly, original_male_fly, applied_swap, confidence, method, note}`.
  `male_fly` refers to the **current on-disk dir index**.

## Decisions made during brainstorming

1. **Evidence shown:** both flies' existing `sidebyside.mp4`, side by side,
   time-synced. No new preprocessing.
2. **Access:** single-file Python **stdlib** server on Hyak + SSH port-forward
   (VS Code Remote auto-forwards). No frameworks, no Jupyter, no artifact
   (videos total ~300 MB; artifacts cap at 16 MB and cannot read the filesystem).
3. **Storage:** per-bout `sex.json` in the existing schema **and** one central
   manifest `<root>/id_review.json` recording every decision.
4. **Consistency step:** separate script that physically swaps `fly0/`↔`fly1/`
   dirs per the manifest (extends the canonicalize_session_sex.py approach).
   Strict by default: refuses to apply while pending/unsure bouts exist
   (`--allow-pending` overrides).

## Component 1: `scripts/viz/fly_id_review.py` (review server)

Single file, stdlib only (`http.server`, `json`, `pathlib`, ...).

### Startup scan

- Walk `<root>/Session*/<rec>/pose/bouts/bout_*/fly{0,1}/`.
- Build or refresh `<root>/id_review.json`:
  - New bouts enter as `status: "pending"` with the original assignment taken
    from an existing `sex.json` if present (Session0), else the default
    convention (fly0 = female, fly1 = male).
  - Rescans **never overwrite** existing decisions; they only add new bouts and
    flag bouts that disappeared.
- Bouts missing a fly dir or a video are listed with a warning badge and are
  excluded from apply.

### Manifest schema (`id_review.json`)

```json
{
  "root": "/gscratch/.../processed/courtship",
  "convention": {"female": 0, "male": 1},
  "bouts": {
    "Session1/2026_04_02_16_03_48/bout_00001": {
      "original_male_fly": 1,
      "reviewed_male_fly": 1,
      "status": "pending | confirmed | swapped | unsure",
      "source": "default | sex.json",
      "reviewed_at": "ISO-8601 or null",
      "applied": false,
      "warning": null
    }
  }
}
```

### HTTP endpoints (localhost only)

- `GET /` — embedded single-page HTML/JS UI.
- `GET /api/bouts` — full manifest.
- `POST /api/decision` — `{bout_key, reviewed_male_fly, status}`; updates the
  manifest atomically (write temp + rename) and writes the bout's `sex.json`
  immediately (`method: "manual-gui"`, `confidence: "user"`).
- `GET /media/<relpath>` — serves files under the data root **with HTTP 206
  byte-range support** (custom ~30-line Range responder; stdlib's handler lacks
  it). Path-traversal guard: resolved path must stay under the root.

### Seek-performance requirements (explicit)

1. 206 Range responses on `/media` — the browser's scrub bar depends on it.
2. Files are faststart already (verified) — no re-encode pass.
3. ~1 MB files fully buffer in <1 s through an SSH tunnel; seeking is then local.
4. UI prefetches bout N+1's two videos via hidden `<video>` elements while the
   user judges bout N, so advancing is instant.

### UI / review flow

One screen: fly0 and fly1 videos side by side, looping, muted, autoplaying,
time-synced (periodic `currentTime` re-sync). Big **M**/**F** badge over each
video showing the *current* assignment. Header: progress `n/total`, current
recording, recording filter dropdown.

Keyboard-only operation; every keystroke POSTs immediately (no manual save):

| Key | Action |
|-----|--------|
| `Enter` / `→` | confirm current assignment → status `confirmed`, advance |
| `S` | swap M/F → status `swapped`, advance |
| `U` | mark `unsure`, advance |
| `←` | previous bout |
| `Space` | play/pause both |
| `R` | restart both from 0 |
| `1` / `2` | 0.5x / 2x playback rate |
| `J` | jump to next unresolved (pending/unsure) bout |

Pending bouts already carry the original assignment, so "confirm" is just
flipping through — satisfying the "default to original" requirement.

## Component 2: `scripts/viz/apply_fly_id_review.py` (consistency step)

- Reads `id_review.json`. For every bout whose `reviewed_male_fly == 0`
  (i.e. reviewed male currently lives in `fly0/`):
  1. Swap dirs: `fly0 → .fly_swap_tmp`, `fly1 → fly0`, `.fly_swap_tmp → fly1`.
  2. Rewrite `sex.json`: `male_fly: 1`, `original_male_fly` preserved from the
     pre-review original, `applied_swap: true`, `method: "manual-gui"`.
  3. Mark `applied: true` (and `reviewed_male_fly: 1`) in the manifest.
- **Dry-run by default** — prints the full plan; `--apply` executes.
- **Strict:** refuses to run while any bout is `pending` or `unsure`;
  `--allow-pending` treats pending as confirmed-original (unsure always blocks).
- Idempotent: applied bouts are skipped on re-run.
- Scope: touches only `pose/bouts/bout_*/fly{0,1}` dirs and their `sex.json`.
  Per-recording dirs (`sam3_masks/`, `predictions_matched/`, ...) are untouched;
  any bout-indexed collision it detects is reported, not guessed at.
- End state: entire tree uniformly fly0 = female, fly1 = male.

## Error handling & safety

- All manifest writes are atomic (temp file + `os.replace`).
- Server binds `127.0.0.1` only; `/media` rejects paths escaping the root.
- Missing/partial bouts: warning badge in UI, excluded from apply.
- Apply uses per-bout try/except; a failure mid-swap leaves a `.fly_swap_tmp`
  marker that the script detects and reports on next run.

## Testing

- **Range handler:** `curl -r 0-1023` / mid-file / suffix ranges against a real
  video; assert 206, correct `Content-Range`, correct byte count.
- **Scan:** run read-only against the real tree; expect 30 (Session0) + ~130
  (Session1) bouts; Session0 originals sourced from `sex.json`.
- **Apply:** synthetic temp tree with fake bout dirs → mark swaps → run →
  verify dir contents exchanged, `sex.json` updated, idempotent on re-run.
- **Dry-run** against the real tree before any `--apply`.
- **Manual:** full review pass by the user (that's the actual job).

## Out of scope

- Re-encoding or generating new videos.
- Whole-arena two-fly renders.
- Changing upstream sexing (wing-song CV) in jarvis_jax.
- Any modification outside `pose/bouts/*/fly{0,1}` and the manifest.

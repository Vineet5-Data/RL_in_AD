# Running this project on Kaggle yourself

A from-scratch runbook: account → credentials → upload data → push a kernel → get results back,
plus how to run several kernels at once.

Everything here is specific to *this* repo. Commands are copy-pasteable from the repo root
(`C:\Users\Vinee\Desktop\AlphaEvolve_research`). Shell examples use **Git Bash**; PowerShell
differences are called out where they matter.

---

## 0. Why Kaggle at all

Kaggle gives ~30 GPU-hours/week free on T4x2, and this project's runs are 2–9h each. The
local RTX 4060 is fine for evals and smoke tests, but not for multi-hour training. The
standing rule in `CLAUDE.md`: **Kaggle runs need no permission, local runs do.**

Two hard limits shape every design decision below:

| Limit | Value | Consequence |
|---|---|---|
| Session wall-clock | ~12h, then killed | Every run must be resumable |
| GPU quota | ~30h/week | Budget before launching, especially concurrent runs |
| `/kaggle/working` output | 20GB | Delete caches before the run ends |
| Log streaming | **none** | You cannot watch progress via the API |

---

## 1. Account + API credentials

### 1.1 Account
1. Sign up at <https://www.kaggle.com>.
2. **Verify your phone number** (Settings → Phone Verification). Without this you get no
   GPU/TPU accelerators and no internet inside notebooks. This is the single most common
   "why is my kernel failing" cause for new accounts.

### 1.2 Get an API token
Go to <https://www.kaggle.com/settings> → **API** section. There are two different things
there and the difference matters:

| Credential | Button | Use |
|---|---|---|
| **API token** (preferred) | "Generate New Token" | Works with everything, incl. `kernels list --mine`, `datasets status` |
| **Legacy key** (`kaggle.json`) | "Create Legacy API Key" | Older tooling only — *silently fails* on some commands |

`run_kaggle.sh`'s `check_auth()` warns about exactly this: with only legacy `kaggle.json`,
`kaggle kernels list --mine`, `kaggle datasets status`, and paginated dataset listing fail
**without an error message** — they just return nothing. If a command mysteriously returns
empty, check which credential you're using first.

### 1.3 Install the credential

Preferred (access token):
```bash
mkdir -p ~/.kaggle
# paste the token file downloaded from the settings page:
mv ~/Downloads/access_token ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

Or as an environment variable (useful in CI):
```bash
export KAGGLE_API_TOKEN='<token>'
```

Legacy fallback:
```bash
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

> **Never commit these.** `.gitignore` should cover `kaggle.json`, `access_token`, `.env`.
> If you ever paste one into a chat or a public repo, rotate it immediately at
> <https://www.kaggle.com/settings>.

### 1.4 Install and verify the CLI
```bash
pip install --upgrade kaggle
kaggle --version
kaggle datasets list --mine | head        # should list your datasets, not error
```

If `kaggle` isn't on PATH on Windows, it lives in `%APPDATA%\Python\Python311\Scripts\`
or your conda env's `Scripts\`. Use `python -m kaggle` as a fallback.

### 1.5 Windows-only environment fixes
Put these in your shell profile. Both are baked into `run_kaggle.sh` because both caused
real, silent data loss during earlier runs:

```bash
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
```

Without them, `kaggle kernels output` hits a Windows `charmap` crash that **silently
truncates the downloaded log to 0 bytes** — you get a file, it's just empty.

---

## 2. What's already uploaded

The project uses 9 datasets. You do not need to re-upload these unless the underlying data
changes.

| Dataset | Size | Contents |
|---|---|---|
| `alphaevolve-osm-graphs` | 17MB | Porto road graph + `highway_vocab.json` |
| `alphaevolve-osm-hybrid-cities` | 87MB | tdrive / cabspotting / romataxi graphs |
| `alphaevolve-gps-processed` | 1.8GB | Porto GPS parquet (flat `part-000.parquet`) |
| `alphaevolve-gps-hybrid-cities` | 935MB | The other 4 cities, one dir each |
| `alphaevolve-precompute-cache` | 7.2GB | Porto candidate cache |
| `alphaevolve-precompute-cache-hybrid` | 1.3GB | Hybrid-cities candidate caches |
| `alphaevolve-code-research2` | 112KB | **The code**, flat layout — re-upload after every code change |
| `alphaevolve-ckpt` | 181MB | stage0 / stage1 / trained checkpoints |
| `alphaevolve-viterbi-labels` | 927MB | NK-Viterbi expert labels (5 cities) |

List them yourself:
```bash
kaggle datasets list --mine
kaggle datasets files vineetdairashri/alphaevolve-ckpt
```

---

## 3. Uploading data

### 3.1 The metadata file
Every dataset directory needs a `dataset-metadata.json` beside the files:

```json
{
  "title": "alphaevolve-viterbi-labels",
  "id": "vineetdairashri/alphaevolve-viterbi-labels",
  "licenses": [{"name": "other"}]
}
```

`id` must be `<your-username>/<slug>`. `title` and slug should match or Kaggle gets confused
on later versions.

### 3.2 Create (first time) vs version (updates)
```bash
# first upload
kaggle datasets create -p /path/to/dir --dir-mode zip

# every later update — SAME directory, new version
kaggle datasets version -p /path/to/dir -m "what changed"
```

`--dir-mode zip` matters when the directory has subdirectories. Without it, nested dirs are
flattened or rejected.

Datasets are **private by default**. Add `--public` only if you mean it.

### 3.3 Wait for `ready` before using it
Upload returns before Kaggle finishes processing. A kernel attached to a still-processing
dataset sees an empty mount.

```bash
kaggle datasets status vineetdairashri/alphaevolve-viterbi-labels   # want: ready
```

### 3.4 Uploading the code (the important one)
`upload_datasets.sh` handles this. The **flat layout rule** is non-negotiable: the dataset
root must directly contain `models/ training/ dataset/ roadgraph/ preprocessing/ hmm_baseline/`,
because notebooks do `sys.path.insert(0, CODE_ROOT)` and run `python -m training.stage2r`
with `cwd=CODE_ROOT`.

The code is spread across worktrees locally, so it must be *assembled* before upload:

```bash
cd "C:/Users/Vinee/Desktop/AlphaEvolve_research"
bash .worktrees/research2/upload_datasets.sh code     # assembles /tmp/alphaevolve-code-research2
kaggle datasets version -p /tmp/alphaevolve-code-research2 -m "describe your change"
kaggle datasets status vineetdairashri/alphaevolve-code-research2    # wait for ready
```

Expected staging output:
```
=== MODE=code: skipping the one-time OSM/GPS uploads ===
dataset/  hmm_baseline/  models/  preprocessing/  roadgraph/  training/  dataset-metadata.json
```

**Pass `code`.** Bare `upload_datasets.sh` also runs steps 1/3 and 2/3, which re-upload
~1.8GB of OSM+GPS raw data that changes approximately never. `code` skips straight to the
step you actually want. The script copies `dataset-metadata.json` in for you (from
`.worktrees/research2/input/code/`), so the staged dir is upload-ready as-is.

**Re-upload the code dataset after every code change you want a kernel to pick up.** Kernels
read the dataset, not your laptop. Forgetting this means your kernel silently runs the old
code — which looks like "my fix didn't work".

### 3.5 The Windows upload cache bug
The Windows Kaggle CLI fails on first upload with
`[Errno 2] No such file or directory: '...\\resumable\\...'` because it doesn't `mkdir -p`
its own cache dir. `run_kaggle.sh`'s `dataset_sync_retry()` auto-heals this. Manually:

```bash
mkdir -p "$(dirname '<the path from the error>')"
# then re-run the upload
```

Or just use the wrapper:
```bash
bash .worktrees/research2/run_kaggle.sh sync-dataset /tmp/alphaevolve-code-research2 code
```

---

## 4. Anatomy of a kernel

A kernel directory contains exactly two things: a notebook and `kernel-metadata.json`.

```
notebooks_relabel_arm/
├── arm.ipynb
└── kernel-metadata.json
```

```json
{
  "id": "vineetdairashri/alphaevolve-relabel-arm-t4",
  "title": "AlphaEvolve Relabel Arm T4",
  "code_file": "arm.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "machine_shape": "NvidiaTeslaT4",
  "enable_internet": true,
  "dataset_sources": [
    "vineetdairashri/alphaevolve-osm-graphs",
    "vineetdairashri/alphaevolve-code-research2",
    "vineetdairashri/alphaevolve-ckpt"
  ],
  "competition_sources": [],
  "kernel_sources": []
}
```

Field notes:

- **`id`** — changing it creates a *new* kernel. Keep it stable to keep version history.
- **`enable_gpu`** — set `false` for CPU-only work. **CPU kernels cost zero GPU quota.** The
  candidate-cache precompute and the Viterbi relabel both run CPU-only for this reason.
- **`machine_shape`** — always `NvidiaTeslaT4`. Kaggle otherwise sometimes assigns a **P100
  (sm_60), which is incompatible with the current torch image** and crashes with
  "no kernel image is available for execution". `run_kaggle.sh` forces it explicitly.
- **`enable_internet`** — needed for the `pip install` cell.
- **`dataset_sources`** — every dataset the notebook mounts. Missing one = empty mount at runtime.

Scaffold a new one:
```bash
kaggle kernels init -p my_kernel_dir     # writes a template kernel-metadata.json
```

---

## 5. Push, monitor, fetch

### 5.1 Manual (three commands)
```bash
cd .worktrees/research2/notebooks_relabel_arm
kaggle kernels push -p .
kaggle kernels status vineetdairashri/alphaevolve-relabel-arm-t4
kaggle kernels output vineetdairashri/alphaevolve-relabel-arm-t4 -p ./out
```

Status values: `QUEUED` → `RUNNING` → `COMPLETE` / `ERROR` / `CANCELLED`.

### 5.2 There is no live log
**Kaggle has no streaming log API.** `kaggle kernels output` and `kaggle kernels logs`
return *empty* while the kernel is `RUNNING`. You only get the log after it reaches a
terminal state.

Practical consequences:

- Poll `status` in a loop; don't try to tail logs.
- Build **fail-fast asserts** into the first cells (mounts present, caches found, smoke test
  passes) so a broken run dies in 2 minutes instead of wasting 4 hours.
- If you want to watch progress live, the **Kaggle web UI does show output** while running —
  the API just doesn't expose it.

A poll loop:
```bash
while true; do
  s=$(kaggle kernels status vineetdairashri/alphaevolve-relabel-arm-t4 2>&1 | tail -1)
  echo "$(date +%H:%M:%S) $s"
  case "$s" in *RUNNING*|*QUEUED*) sleep 600;; *) break;; esac
done
```

### 5.3 Automated wrapper (push + poll + pull in one)
```bash
bash .worktrees/research2/run_kaggle.sh push .worktrees/research2/notebooks_relabel_arm
```
Pushes, polls every 60s to a terminal state, then pulls output into
`notebooks_relabel_arm/../ckpt_out_<kernel-name>/` (or `..._FAILED` on error, so you can
read the log either way).

### 5.4 Downloading output
Full output:
```bash
kaggle kernels output <kernel-id> -p ./out
```

Log only (much faster — outputs can be gigabytes):
```bash
kaggle kernels output <kernel-id> -p ./out --file-pattern ".*\.log"
```

The `.log` is JSONL, one JSON object per line with a `data` field. To read it:
```bash
python -c "
import json
for ln in open('out/<kernel>.log', encoding='utf-8', errors='replace'):
    try: print(json.loads(ln).get('data',''), end='')
    except Exception: print(ln, end='')
"
```

> **Large downloads time out.** A 5GB output once timed out at 5 minutes, leaving a partial
> download. A missing file then looked like a failed city — it wasn't.
> **Read the log before concluding anything from missing artifacts.**

---

## 6. Running multiple kernels at the same time

**Different kernel `id` = different session. They run in parallel.** There is no batching
mechanism; you just push several kernels.

```bash
cd .worktrees/research2/notebooks_multicity        && kaggle kernels push -p .
cd ../notebooks_relabel                            && kaggle kernels push -p .
```

Rules and gotchas:

1. **One running version per kernel id.** Pushing a new version of a kernel that's already
   running *supersedes* it. There is no `kaggle kernels cancel` — re-pushing is the only way
   to stop a run, and you lose its progress. To run two variants at once, they need **two
   different ids** and two directories.

2. **Quota is shared, not per-kernel.** Two concurrent 4h GPU kernels burn 8h of your ~30h
   weekly budget, not 4. Concurrency buys wall-clock, never quota.

3. **The CLI exposes no quota endpoint.** You cannot check remaining GPU hours from the
   terminal. Check the web UI (Settings, or the accelerator dropdown in any notebook editor)
   **before** launching concurrent GPU runs. Running out mid-week means queued kernels
   simply never start.

4. **Concurrent GPU sessions are capped** (typically 2). Extra GPU kernels queue rather than
   fail. CPU-only kernels have a more generous cap.

5. **Free parallelism: make it CPU.** Set `enable_gpu: false` and the run costs *zero* GPU
   quota, so it can run alongside GPU work with no budget interaction at all. Both the
   candidate-cache precompute and the 9.7h Viterbi relabel were deliberately built this way.

6. **Don't share output paths.** Each kernel gets its own `/kaggle/working`, so there's no
   collision risk — but if two kernels write to the *same dataset*, version them one at a time.

---

## 7. Writing a notebook that survives Kaggle

### 7.1 Mount paths are not guessable
Datasets mount under `/kaggle/input/`, but the nesting varies. A hardcoded path that looks
right can find nothing at runtime — this cost a full debugging cycle once. **Search by
filename instead:**

```python
BASE = Path("/kaggle/input/datasets/vineetdairashri")
if not BASE.exists():
    BASE = Path("/kaggle/input")          # some mounts land flat

def mount(name):
    for p in [BASE / name, Path("/kaggle/input") / name]:
        if p.exists(): return p
    raise FileNotFoundError(f"dataset not mounted: {name}")

hit = next(Path("/kaggle/input").rglob("porto_n1658437_r50_k10.npz"), None)
```

### 7.2 Merge multi-city data with symlinks, never copies
The training code wants one `OSM_ROOT` and one `PROC_ROOT`. Datasets are separate mounts.
Symlink them together — copying 5GB wastes a quarter of the 20GB output cap and makes every
later download crawl:

```python
OSM_ROOT = Path("/kaggle/working/osm_merged"); OSM_ROOT.mkdir(exist_ok=True)
for city, src in [("porto", OSM_A), ("tdrive", OSM_B)]:
    dst = OSM_ROOT / city
    if not dst.exists(): os.symlink(src / city, dst)
```

### 7.3 Cache directory naming differs between single- and multi-city
This one silently costs hours:

| Entry point | Cache path |
|---|---|
| `train()` (single city) | `<out-dir>/cache/<source>_n..._r50_k10.npz` |
| `train_multi()` (multi city) | `<out-dir>/cache/<city>__<source>/<source>_n..._r50_k10.npz` |

Get it wrong and the candidate precompute **silently re-runs from scratch** inside your GPU
session instead of reusing the cache — expensive, and OOM-prone at Porto scale. Always
assert the cache exists before training starts:

```python
missing = [f"{c}:{s}" for c, s in CACHE_SPECS if not cache_path(c, s).exists()]
assert not missing, f"cache missing, refusing to burn GPU quota on precompute: {missing}"
```

### 7.4 Make every run resumable
Kaggle kills sessions at ~12h. Always:

```
--max-hours 8.5        # stop cleanly BEFORE the hard kill
--ckpt-every 1000      # so a kill loses at most 1000 steps
--resume <path>        # pick up where the last session stopped
```

`--max-hours` measures **training time only** — it starts after the data loaders are built
(`stage2r.py:548`). Loader construction, `pip install`, and the smoke test are all *extra*.
Budget ~0.4h of overhead on top.

To chain sessions: push this session's output checkpoint back into `alphaevolve-ckpt`, then
re-run with `--resume` pointing at the mounted input copy (`/kaggle/working` is empty on a
fresh container).

### 7.5 Subprocess output double-prints
Kaggle prints every line twice when a child process inherits the file descriptor. Pipe it
through a single print loop:

```python
def run_streamed(cmd, cwd):
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout: print(line, end="")
    p.wait()
    if p.returncode != 0: raise subprocess.CalledProcessError(p.returncode, cmd)
```

### 7.6 Clean up before the run ends
Kaggle snapshots `/kaggle/working` when the kernel finishes. Delete caches and scratch first,
or your 20GB budget goes to files you'll never read.

Gate cleanup on **whether the work actually finished**, not on a config flag — a
budget-stopped run still needs its intermediate shards to resume:

```python
ALL_DONE = all(out_path(c, s).exists() for c, s in CITY_SOURCES)
if ALL_DONE:
    shutil.rmtree(SHARDS, ignore_errors=True)
else:
    print("run incomplete -- KEEPING shards so the next run resumes from them")
```

Scratch that doesn't count against the output cap goes in **`/kaggle/temp`** (this is where
`CAND_MMAP_DIR` points by default).

---

## 8. Concrete recipes

### 8.1 Multi-city training (GPU, resumable)
```bash
cd .worktrees/research2/notebooks_multicity
kaggle kernels push -p .
kaggle kernels status vineetdairashri/alphaevolve-multi-city-t4
```
After it completes, push the checkpoint back for the next session:
```bash
kaggle kernels output vineetdairashri/alphaevolve-multi-city-t4 -p ./out
cp ./out/ckpt/stage2r_multi_*.pt /tmp/alphaevolve-ckpt/
kaggle datasets version -p /tmp/alphaevolve-ckpt -m "multicity session N"
```

### 8.2 Viterbi relabel (CPU, zero GPU quota)
```bash
cd .worktrees/research2/notebooks_relabel
kaggle kernels push -p .
```
~9.7h for all 5 cities. Resumable via per-shard files: re-running skips finished shards.

### 8.3 A/B arm (two branches, one session)
See `notebooks_relabel_arm/`. Both branches run in one kernel with separate `--out-dir`s,
identical argv except the one variable under test. Regenerate the notebook from its
generator script rather than editing JSON by hand.

### 8.4 Local smoke test before spending any Kaggle time
```bash
cd .worktrees/research2
PYTHONPATH="../data-preprocess;../HMM_baseline" python -m training.stage2r --smoke
```
Runs in ~1 min on CPU and catches most breakage before a push.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Command returns empty, no error | Legacy `kaggle.json` auth | Switch to access token |
| Downloaded log is 0 bytes (Windows) | `charmap` crash | `export PYTHONIOENCODING=utf-8 PYTHONUTF8=1` |
| `No such file or directory: ...resumable...` | Windows CLI cache-dir bug | `mkdir -p` the parent, retry |
| "no kernel image is available" | Got a P100, not a T4 | `"machine_shape": "NvidiaTeslaT4"` |
| Dataset mounts empty | Not in `dataset_sources`, or still processing | Add it; wait for `status: ready` |
| Kernel can't find your code change | Code dataset not re-uploaded | `datasets version` the code dataset |
| `kernels output` returns nothing | Kernel still `RUNNING` | Poll to terminal state first |
| Kernel SIGKILLed with no traceback | OOM | Memmap instead of materializing; cap in-memory caches |
| Push fails with 409 Conflict | Known, root cause undiagnosed | Bump to a new kernel id |
| No accelerator offered | Phone not verified | Verify in Settings |
| Files missing from output | Download timed out mid-transfer | Re-download; **read the log before assuming failure** |

---

## 10. Habits that have paid off here

1. **Write the success bar before launching.** Every run gets a written pass/fail criterion
   in advance. This has caught every bug so far — including one where a checkpoint sat at the
   end of its LR schedule and would have trained at LR≈0, producing a meaningless null result.
2. **Calibrate on a tiny sample first.** A 300-trajectory CPU pass predicted both the go/no-go
   signal and the cost of a 9.7h run, for 13 minutes of free compute.
3. **Assert before the expensive part.** Cache present, labels present, smoke passing — all
   checked before the GPU spins up.
4. **Read the log before diagnosing.** Missing artifacts have twice meant "download was
   interrupted", not "the run failed".
5. **Prefer CPU where the work is CPU-shaped.** Graph traversal and Viterbi decoding get
   nothing from a GPU or TPU. Running them CPU-only costs zero GPU quota.

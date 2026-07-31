"""Trajectory dataset: cleaned GPS sequence + per-fix road candidates -> tensors.

Each item is one cleaned (sub)trajectory (a single `traj_id`). Rows of a traj_id
are contiguous in the processed Parquet (the writer emits them in order), so the
group index is built from a single pass over the traj_id column.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dataset.config import CITY_OF_SOURCE, RetrievalConfig, SequenceConfig
from dataset.temporal import day_of_week, hour_of_day


def load_source_df(processed_root, source: str, limit_trajs: int | None = None,
                    skip_trajs: int = 0) -> pd.DataFrame:
    """Read one source's processed Parquet, ordered by (traj_id, seq).

    With `limit_trajs` set, read ONLY the first-N trajs' rows rather than loading the
    whole file and subsetting. Trajs are contiguous in the file (the writer emits them
    in order), so the first N trajs are a row prefix: scan just the traj_id column to
    find the cutoff, then batch-read that prefix. Keeps peak RAM proportional to
    limit_trajs, not the full corpus -- the old full-file load OOM-killed large-data
    runs on memory-limited hosts (Kaggle).

    `skip_trajs`: skip the first N trajs before reading (same contiguous-prefix trick,
    via a cheap single-column pass). Combined with `limit_trajs` this reads one
    traj-range chunk without ever materializing the skipped rows -- lets a full-pool
    precompute run as two (or more) memory-safe chunks instead of one big load.
    """
    cols = ["traj_id", "source", "seq", "lat", "lon", "t", "dt", "speed_mps", "bearing_deg"]
    path = Path(processed_root) / source / "part-000.parquet"
    if limit_trajs is None and skip_trajs == 0:
        return pd.read_parquet(path, columns=cols).reset_index(drop=True)
    import pyarrow as pa
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)

    skip_row = 0
    if skip_trajs:
        tid = pq.read_table(path, columns=["traj_id"]).column("traj_id").to_numpy(zero_copy_only=False)
        change = np.flatnonzero(tid[1:] != tid[:-1]) + 1
        skip_row = int(change[skip_trajs - 1]) if skip_trajs - 1 < len(change) else len(tid)
        del tid

    # Stream row-group batches and STOP once limit_trajs trajs are seen past skip_row --
    # reads only the target chunk, never the full 68M-row column (slow AND memory-hungry).
    batches, n_trajs, prev_last, row = [], 0, None, 0
    for b in pf.iter_batches(columns=cols, batch_size=1_000_000):
        blen = b.num_rows
        if row + blen <= skip_row:
            row += blen
            continue  # whole batch inside the skipped prefix -- discard, never retain it
        if row < skip_row:
            b = b.slice(skip_row - row)  # batch straddles the cutoff -- keep the tail only
        row += blen
        tb = b.column("traj_id").to_numpy(zero_copy_only=False)
        if len(tb) == 0:
            continue
        internal = int(np.count_nonzero(tb[1:] != tb[:-1]))          # traj boundaries within batch
        boundary = 0 if (prev_last is not None and tb[0] == prev_last) else 1  # boundary vs prev batch
        n_trajs += internal + boundary
        prev_last = tb[-1]
        batches.append(b)
        if limit_trajs is not None and n_trajs >= limit_trajs:
            break
    df = pa.Table.from_batches(batches).to_pandas()
    if limit_trajs is None:
        return df.reset_index(drop=True)
    tid = df["traj_id"].to_numpy()
    # contiguous -> the limit_trajs-th traj's start row IS the cutoff. Avoids np.isin,
    # which has no fast path for string/object dtype traj_id and was measured at 7+
    # minutes for a single 1M-row batch (minutes-to-hours at real limit_trajs scale).
    change = np.flatnonzero(tid[1:] != tid[:-1]) + 1
    cutoff = int(change[limit_trajs - 1]) if limit_trajs - 1 < len(change) else len(tid)
    return df.iloc[:cutoff].reset_index(drop=True)


def _contiguous_groups(traj_id: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end) row ranges for each contiguous run of equal traj_id."""
    if len(traj_id) == 0:
        return []
    change = np.flatnonzero(traj_id[1:] != traj_id[:-1]) + 1
    bounds = np.concatenate(([0], change, [len(traj_id)]))
    return list(zip(bounds[:-1], bounds[1:]))


class TrajectoryGraphDataset(Dataset):
    def __init__(self, df: pd.DataFrame, indices: dict, seq_cfg: SequenceConfig | None = None,
                 retr_cfg: RetrievalConfig | None = None, cache_path: str | Path | None = None,
                 perm_seed: int | None = None):
        """`indices` maps source name -> CandidateIndex for that source's city.

        Candidates are retrieved once for the whole df (vectorized bulk STRtree
        query per source, see `_precompute_candidates`) instead of per fix inside
        `__getitem__` -- the old per-row shapely loop re-ran on every epoch.
        `cache_path`, if given, persists the precomputed arrays to an .npz so a
        re-launched run skips retrieval entirely.

        `perm_seed`: stage2_correction.md Change 7A. The cache is built
        distance-sorted (slot 0 == nearest candidate for every fix), so any
        downstream `argmin(d_perp)` target was ALWAYS slot 0 -- a constant,
        trivially learnable/leaky target. `__getitem__` now permutes the K axis
        independently per fix (fresh draw every call -> free augmentation over
        epochs); downstream `argmin(cand_d_perp_m)` recomputes the target
        post-permutation automatically, no separate pos_idx field needed.
        `perm_seed=None` (default, training) draws from global numpy state.
        `perm_seed=<int>` uses a dedicated Generator -- reproducible eval runs
        (shuffle=False, single-process) get the same permutation sequence.
        """
        self.df = df.reset_index(drop=True)
        self.indices = indices
        self.seq = seq_cfg or SequenceConfig()
        self.retr = retr_cfg or RetrievalConfig()
        groups = _contiguous_groups(self.df["traj_id"].to_numpy())
        self.groups = [(s, e) for (s, e) in groups if (e - s) >= self.seq.min_len]
        self._cand = self._load_or_build_cache(cache_path)
        self._perm_rng = np.random.default_rng(perm_seed) if perm_seed is not None else None

    _CACHE_FIELDS = ("segment_id", "d_perp_m", "heading_deg", "speed_kph", "oneway",
                      "highway_id", "mask")

    def _load_or_build_cache(self, cache_path: str | Path | None) -> dict:
        n = len(self.df)
        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                z = np.load(cache_path)
                # >= not ==: load_source_df reads part-000.parquet strictly top-down
                # with no reordering, so a df of n fixes is an exact row-prefix of any
                # larger df from the same parquet -- cache row i maps to df row i for
                # all i < n, and __getitem__ only ever indexes rows [0, n). The full-pool
                # precompute cache (n_rows 82,351,535, chunked skip_trajs load) is 89 rows
                # longer than a single limit_trajs=1658437 prefix (82,351,446); those tail
                # rows are never touched, so reuse it instead of rebuilding. == forced a
                # MISS -> _precompute_candidates' np.full of all 7 (n,k) arrays (~33GB) ->
                # OOM-kill on the 29GB Kaggle box. ponytail: that np.full is the remaining
                # OOM ceiling if the cache is ever SMALLER than n (>= is False -> rebuild);
                # memmap the precompute output too if a run ever needs more trajs than any cache.
                if int(z["n_rows"]) >= n and int(z["k"]) == self.retr.k:
                    return self._memmap_cache(z, n)
        cand = self._precompute_candidates()
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, n_rows=n, k=self.retr.k, **cand)
        return cand

    def _memmap_cache(self, z, n: int) -> dict:
        # Materializing all 7 dense (n_fixes, k) arrays at once is ~33GB at the full
        # Porto pool (n_fixes=82.4M) and OOM-kills the ~29GB Kaggle box (SIGKILL 9 at
        # cache load). Decompress each field to a scratch .npy once (transient peak =
        # one int64 field, ~6.6GB) and reopen it mmap_mode='r', so resident RAM stays
        # ~0 and __getitem__'s arr[s:e][rows, order] pages in only the rows it slices.
        # Dtypes/values are byte-identical to the eager path -- no behavior change.
        # ponytail: scratch defaults to /kaggle/temp (~73GB disk, not the 20GB
        # output-capped /kaggle/working); override via CAND_MMAP_DIR for local runs.
        scratch = Path(os.environ.get("CAND_MMAP_DIR", "/kaggle/temp")) / f"candmm_n{n}_k{self.retr.k}"
        scratch.mkdir(parents=True, exist_ok=True)
        out = {}
        for f in self._CACHE_FIELDS:
            npy = scratch / f"{f}.npy"
            if not npy.exists():
                tmp = npy.with_name(npy.name + ".tmp")
                with open(tmp, "wb") as fh:
                    np.save(fh, z[f])       # z[f] materializes ONE field transiently
                tmp.replace(npy)            # atomic: a half-written file never looks done
            out[f] = np.load(npy, mmap_mode="r")
        return out

    # Bound query_batch's transient memory (STRtree match set + `ranked` DataFrame,
    # both O(fixes) pre-truncation) to a fixed block instead of full-n. The old
    # single bulk call held ~20GB at 400k trajs (~19M fixes) and OOM-killed the
    # Kaggle box; the resident `cand` arrays (~8GB) stay, transients cap at ~1 chunk.
    _PRECOMP_CHUNK = 1_000_000

    def _precompute_candidates(self) -> dict:
        """Bulk STRtree query per source, chunked over rows to cap peak RAM."""
        n, k = len(self.df), self.retr.k
        fields = (("segment_id", self.seq.pad_seg_id, np.int64),
                  ("d_perp_m", 0.0, np.float32), ("heading_deg", 0.0, np.float32),
                  ("speed_kph", 0.0, np.float32), ("oneway", 0, np.int64),
                  ("highway_id", 0, np.int64), ("mask", 0.0, np.float32))
        cand = {name: np.full((n, k), fill, dtype=dt) for name, fill, dt in fields}
        lat_all = self.df["lat"].to_numpy()
        lon_all = self.df["lon"].to_numpy()
        for source, idx in self.indices.items():
            rows = np.flatnonzero((self.df["source"] == source).to_numpy())
            nchunks = (rows.size + self._PRECOMP_CHUNK - 1) // self._PRECOMP_CHUNK
            # heartbeat: precompute is a silent ~20-40min prelude to training at large
            # limit_trajs -- log per chunk so a live run is distinguishable from a hang/OOM.
            print(f"[precompute] {source}: {rows.size} fixes, k={k}, {nchunks} chunk(s)", flush=True)
            for ci, b0 in enumerate(range(0, rows.size, self._PRECOMP_CHUNK)):
                blk = rows[b0:b0 + self._PRECOMP_CHUNK]          # global row indices for this chunk
                q = idx.query_batch(lat_all[blk], lon_all[blk], self.retr.radius_m, k)
                seg = q["segment_id"].copy()
                seg[q["mask"] == 0] = self.seq.pad_seg_id  # query_batch pads with -1; honour configured pad id
                cand["segment_id"][blk] = seg
                for name, _, _ in fields[1:]:
                    cand[name][blk] = q[name]
                print(f"[precompute] {source} chunk {ci + 1}/{nchunks} done", flush=True)
        return cand

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, i: int) -> dict:
        s, e = self.groups[i]
        e = min(e, s + self.seq.max_len)            # truncate long trajectories
        g = self.df.iloc[s:e]
        L = len(g)
        t = g["t"].to_numpy(np.int64)

        # Change 7A: per-fix independent permutation of the K candidate slots,
        # jointly across every cand_* field -- see __init__ docstring.
        rng = self._perm_rng if self._perm_rng is not None else np.random
        K = self._cand["segment_id"].shape[1]
        order = rng.random((L, K)).argsort(axis=1)
        rows = np.arange(L)[:, None]
        cand = {name: arr[s:e][rows, order] for name, arr in self._cand.items()}

        return {
            "lat": torch.tensor(g["lat"].to_numpy(), dtype=torch.float32),
            "lon": torch.tensor(g["lon"].to_numpy(), dtype=torch.float32),
            "dt": torch.tensor(g["dt"].to_numpy(), dtype=torch.float32),
            "speed_mps": torch.tensor(g["speed_mps"].to_numpy(), dtype=torch.float32),
            "bearing_deg": torch.tensor(g["bearing_deg"].to_numpy(), dtype=torch.float32),
            "hour": torch.tensor(hour_of_day(t), dtype=torch.long),
            "dow": torch.tensor(day_of_week(t), dtype=torch.long),
            "cand_segment_id": torch.from_numpy(np.ascontiguousarray(cand["segment_id"])),
            "cand_d_perp_m": torch.from_numpy(np.ascontiguousarray(cand["d_perp_m"])),
            "cand_heading_deg": torch.from_numpy(np.ascontiguousarray(cand["heading_deg"])),
            "cand_speed_kph": torch.from_numpy(np.ascontiguousarray(cand["speed_kph"])),
            "cand_oneway": torch.from_numpy(np.ascontiguousarray(cand["oneway"])),
            "cand_highway_id": torch.from_numpy(np.ascontiguousarray(cand["highway_id"])),
            "cand_mask": torch.from_numpy(np.ascontiguousarray(cand["mask"])),
            "length": L,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Right-pad variable-length trajectories to the batch max; add seq_mask."""
    B = len(batch)
    Lmax = max(b["length"] for b in batch)
    K = batch[0]["cand_mask"].shape[1]
    seq_keys_1d = ["lat", "lon", "dt", "speed_mps", "bearing_deg", "hour", "dow"]
    cand_keys = ["cand_segment_id", "cand_d_perp_m", "cand_heading_deg",
                 "cand_speed_kph", "cand_oneway", "cand_highway_id", "cand_mask"]

    out = {"seq_mask": torch.zeros(B, Lmax, dtype=torch.float32),
           "length": torch.tensor([b["length"] for b in batch], dtype=torch.long)}
    for k in seq_keys_1d:
        out[k] = torch.zeros(B, Lmax, dtype=batch[0][k].dtype)
    for k in cand_keys:
        out[k] = torch.zeros(B, Lmax, K, dtype=batch[0][k].dtype)

    for b, item in enumerate(batch):
        L = item["length"]
        out["seq_mask"][b, :L] = 1.0
        for k in seq_keys_1d:
            out[k][b, :L] = item[k]
        for k in cand_keys:
            out[k][b, :L] = item[k]
    return out


if __name__ == "__main__":
    # self-check: memmap round-trip returns values byte-identical to the eager npz
    # load, as numpy.memmap (~0 resident), and the __getitem__ fancy-slice matches.
    import tempfile
    import types

    tmpd = tempfile.mkdtemp()
    os.environ["CAND_MMAP_DIR"] = tmpd
    rng = np.random.default_rng(0)
    n, k = 37, 10
    int_fields = ("segment_id", "oneway", "highway_id")
    ref = {f: (rng.integers(0, 99, (n, k)).astype(np.int64) if f in int_fields
               else rng.random((n, k)).astype(np.float32))
           for f in TrajectoryGraphDataset._CACHE_FIELDS}
    npz = Path(tmpd) / "c.npz"
    np.savez_compressed(npz, n_rows=n, k=k, **ref)

    stub = types.SimpleNamespace(retr=types.SimpleNamespace(k=k),
                                 _CACHE_FIELDS=TrajectoryGraphDataset._CACHE_FIELDS)
    out = TrajectoryGraphDataset._memmap_cache(stub, np.load(npz), n)
    r = np.arange(6)[:, None]
    order = np.zeros((6, k), int)
    for f in TrajectoryGraphDataset._CACHE_FIELDS:
        assert isinstance(out[f], np.memmap), f"{f} not memmap"
        assert out[f].dtype == ref[f].dtype, f"{f} dtype drift"
        assert np.array_equal(np.asarray(out[f]), ref[f]), f"{f} value mismatch"
        assert np.array_equal(out[f][3:9][r, order], ref[f][3:9][r, order]), f"{f} slice mismatch"
    print("[selfcheck] memmap round-trip OK")

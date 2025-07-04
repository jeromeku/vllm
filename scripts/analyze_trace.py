#!/usr/bin/env python3
# analyze_trace.py
#
# Reconstruct ancestor chains in a Chrome/Perfetto JSON trace, then
# - find every GPU slice whose name contains **“RadixSort”**
# - link it to the matching **cudaLaunchKernel** host slice via the shared
#   correlation id
# - emit a JSON file whose rows look like
#   {
#     "radix_kernel":      { "name": "...", "ts": 123, "dur": 42 },
#     "cudaLaunchKernel":  { "name": "...", "ts": 120, "dur": 55 },
#     "ancestors":         [ { "name": "...", ... }, … ]   # inner → outer
#   }
#
# Works on traces that contain only complete slices (`"ph":"X"`).
# No Perfetto tools required—pure Pandas / JSON.
#
# ---------------------------------------------------------------------------

import json
import sys
from pathlib import Path
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_events(path: Path):
    """Return the list[dict] of events, regardless of wrapper format."""
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "traceEvents" in data:
        return data["traceEvents"]
    if isinstance(data, list):
        return data
    raise ValueError("File is not a valid Chrome-trace JSON.")


def flatten_args(df: pd.DataFrame) -> pd.DataFrame:
    """Move nested args.X fields to top-level args.X columns."""
    if "args" in df.columns:
        args = pd.json_normalize(df["args"]).add_prefix("args.")
        df = pd.concat([df.drop(columns=["args"]), args], axis=1)
    return df


def detect_corr_col(df: pd.DataFrame):
    """Return the first present correlation column or None."""
    for c in ("args.correlation", "args.correlation_id", "args.correlationId"):
        if c in df.columns:
            return c
    return None


def slim(row: pd.Series):
    """Tiny dict for human-readable / compact JSON."""
    return {
        "name": row.get("name"),
        "ts":   float(row.get("ts", 0)),
        "dur":  float(row.get("dur", 0)),
    }


def build_ancestor_map(df: pd.DataFrame):
    """
    Return {event_id: [Series …]} where the list contains ancestors
    (inner-to-outer) on the same (pid, tid) track, based on interval containment.
    """
    anc = {}
    for (pid, tid), track in df.sort_values(["pid", "tid", "ts"]).groupby(["pid", "tid"]):
        stack = []                         # [(end_time, row), …]  inner-most last
        for _, row in track.iterrows():
            ts = row["ts"]
            # discard finished intervals
            while stack and stack[-1][0] <= ts:
                stack.pop()
            anc[row["_evt_id"]] = [r[1] for r in stack]     # current ancestors
            stack.append((row["ts_end"], row))
    return anc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage:  python analyze_trace.py trace.json  [out.json]")

    src   = Path(sys.argv[1])
    dst   = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("radix_ancestors.json")

    # 1. load & tidy ---------------------------------------------------------
    df = flatten_args(pd.DataFrame(load_events(src)))
    if df.empty:
        sys.exit("Trace contains no events.")

    if "ts" not in df.columns:
        sys.exit("Trace is missing mandatory 'ts' field.")

    df["_evt_id"] = df.index
    df["dur"]     = df["dur"].fillna(0)
    df["ts_end"]  = df["ts"] + df["dur"]

    corr_col = detect_corr_col(df)

    # 2. identify GPU kernels & host launches -------------------------------
    gpu_mask  = df["name"].str.contains("RadixSort", case=False, na=False) & (
                  df.get("cat", "").str.contains("kernel", case=False, na=False, regex=False) |
                  (df.get("pid") == 0)                                # many traces use pid==0 for GPU
               )
    gpu_df    = df[gpu_mask]

    host_df   = df[(df["name"] == "cudaLaunchKernel")]
    if corr_col:
        host_df = host_df[host_df[corr_col].notna()]

    # bail early if nothing to process
    if gpu_df.empty:
        sys.exit("No RadixSort kernels found in trace.")

    # 3. ancestor map for every slice ---------------------------------------
    anc_map = build_ancestor_map(df)

    # 4. build records  ------------------------------------------------------
    records = []
    if corr_col:
        # link via correlation id
        host_by_corr = host_df.set_index(corr_col)
        for _, g in gpu_df.iterrows():
            corr = g[corr_col]
            if corr not in host_by_corr.index:
                continue
            host_row = host_by_corr.loc[corr]
            # if several launches share id (rare), pick first
            if isinstance(host_row, pd.DataFrame):
                host_row = host_row.iloc[0]

            ancestors = [host_row, *anc_map[host_row["_evt_id"]]]
            records.append({
                "radix_kernel":     slim(g),
                "cudaLaunchKernel": slim(host_row),
                "ancestors":        [slim(a) for a in ancestors],   # inner → outer
            })
    else:
        # no correlation column -> just dump ancestors of each kernel itself
        for _, g in gpu_df.iterrows():
            records.append({
                "radix_kernel": slim(g),
                "ancestors":    [slim(a) for a in anc_map[g["_evt_id"]]],
            })

    # 5. write out ----------------------------------------------------------
    pd.DataFrame(records).to_json(dst, orient="records", indent=2)
    print(f"✅  wrote {dst}  ({len(records)} kernels processed)")


if __name__ == "__main__":
    main()

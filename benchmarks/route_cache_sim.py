import argparse
import json
import math
from collections import Counter
from pathlib import Path

EXPERT_BYTES = 6 * 2**20
LAYERS = 40
H2D_GIB_PER_S = 28.6


def load_sessions(paths):
    """Load route events in execution order: preview round, then layer."""
    sessions = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        layers = {int(k): v for k, v in data["by_layer"].items()}
        max_rounds = max((len(v.get("preview_events", [])) for v in layers.values()), default=0)
        events = []
        for ridx in range(max_rounds):
            for li in sorted(layers):
                previews = layers[li].get("preview_events", [])
                if ridx >= len(previews):
                    continue
                ids = tuple(sorted(set(previews[ridx].get("unique_ids", []))))
                events.append({"seq": len(events), "layer": li, "round": ridx, "ids": ids})
        sessions.append({"path": str(path), "events": events, "profile": data})
    return sessions


def _key(layer, expert):
    return (layer, expert)


def _evict(resident, score, limit):
    while len(resident) > limit:
        resident.remove(min(resident, key=score))


def _warm_dynamic(events, capacity, policy, per_layer, tau):
    resident = set()
    last = {}
    freq = Counter()
    now = 0
    cap_by_layer = max(1, capacity // LAYERS) if per_layer else capacity

    def score(k):
        if policy == "lru":
            return (last.get(k, -1), k)
        if policy == "lfu":
            return (freq.get(k, 0), last.get(k, -1), k)
        # Decayed frequency is an approximation of the moex-style recency/frequency policy.
        age = max(0, now - last.get(k, now))
        return (freq.get(k, 0) * math.exp(-age / tau), last.get(k, -1), k)

    for event in events:
        now += 1
        li = event["layer"]
        keys = {_key(li, e) for e in event["ids"]}
        for k in keys:
            resident.add(k)
            last[k] = now
            freq[k] += 1
        if per_layer:
            current = {k for k in resident if k[0] == li}
            _evict(current, score, cap_by_layer)
            resident = {k for k in resident if k[0] != li} | current
        else:
            _evict(resident, score, capacity)
    return resident, last, freq, now


def _static_hot(warm_events, capacity, per_layer):
    counts = Counter(
        _key(event["layer"], expert)
        for event in warm_events
        for expert in event["ids"]
    )
    if per_layer:
        cap_by_layer = max(1, capacity // LAYERS)
        resident = set()
        for li in range(LAYERS):
            layer_counts = [(count, key) for key, count in counts.items() if key[0] == li]
            layer_counts.sort(key=lambda item: (-item[0], item[1]))
            resident.update(key for _, key in layer_counts[:cap_by_layer])
        return resident
    chosen = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:capacity]
    return {key for key, _ in chosen}


def simulate_session(events, capacity, policy, per_layer=False, warm_fraction=0.5, tau=10.0):
    split = max(1, int(len(events) * warm_fraction)) if events else 0
    warm = events[:split]
    test = events[split:]

    if policy == "static_hot":
        resident = _static_hot(warm, capacity, per_layer)
        last = {}
        freq = Counter()
        now = 0
    else:
        resident, last, freq, now = _warm_dynamic(warm, capacity, policy, per_layer, tau)

    hits = misses = active = h2d_bytes = 0
    unique_misses = set()
    per_round = []
    cap_by_layer = max(1, capacity // LAYERS) if per_layer else capacity

    for event in test:
        now += 1
        li = event["layer"]
        keys = {_key(li, expert) for expert in event["ids"]}
        h = sum(k in resident for k in keys)
        m = len(keys) - h
        hits += h
        misses += m
        active += len(keys)
        h2d_bytes += m * EXPERT_BYTES
        unique_misses.update(k for k in keys if k not in resident)

        if policy != "static_hot":
            for k in keys:
                resident.add(k)
                last[k] = now
                freq[k] += 1

            def score(k):
                if policy == "lru":
                    return (last.get(k, -1), k)
                if policy == "lfu":
                    return (freq.get(k, 0), last.get(k, -1), k)
                age = max(0, now - last.get(k, now))
                return (freq.get(k, 0) * math.exp(-age / tau), last.get(k, -1), k)

            if per_layer:
                current = {k for k in resident if k[0] == li}
                _evict(current, score, cap_by_layer)
                resident = {k for k in resident if k[0] != li} | current
            else:
                _evict(resident, score, capacity)

        per_round.append({
            "layer": li,
            "round": event["round"],
            "active": len(keys),
            "hits": h,
            "misses": m,
        })

    h2d_gib = h2d_bytes / 2**30
    return {
        "policy": policy,
        "per_layer": per_layer,
        "capacity_slots": capacity,
        "capacity_gib": capacity * EXPERT_BYTES / 2**30,
        "warm_events": len(warm),
        "test_events": len(test),
        "active_expert_requests": active,
        "hits": hits,
        "misses": misses,
        "hit_rate": hits / active if active else None,
        "h2d_gib": h2d_gib,
        "estimated_h2d_ms_serial_lower_bound": h2d_gib / H2D_GIB_PER_S * 1000,
        "unique_missed_expert_instances": len(unique_misses),
        "final_resident": len(resident),
        "round_stats": per_round,
    }


def aggregate(rows):
    if not rows:
        return {}
    total_active = sum(row["active_expert_requests"] for row in rows)
    total_hits = sum(row["hits"] for row in rows)
    total_misses = sum(row["misses"] for row in rows)
    total_h2d = sum(row["h2d_gib"] for row in rows)
    return {
        "policy": rows[0]["policy"],
        "per_layer": rows[0]["per_layer"],
        "capacity_slots": rows[0]["capacity_slots"],
        "capacity_gib": rows[0]["capacity_gib"],
        "sessions": len(rows),
        "active_expert_requests": total_active,
        "hits": total_hits,
        "misses": total_misses,
        "hit_rate": total_hits / total_active if total_active else None,
        "h2d_gib": total_h2d,
        "estimated_h2d_ms_serial_lower_bound": total_h2d / H2D_GIB_PER_S * 1000,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profiles", nargs="+")
    ap.add_argument("--output", default="benchmarks/results/route-cache-sim.json")
    ap.add_argument("--warm-fraction", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=10.0)
    args = ap.parse_args()

    sessions = load_sessions(args.profiles)
    rows = []
    session_rows = []
    for gib in (3, 4, 5):
        slots = int(gib * 2**30 // EXPERT_BYTES)
        for policy in ("lru", "lfu", "decayed", "static_hot"):
            for per_layer in (False, True):
                current = []
                for session in sessions:
                    result = simulate_session(
                        session["events"], slots, policy, per_layer=per_layer,
                        warm_fraction=args.warm_fraction, tau=args.tau,
                    )
                    result["profile"] = session["path"]
                    current.append(result)
                    session_rows.append(result)
                rows.append(aggregate(current))

    out = {
        "expert_bytes": EXPERT_BYTES,
        "layers": LAYERS,
        "h2d_gib_per_s": H2D_GIB_PER_S,
        "event_order": "preview_round_then_layer",
        "warm_fraction": args.warm_fraction,
        "decay_tau_events": args.tau,
        "profiles": [session["path"] for session in sessions],
        "sessions": [
            {"profile": session["path"], "events": len(session["events"])}
            for session in sessions
        ],
        "note": (
            "Events contain unique expert IDs per preview call only. Token counts, gate mass, "
            "confidence, compute time, and CPU fallback time are unavailable; hit rates are directional."
        ),
        "rows": rows,
        "per_profile_rows": session_rows,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("event_order", "sessions", "rows")}, indent=2))


if __name__ == "__main__":
    main()

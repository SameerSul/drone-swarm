"""End-to-end demo: trained swarm explores, finds the target, guide drone routes
and reroutes live, then a 30-seed benchmark against the baselines."""
import json
import os
from statistics import median

from env import SwarmSearchEnv
from plan import Replanner
from policies import FrontierPolicy, LawnmowerPolicy, QPolicy, RandomPolicy, run_episode

BASE = os.path.dirname(os.path.abspath(__file__))
POLICY_PATH = os.path.join(BASE, "policy.json")
RESULTS_PATH = os.path.join(BASE, "results.json")
BENCH_SEEDS = list(range(3000, 3030))


def ensure_policy():
    if not os.path.exists(POLICY_PATH):
        print("policy.json missing, training first (about 40s)\n")
        import train
        train.main()
        print()


def run_showcase(seed, verbose=False):
    env = SwarmSearchEnv()
    pol = QPolicy()
    obs = env.reset(seed)
    pol.reset(env)

    def show(label, overlay=None, guide=None):
        if verbose:
            print("== %s ==" % label)
            print(env.render(overlay=overlay, guide=guide))
            print()

    show("start (seed %d): drones dropped at corner" % seed)
    mid_shown = False
    done = False
    info = {}
    while not done:
        obs, r, done, info = env.step(pol.act(obs, env))
        if not mid_shown and info["coverage"] >= 0.30:
            show("mid-exploration: step %d, coverage %.2f"
                 % (info["steps"], info["coverage"]))
            mid_shown = True
    if not info["target_found"]:
        show("budget expired, target not found")
        return 0
    show("target found: step %d, coverage %.2f, hazard hits %d"
         % (info["steps"], info["coverage"], info["hazard_hits"]))

    rp = Replanner(env.fused, env.w, env.h, (0, 0), env.target)
    show("guide route planned from corner to target (A* on fused cost map)",
         overlay=rp.path, guide=rp.pos)
    reroute_shown = False
    guard = 0
    while not rp.done() and guard < 4 * (env.w + env.h):
        guard += 1
        env.sense_at(*rp.pos)
        pos, rer = rp.tick(env.fused)
        if rer and not reroute_shown:
            show("hazard revealed on route at guide step %d, rerouted" % guard,
                 overlay=rp.path[max(rp.i - 1, 0):], guide=pos)
            reroute_shown = True
    show("guide arrived at target: %d guide steps, %d reroutes"
         % (guard, rp.reroutes), guide=rp.pos)
    return rp.reroutes


def pick_showcase_seed():
    for seed in BENCH_SEEDS:
        if run_showcase(seed, verbose=False) >= 1:
            return seed
    return BENCH_SEEDS[0]


def benchmark():
    env = SwarmSearchEnv()
    policies = [RandomPolicy(seed=1), LawnmowerPolicy(), FrontierPolicy(), QPolicy()]
    results = {}
    for pol in policies:
        rows = [run_episode(env, pol, s) for s in BENCH_SEEDS]
        results[pol.name] = {
            "median_steps_to_find": median(r["steps"] for r in rows),
            "mean_coverage_at_find": sum(r["coverage"] for r in rows) / len(rows),
            "mean_hazard_hits": sum(r["hazard_hits"] for r in rows) / len(rows),
            "find_rate": sum(r["target_found"] for r in rows) / len(rows),
            "episodes": rows,
        }
    return results


def print_table(results):
    print("benchmark over %d held-out seeds, %dx%d grid, budget %d steps"
          % (len(BENCH_SEEDS), 32, 32, SwarmSearchEnv().budget))
    print("(unfound episodes count as full budget in median steps)\n")
    hdr = "%-10s %16s %18s %14s %10s" % (
        "policy", "med steps->find", "coverage at find", "hazard hits", "find rate")
    print(hdr)
    print("-" * len(hdr))
    for name in ("random", "lawnmower", "frontier", "rl"):
        r = results[name]
        print("%-10s %16d %18.2f %14.1f %10.2f" % (
            name, r["median_steps_to_find"], r["mean_coverage_at_find"],
            r["mean_hazard_hits"], r["find_rate"]))


def main():
    ensure_policy()
    seed = pick_showcase_seed()
    run_showcase(seed, verbose=True)
    results = benchmark()
    print_table(results)
    slim = {k: {kk: vv for kk, vv in v.items() if kk != "episodes"}
            for k, v in results.items()}
    with open(RESULTS_PATH, "w") as f:
        json.dump({"seeds": BENCH_SEEDS, "summary": slim,
                   "episodes": {k: v["episodes"] for k, v in results.items()}}, f, indent=1)
    print("\nwrote %s" % RESULTS_PATH)


if __name__ == "__main__":
    main()

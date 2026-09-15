"""Tabular Q-learning over per-drone features: shared Q-table, decentralized execution."""
import json
import os
import random
import time
from statistics import median

from env import SwarmSearchEnv
from policies import QPolicy, RandomPolicy, key, run_episode

PREF = {1: (1,), 2: (4, 1), 3: (4,), 4: (4, 2), 5: (2,), 6: (3, 2), 7: (3,), 8: (3, 1)}


def _new_row(o):
    row = [0.0] * 5
    for a in PREF.get(o[0], ()):
        if o[2 + a - 1] != 2:
            row[a] = 2.0
    return row

BASE = os.path.dirname(os.path.abspath(__file__))
POLICY_PATH = os.path.join(BASE, "policy.json")


def train(episodes=4000, time_limit=50.0, sizes=((24, 24), (32, 32)), n=3,
          alpha=0.15, gamma=0.9, seed0=0, verbose=True, env_kwargs=None):
    envs = [SwarmSearchEnv(w, h, n, **(env_kwargs or {})) for (w, h) in sizes]
    rng = random.Random(7)
    Q = {}
    t0 = time.time()
    hist = []
    for ep in range(episodes):
        env = envs[ep % len(envs)]
        eps = max(0.05, 0.998 ** ep)
        obs = env.reset(seed=seed0 + ep)
        done = False
        total = 0.0
        while not done:
            ks = [key(o) for o in obs]
            acts = []
            for i in range(n):
                if rng.random() < eps:
                    acts.append(rng.randrange(5))
                else:
                    row = Q.get(ks[i])
                    acts.append(max(range(5), key=lambda a: row[a]) if row else rng.randrange(5))
            obs2, r, done, info = env.step(acts)
            total += r
            dr = info["drone_rewards"]
            ks2 = [key(o) for o in obs2]
            for i in range(n):
                row = Q.get(ks[i])
                if row is None:
                    row = Q[ks[i]] = _new_row(obs[i])
                nrow = Q.get(ks2[i])
                boot = 0.0 if done else gamma * (max(nrow) if nrow else 2.0)
                row[acts[i]] += alpha * (dr[i] + boot - row[acts[i]])
            obs = obs2
        hist.append((total, info["coverage"], info["target_found"]))
        if verbose and (ep + 1) % 200 == 0:
            rec = hist[-200:]
            print("ep %5d  eps %.2f  avg_reward %7.1f  avg_coverage %.2f  "
                  "find_rate %.2f  states %d" % (
                      ep + 1, eps, sum(x[0] for x in rec) / len(rec),
                      sum(x[1] for x in rec) / len(rec),
                      sum(x[2] for x in rec) / len(rec), len(Q)))
        if time.time() - t0 > time_limit:
            if verbose:
                print("time limit reached at episode %d" % (ep + 1))
            break
    return Q, time.time() - t0


def main():
    Q, dt = train()
    with open(POLICY_PATH, "w") as f:
        json.dump(Q, f)
    print("trained %d states in %.1fs, saved %s" % (len(Q), dt, POLICY_PATH))
    env = SwarmSearchEnv(24, 24, 3)
    rl, rnd = QPolicy(table=Q), RandomPolicy(seed=1)
    rs, xs = [], []
    for s in range(5000, 5015):
        rs.append(run_episode(env, rl, s)["steps"])
        xs.append(run_episode(env, rnd, s)["steps"])
    print("holdout sanity (24x24, 15 seeds): rl median steps %d, random median steps %d"
          % (median(rs), median(xs)))
    return Q


if __name__ == "__main__":
    main()

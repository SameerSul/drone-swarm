"""Test suite: python3 tests.py runs everything, exits nonzero on failure."""
import os
import random
import sys
from statistics import median

from env import SwarmSearchEnv, UNK, FREE, HAZ, OBS
from plan import Replanner, astar
from policies import QPolicy, RandomPolicy, run_episode

BASE = os.path.dirname(os.path.abspath(__file__))


def test_determinism():
    def rollout():
        env = SwarmSearchEnv(24, 24, 3)
        rng = random.Random(5)
        obs = env.reset(seed=11)
        log = [tuple(env.drones)]
        rewards = []
        for _ in range(120):
            obs, r, done, info = env.step([rng.randrange(5) for _ in range(3)])
            log.append(tuple(env.drones))
            rewards.append(round(r, 9))
            if done:
                break
        return log, rewards, list(env.fused), env.target

    a, b = rollout(), rollout()
    assert a == b, "same seed must give identical trajectories, maps and rewards"


def test_fused_map_only_sensed():
    env = SwarmSearchEnv(16, 16, 3)
    rng = random.Random(2)
    sensed = set()

    def mark():
        for (x, y) in env.drones:
            for dx, dy in env._offsets:
                nx, ny = x + dx, y + dy
                if 0 <= nx < env.w and 0 <= ny < env.h:
                    sensed.add((nx, ny))

    env.reset(seed=7)
    mark()
    for _ in range(80):
        _, _, done, _ = env.step([rng.randrange(5) for _ in range(3)])
        mark()
        if done:
            break
    for y in range(env.h):
        for x in range(env.w):
            if env.fused[y * env.w + x] != UNK:
                assert (x, y) in sensed, \
                    "fused map contains cell (%d,%d) no drone ever sensed" % (x, y)


def test_astar_optimal():
    w = h = 7
    fused = [FREE] * (w * h)
    path = astar(fused, w, h, (0, 0), (6, 4))
    assert path is not None and len(path) - 1 == 10, "free map path must be Manhattan optimal"
    for x in (3,):
        for y in range(0, 6):
            fused[y * w + x] = OBS
    path = astar(fused, w, h, (0, 0), (6, 0))
    assert path is not None and len(path) - 1 == 18, \
        "wall detour must cost exactly 18 moves, got %s" % (len(path) - 1)


def test_replan_avoids_hazard():
    w = h = 9
    fused = [FREE] * (w * h)
    rp = Replanner(fused, w, h, (0, 4), (8, 4))
    assert rp.path is not None and (4, 4) in rp.path
    fused[4 * w + 4] = HAZ  # hazard revealed mid-route
    old = list(rp.path)
    for _ in range(40):
        pos, rer = rp.tick(fused)
        if rp.done():
            break
    assert rp.reroutes >= 1, "replanner must react to the revealed hazard"
    assert (4, 4) not in rp.path, "new route must avoid the hazard cell"
    assert rp.path != old and rp.pos == (8, 4)


def test_rl_beats_random():
    if not os.path.exists(os.path.join(BASE, "policy.json")):
        import train
        train.main()
    env = SwarmSearchEnv(24, 24, 3)
    rl, rnd = QPolicy(), RandomPolicy(seed=3)
    rl_steps, rd_steps, rl_found, rd_found = [], [], 0, 0
    for s in range(9000, 9012):
        a = run_episode(env, rl, s)
        b = run_episode(env, rnd, s)
        rl_steps.append(a["steps"])
        rd_steps.append(b["steps"])
        rl_found += a["target_found"]
        rd_found += b["target_found"]
    assert median(rl_steps) < 0.5 * median(rd_steps), \
        "rl median %s vs random %s" % (median(rl_steps), median(rd_steps))
    assert rl_found > rd_found, "rl must find the target more often than random"


def main():
    tests = [test_determinism, test_fused_map_only_sensed, test_astar_optimal,
             test_replan_avoids_hazard, test_rl_beats_random]
    failed = 0
    for t in tests:
        try:
            t()
            print("PASS %s" % t.__name__)
        except AssertionError as e:
            failed += 1
            print("FAIL %s: %s" % (t.__name__, e))
    if failed:
        sys.exit("%d test(s) failed" % failed)
    print("all %d tests passed" % len(tests))


if __name__ == "__main__":
    main()

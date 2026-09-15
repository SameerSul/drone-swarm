"""Baseline and learned policies. Shared interface: act(obs, env) -> joint action list."""
import json
import os
import random
from collections import deque

from env import ACTIONS, FREE, HAZ, OBS

BASE = os.path.dirname(os.path.abspath(__file__))


def key(o):
    return ",".join(map(str, o))


def bfs_step(env, start, goal, known_only=True):
    """First action of a BFS path on the fused map, 0 if unreachable or arrived.
    known_only=True restricts to sensed free/hazard cells, else any non-obstacle."""
    if start == goal:
        return 0
    w, h = env.w, env.h
    ok = (FREE, HAZ) if known_only else (FREE, HAZ, 0)
    first = {start: 0}
    q = deque([start])
    while q:
        x, y = q.popleft()
        for a in (1, 2, 3, 4):
            dx, dy = ACTIONS[a]
            p = (x + dx, y + dy)
            if 0 <= p[0] < w and 0 <= p[1] < h and p not in first \
                    and env.fused[p[1] * w + p[0]] in ok:
                first[p] = a if (x, y) == start else first[(x, y)]
                if p == goal:
                    return first[p]
                q.append(p)
    gx, gy = goal
    x, y = start
    if abs(gx - x) >= abs(gy - y) and gx != x:
        return 4 if gx > x else 3
    if gy != y:
        return 2 if gy > y else 1
    return 0


class RandomPolicy:
    name = "random"

    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def reset(self, env):
        pass

    def act(self, obs, env):
        return [self.rng.randrange(5) for _ in range(env.n)]


class LawnmowerPolicy:
    """Fixed serpentine sweep, horizontal bands partitioned across drones."""
    name = "lawnmower"

    def reset(self, env):
        n, w, h = env.n, env.w, env.h
        band = h / n
        stride = 2 * env.rad + 1
        self.wps, self.idx, self.tries = [], [0] * n, [0] * n
        for i in range(n):
            y0, y1 = int(i * band), int((i + 1) * band)
            rows = list(range(min(y0 + env.rad, h - 1), y1, stride)) or [y0]
            pts = []
            for k, r in enumerate(rows):
                xs = (env.rad, w - 1 - env.rad) if k % 2 == 0 else (w - 1 - env.rad, env.rad)
                pts += [(xs[0], r), (xs[1], r)]
            self.wps.append(pts)

    def act(self, obs, env):
        acts = []
        for i, (x, y) in enumerate(env.drones):
            pts = self.wps[i]
            while self.idx[i] < len(pts) and (
                    pts[self.idx[i]] == (x, y)
                    or env.fused[pts[self.idx[i]][1] * env.w + pts[self.idx[i]][0]] == OBS):
                self.idx[i] += 1
                self.tries[i] = 0
            if self.idx[i] >= len(pts):
                acts.append(0)
                continue
            self.tries[i] += 1
            if self.tries[i] > 60:
                self.idx[i] += 1
                self.tries[i] = 0
            if self.idx[i] >= len(pts):
                acts.append(0)
                continue
            acts.append(bfs_step(env, (x, y), pts[self.idx[i]], known_only=False))
        return acts


class FrontierPolicy:
    """Each drone greedily assigned to nearest unclaimed frontier, BFS pathing on fused map."""
    name = "frontier"

    def reset(self, env):
        pass

    def act(self, obs, env):
        pairs = []
        for i, (x, y) in enumerate(env.drones):
            for f in env.frontiers:
                pairs.append((abs(f[0] - x) + abs(f[1] - y), i, f))
        pairs.sort()
        assign, used = {}, set()
        for d, i, f in pairs:
            if i not in assign and f not in used:
                assign[i] = f
                used.add(f)
        acts = []
        for i, (x, y) in enumerate(env.drones):
            f = assign.get(i)
            acts.append(bfs_step(env, (x, y), f) if f else 0)
        return acts


class QPolicy:
    """Greedy over a learned shared Q-table, decentralized execution."""
    name = "rl"

    def __init__(self, table=None, path=None):
        if table is None:
            with open(path or os.path.join(BASE, "policy.json")) as f:
                table = json.load(f)
        self.q = table
        self.hist = []
        self.rng = random.Random(0)

    def reset(self, env):
        self.hist = [deque(maxlen=10) for _ in range(env.n)]
        self.rng = random.Random(0)

    INV = {0: 0, 1: 2, 2: 1, 3: 4, 4: 3}

    def act(self, obs, env):
        if len(self.hist) != env.n:
            self.reset(env)
        acts = []
        for i, o in enumerate(obs):
            nb = o[2:6]
            allowed = [a for a in (1, 2, 3, 4) if nb[a - 1] != 2]
            back = self.INV[env.last[i]]
            if back in allowed and len(allowed) > 1:
                allowed.remove(back)
            if not allowed:
                allowed = [0]
            pos = env.drones[i]
            self.hist[i].append(pos)
            if list(self.hist[i]).count(pos) >= 3:
                acts.append(self.rng.choice(allowed))
                continue
            row = self.q.get(key(o))
            if row:
                acts.append(max(allowed, key=lambda a: row[a]))
            else:
                acts.append(self._heur(o, allowed))
        return acts

    def _heur(self, o, allowed):
        pref = {1: [1], 2: [4, 1], 3: [4], 4: [4, 2],
                5: [2], 6: [3, 2], 7: [3], 8: [3, 1]}.get(o[0], [])
        for a in pref + [1, 2, 3, 4]:
            if a in allowed:
                return a
        return allowed[0]


def run_episode(env, policy, seed):
    obs = env.reset(seed)
    policy.reset(env)
    total = 0.0
    info = {"coverage": env.revealed / (env.w * env.h), "steps": 0,
            "hazard_hits": 0, "target_found": env.target_found}
    done = env.target_found and env.terminate_on_find
    while not done:
        obs, r, done, info = env.step(policy.act(obs, env))
        total += r
    info = dict(info)
    info["reward"] = total
    return info

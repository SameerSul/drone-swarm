"""SwarmSearchEnv: multi-drone search RL environment, Gym-style, pure stdlib."""
import math
import random

UNK, FREE, HAZ, OBS = 0, 1, 2, 3
ACTIONS = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]  # stay, N, S, W, E
N_ACTIONS = 5
DIR8 = {(0, -1): 1, (1, -1): 2, (1, 0): 3, (1, 1): 4,
        (0, 1): 5, (-1, 1): 6, (-1, 0): 7, (-1, -1): 8}
CHARS = {UNK: "?", FREE: ".", HAZ: "~", OBS: "#"}


def _sgn(v):
    return (v > 0) - (v < 0)


class SwarmSearchEnv:
    """reset(seed) -> obs, step(joint_action) -> (obs, reward, done, info).

    obs is a list of per-drone feature tuples:
      (frontier_dir 0..8, frontier_dist_class 0..3,
       nbr_N, nbr_S, nbr_W, nbr_E each 0 free/unknown 1 hazard 2 blocked,
       crowd_dir 0 none or 1..4 toward nearest other drone within 3 cells,
       heading 0..4, the drone's last executed move)

    frontier_dir points at the drone's assigned frontier goal. The ground station
    assigns each drone the nearest frontier inside its own angular sector around
    the drop corner (falling back to any unclaimed frontier), sticky until the
    frontier is consumed. Goal assignment uses the fused map only.
    """

    def __init__(self, width=32, height=32, n_drones=3, sense_radius=2,
                 step_budget=None, terminate_on_find=True,
                 r_reveal=0.15, r_target=50.0, r_hazard=-2.0, r_step=-0.1, r_shape=0.4,
                 depth_w=0.0):
        self.w, self.h, self.n = width, height, n_drones
        self.rad = sense_radius
        self.budget = step_budget if step_budget else max(120, (width * height * 2) // 5)
        self.terminate_on_find = terminate_on_find
        self.r_reveal, self.r_target = r_reveal, r_target
        self.r_hazard, self.r_step, self.r_shape = r_hazard, r_step, r_shape
        self.depth_w = depth_w
        self._offsets = [(dx, dy) for dy in range(-sense_radius, sense_radius + 1)
                         for dx in range(-sense_radius, sense_radius + 1)]

    def _blob(self, rng, size):
        cells = [(rng.randrange(self.w), rng.randrange(self.h))]
        seen = set(cells)
        tries = 0
        while len(cells) < size and tries < 60:
            tries += 1
            x, y = cells[rng.randrange(len(cells))]
            dx, dy = ACTIONS[1 + rng.randrange(4)]
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.w and 0 <= ny < self.h and (nx, ny) not in seen:
                seen.add((nx, ny))
                cells.append((nx, ny))
        return cells

    def _gen(self, seed):
        rng = random.Random(seed)
        w, h = self.w, self.h
        g = [FREE] * (w * h)
        for _ in range(w * h // 70):
            for (x, y) in self._blob(rng, rng.randint(3, 8)):
                g[y * w + x] = OBS
        for _ in range(w * h // 55):
            for (x, y) in self._blob(rng, rng.randint(4, 10)):
                if g[y * w + x] != OBS:
                    g[y * w + x] = HAZ
        for y in range(3):
            for x in range(3):
                g[y * w + x] = FREE
        dist = {(0, 0): 0}
        q = [(0, 0)]
        while q:
            nq = []
            for (x, y) in q:
                for dx, dy in ACTIONS[1:]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in dist \
                            and g[ny * w + nx] != OBS:
                        dist[(nx, ny)] = dist[(x, y)] + 1
                        nq.append((nx, ny))
            q = nq
        far = sorted(c for c, d in dist.items()
                     if d >= (w + h) // 4 and g[c[1] * w + c[0]] == FREE)
        if not far:
            dmax = max(dist.values())
            far = sorted(c for c, d in dist.items() if d >= dmax // 2)
        self.target = rng.choice(far)
        self.truth = g

    def reset(self, seed=0):
        self._gen(seed)
        self.fused = [UNK] * (self.w * self.h)
        self.frontiers = set()
        corner = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2)]
        self.drones = list(corner[:self.n])
        self.last = [0] * self.n
        self.ftarget = [None] * self.n
        self.steps = 0
        self.revealed = 0
        self.hazard_hits = 0
        self.target_found = False
        for i in range(self.n):
            self.sense_at(*self.drones[i])
        return self._all_obs()

    def _upd_frontier(self, x, y):
        j = y * self.w + x
        if self.fused[j] in (FREE, HAZ):
            for dx, dy in ACTIONS[1:]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.w and 0 <= ny < self.h \
                        and self.fused[ny * self.w + nx] == UNK:
                    self.frontiers.add((x, y))
                    return
        self.frontiers.discard((x, y))

    def sense_at(self, x0, y0):
        """Reveal ground truth within sense radius, update fused map and frontiers."""
        new = 0
        for dx, dy in self._offsets:
            x, y = x0 + dx, y0 + dy
            if 0 <= x < self.w and 0 <= y < self.h:
                j = y * self.w + x
                if self.fused[j] == UNK:
                    self.fused[j] = self.truth[j]
                    new += 1
                    if (x, y) == self.target:
                        self.target_found = True
                    self._upd_frontier(x, y)
                    for ddx, ddy in ACTIONS[1:]:
                        ax, ay = x + ddx, y + ddy
                        if 0 <= ax < self.w and 0 <= ay < self.h:
                            self._upd_frontier(ax, ay)
        self.revealed += new
        return new

    def step(self, actions):
        dr = [self.r_step] * self.n
        for i, a in enumerate(actions):
            dx, dy = ACTIONS[a]
            g = self.ftarget[i]
            x0, y0 = self.drones[i]
            if dx or dy:
                x, y = self.drones[i]
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.w and 0 <= ny < self.h \
                        and self.truth[ny * self.w + nx] != OBS:
                    self.drones[i] = (nx, ny)
                    self.last[i] = a
                    if self.truth[ny * self.w + nx] == HAZ:
                        dr[i] += self.r_hazard
                        self.hazard_hits += 1
            if g is not None:
                x1, y1 = self.drones[i]
                dr[i] += self.r_shape * ((abs(g[0] - x0) + abs(g[1] - y0))
                                         - (abs(g[0] - x1) + abs(g[1] - y1)))
            before = self.target_found
            new = self.sense_at(*self.drones[i])
            dr[i] += new * self.r_reveal
            if self.target_found and not before:
                dr[i] += self.r_target
        self.steps += 1
        done = (self.target_found and self.terminate_on_find) or self.steps >= self.budget
        info = {"coverage": self.revealed / (self.w * self.h), "steps": self.steps,
                "hazard_hits": self.hazard_hits, "target_found": self.target_found,
                "drone_rewards": dr}
        return self._all_obs(), sum(dr), done, info

    def _all_obs(self):
        return [self._obs_for(i) for i in range(self.n)]

    def _sector(self, x, y):
        ang = math.atan2(y + 0.5, x + 0.5)  # 0..pi/2 measured from drop corner
        return min(self.n - 1, int(ang / (math.pi / 2) * self.n))

    def _frontier_goal(self, i):
        x, y = self.drones[i]
        t = self.ftarget[i]
        if t is not None and t in self.frontiers:
            return t
        claimed = {self.ftarget[j] for j in range(self.n) if j != i}

        def score(f):
            return (abs(f[0] - x) + abs(f[1] - y)) - self.depth_w * (f[0] + f[1])

        best, bd = None, 10 ** 18
        for f in self.frontiers:
            if f in claimed or self._sector(f[0], f[1]) != i:
                continue
            s = score(f)
            if s < bd:
                bd, best = s, f
        if best is None:
            for f in self.frontiers:
                if f in claimed:
                    continue
                s = score(f)
                if s < bd:
                    bd, best = s, f
        if best is None:
            for f in self.frontiers:
                s = score(f)
                if s < bd:
                    bd, best = s, f
        self.ftarget[i] = best
        return best

    def _obs_for(self, i):
        x, y = self.drones[i]
        best = self._frontier_goal(i)
        bd = abs(best[0] - x) + abs(best[1] - y) if best else 0
        if best is None or bd == 0:
            fdir, fdist = 0, 0
        else:
            fdir = DIR8[(_sgn(best[0] - x), _sgn(best[1] - y))]
            fdist = 0 if bd <= 2 else 1 if bd <= 6 else 2 if bd <= 12 else 3
        nb = []
        for dx, dy in ACTIONS[1:]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.w and 0 <= ny < self.h):
                nb.append(2)
            else:
                v = self.fused[ny * self.w + nx]
                nb.append(2 if v == OBS else 1 if v == HAZ else 0)
        crowd, cb, cd = 0, 10 ** 9, None
        for j, (ox, oy) in enumerate(self.drones):
            if j == i:
                continue
            d = max(abs(ox - x), abs(oy - y))
            if d < cb:
                cb, cd = d, (ox - x, oy - y)
        if cd is not None and cb <= 3:
            if abs(cd[0]) >= abs(cd[1]) and cd[0] != 0:
                crowd = 4 if cd[0] > 0 else 3
            elif cd[1] != 0:
                crowd = 2 if cd[1] > 0 else 1
            else:
                crowd = 1
        return (fdir, fdist, nb[0], nb[1], nb[2], nb[3], crowd, self.last[i])

    def render(self, overlay=None, guide=None):
        rows = []
        for y in range(self.h):
            row = []
            for x in range(self.w):
                ch = CHARS[self.fused[y * self.w + x]]
                if self.target_found and (x, y) == self.target:
                    ch = "T"
                row.append(ch)
            rows.append(row)
        if overlay:
            for (x, y) in overlay:
                if rows[y][x] != "T":
                    rows[y][x] = "*"
        if guide:
            rows[guide[1]][guide[0]] = "G"
        for i, (x, y) in enumerate(self.drones):
            rows[y][x] = str(i)
        return "\n".join("".join(r) for r in rows)

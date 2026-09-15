"""Guide-drone path planning on the fused cost map.

A* with replan-on-map-change: the functional stand-in for D*-Lite, which is the
incremental-replan upgrade. Cost = distance + hazard cost + hazard proximity penalty,
unknown cells passable at moderate cost, obstacles impassable.
"""
import heapq

from env import ACTIONS, FREE, HAZ, OBS, UNK

HAZ_COST = 8.0
UNK_COST = 2.0
NEAR_HAZ = 1.5


def cell_cost(fused, w, h, x, y):
    v = fused[y * w + x]
    if v == OBS:
        return None
    c = 1.0
    if v == HAZ:
        c += HAZ_COST
    elif v == UNK:
        c += UNK_COST
    for dx, dy in ACTIONS[1:]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h and fused[ny * w + nx] == HAZ:
            c += NEAR_HAZ
            break
    return c


def astar(fused, w, h, start, goal):
    """Returns path [start, ..., goal] or None."""
    if fused[goal[1] * w + goal[0]] == OBS or fused[start[1] * w + start[0]] == OBS:
        return None
    g = {start: 0.0}
    came = {}
    heap = [(abs(start[0] - goal[0]) + abs(start[1] - goal[1]), 0.0, start)]
    closed = set()
    while heap:
        _, gc, pos = heapq.heappop(heap)
        if pos == goal:
            path = [pos]
            while pos in came:
                pos = came[pos]
                path.append(pos)
            return path[::-1]
        if pos in closed:
            continue
        closed.add(pos)
        x, y = pos
        for dx, dy in ACTIONS[1:]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            c = cell_cost(fused, w, h, nx, ny)
            if c is None:
                continue
            ng = gc + c
            p = (nx, ny)
            if ng < g.get(p, 1e18):
                g[p] = ng
                came[p] = pos
                heapq.heappush(heap, (ng + abs(nx - goal[0]) + abs(ny - goal[1]), ng, p))
    return None


def path_cost(fused, w, h, path):
    return sum(cell_cost(fused, w, h, x, y) or 1e9 for (x, y) in path[1:])


class Replanner:
    """Walks a guide drone along an A* path, replanning when the fused map
    changes for the worse (new hazard or obstacle) on the remaining route."""

    def __init__(self, fused, w, h, start, goal):
        self.w, self.h, self.goal = w, h, goal
        self.pos = start
        self.reroutes = 0
        self._plan(fused)

    def _plan(self, fused):
        self.path = astar(fused, self.w, self.h, self.pos, self.goal)
        self.i = 1
        self.snap = {c: fused[c[1] * self.w + c[0]] for c in (self.path or [])}

    def tick(self, fused):
        """Advance one step. Returns (pos, rerouted_this_tick)."""
        if not self.path or self.pos == self.goal:
            return self.pos, False
        rer = False
        for c in self.path[self.i:]:
            v = fused[c[1] * self.w + c[0]]
            if v != self.snap.get(c) and v in (HAZ, OBS):
                self._plan(fused)
                self.reroutes += 1
                rer = True
                break
        if self.path and self.i < len(self.path):
            self.pos = self.path[self.i]
            self.i += 1
        return self.pos, rer

    def done(self):
        return self.pos == self.goal or not self.path

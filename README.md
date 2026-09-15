# Drone Swarm Search Prototype

We are building a drone fleet that learns how to search unmapped terrain: drones explore, fuse their observations into one shared hazard map, and route a guide drone to the target once found. The prototype is a full software simulation centered on the hard part, a reinforcement learning environment with a Gym-style interface where multi-drone search policies can be trained and scored against coverage, hazard hits, and time to target. A trained policy is benchmarked against lawnmower and frontier baselines, and the guide route replans live as the map changes underneath it.

Pure Python 3 stdlib. No numpy, no gym, no torch, nothing to install.

## Files

- `env.py`: `SwarmSearchEnv`, the RL environment (the centerpiece)
- `policies.py`: random, lawnmower, frontier baselines plus the learned `QPolicy`
- `train.py`: tabular Q-learning, saves `policy.json`
- `plan.py`: A* guide routing with replan-on-map-change
- `demo.py`: end-to-end showcase plus 30-seed benchmark, writes `results.json`
- `tests.py`: determinism, map fusion, planner optimality, replanning, RL-beats-random

## Environment design

`SwarmSearchEnv(width=32, height=32, n_drones=3, sense_radius=2)` with the Gym-style interface `reset(seed) -> obs`, `step(joint_action) -> (obs, reward, done, info)`, and `render()` returning an ASCII frame (`?` unknown, `.` free, `~` hazard, `#` obstacle, `T` target, digits are drones, `G` is the guide, `*` is the planned route).

- Worlds are procedurally generated from the seed: obstacle and hazard blobs grown by seeded random walks, a 3x3 clear drop zone in the corner, and one hidden target cell chosen among reachable free cells at least a quarter perimeter away from the drop corner. Everything downstream is fully deterministic given the seed (verified by `tests.py`).
- Partial observability: each drone senses the 5x5 square around itself. The environment maintains the fused shared map (per cell: unknown, free, hazard, obstacle) exactly as the ground station would, updated the moment any drone senses a cell. Tests assert the fused map never contains a cell no drone sensed.
- Joint action space: per drone, one of stay/N/S/W/E. Moving into an obstacle or off the map wastes the step. Entering a hazard cell applies a penalty and the drone survives, which keeps the learning signal simple.
- Episode ends when the target cell is revealed or the step budget (409 on 32x32) expires. `info` carries coverage fraction, steps, hazard hits, target found flag, and per-drone reward decomposition.

## Feature encoding (the deliberate design decision)

Raw grids are useless to tabular RL, so the observation is a compact per-drone feature tuple computed only from the fused map, exactly the information a ground station could broadcast:

| feature | values | meaning |
|---|---|---|
| frontier_dir | 0..8 | 8-way direction to the drone's assigned frontier goal, 0 if none |
| frontier_dist | 0..3 | Manhattan distance class to that goal (<=2, <=6, <=12, 13+) |
| nbr_N/S/W/E | 0..2 each | neighbor cell: 0 free or unknown, 1 hazard, 2 obstacle or wall |
| crowd_dir | 0..4 | direction to the nearest other drone if within 3 cells, else 0 |
| heading | 0..4 | the drone's last executed move (sweep persistence) |

Goal assignment is part of the observation function: the ground station gives each drone the nearest frontier inside its own angular sector around the drop corner (fallback: nearest unclaimed frontier anywhere), and the assignment is sticky until that frontier is consumed. Sticky sector goals were the two decisive tuning steps: without stickiness the nearest frontier flips every step and greedy policies dither in place, and without sectors all drones crawl one expanding disc and lose badly to stripe sweeps on time-to-find. This abstraction is what makes stdlib tabular RL feasible here, and the same `reset/step` interface accepts richer observations later (`env.fused` is available to any policy). Greedy per-drone assignment is used throughout; Hungarian assignment is the upgrade.

## Reward shaping and why

Per drone, per step (summed into the team reward, exposed per drone in `info["drone_rewards"]` for clean credit assignment):

- +0.15 per newly revealed cell: the core exploration incentive
- +50 for revealing the target: the mission bonus
- -2 for entering a hazard cell: strong enough to teach detours, weak enough not to paralyze
- -0.1 per step: time pressure
- +0.4 * (Manhattan progress toward the assigned frontier goal): potential-style shaping. Without it, coarse features alias states and Q-learning leaves argmax errors that deterministic greedy execution turns into permanent loops; with it, approach gradients dominate everywhere.

Training: shared Q-table across drones (decentralized execution), epsilon-greedy decaying 1.0 -> 0.05, alpha 0.15, gamma 0.9, episodes alternating 24x24 and 32x32 grids, about 4000 episodes in under 40 seconds. New Q rows are initialized with a small prior toward the goal direction, which cut run-to-run variance sharply. Execution is greedy with two guards learned the hard way: never immediately backtrack unless boxed in, and take a seeded random legal move if the same cell repeats 3 times in the last 10 (aliasing escape). Lowering the hazard penalty from -3 to -2 traded a few hazard hits for materially faster finds.

## How to run

```
python3 train.py    # about 40s, writes policy.json, prints learning curve
python3 demo.py     # trains first if policy.json missing, prints ASCII milestones,
                    # 30-seed benchmark table, writes results.json
python3 tests.py    # all 5 tests must pass
```

## Benchmark (30 held-out seeds, 32x32, budget 409, unfound counts as full budget)

| policy | median steps to find | coverage at find | hazard hits | find rate |
|---|---|---|---|---|
| random | 409 | 0.31 | 87.1 | 0.20 |
| lawnmower | 40 | 0.52 | 13.9 | 1.00 |
| frontier | 63 | 0.55 | 22.2 | 1.00 |
| rl | 42 | 0.54 | 13.7 | 1.00 |

The learned policy crushes random, clearly beats frontier search on time-to-find, and matches the lawnmower sweep on median steps (42 vs 40, and it wins or ties 14 of 30 paired seeds) while beating it on coverage at find time and on hazard hits. The lawnmower is a strong baseline on time-to-find precisely because fixed stripes transit deep immediately; the RL policy reaches parity while also being map-adaptive and safer around hazards.

## Guide routing

`plan.py` runs A* over the fused cost map: cost 1 per move, +8 for hazard cells, +1.5 adjacent to hazard, +2 for unknown cells (optimistic but cautious), obstacles impassable. The `Replanner` walks the guide along the route, and whenever a cell on the remaining route is newly revealed as hazard or obstacle it replans from the current position. This is the functional stand-in for D*-Lite, which is the incremental-replan upgrade; on these map sizes full A* replans are microseconds. The demo shows a hazard revealed mid-route and the path visibly rerouting.

## What maps to real hardware later

- The ground station is literally this process: fused map, frontier goal assignment, and guide routing all already consume only sensed data, so they port directly behind a radio link with the same interfaces.
- Drone-side: the per-drone feature tuple is a few bytes, so the downlink is observations and the uplink is goals plus actions, which fits a lossy low-bandwidth link.
- Policy upgrades ride the same `reset/step` env interface: PPO or DQN over the raw local map patch, attention over teammate states, learned assignment instead of the sector heuristic.
- Sensing becomes noisy and probabilistic (occupancy grids), and the fused map gains confidence values.

## Current limitations

- Sector goal assignment assumes all drones start at one corner; arbitrary drop patterns need a generalized partition (or learned assignment).
- Hazards are only penalized, never lethal, and sensing is noise-free line-of-sight-free.
- Greedy frontier claiming, not Hungarian; A* replan-from-scratch, not D*-Lite.
- Tabular Q with coarse features plateaus near the sweep baselines; beating them decisively needs function approximation over richer observations, which is exactly the follow-on work the capstone proposes.

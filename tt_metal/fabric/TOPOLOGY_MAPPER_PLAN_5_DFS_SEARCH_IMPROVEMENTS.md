# Plan 5 — Search improvements to the adjacency-guided DFS

**Fix the search, not the architecture.** Four changes to
`PhysicalGroupingDescriptor::solve_adjacency_guided_placement` that address the faults the Gemma SC36
trace actually shows, none of which require precomputed candidate lists.

**Status: §2 implemented, §3–§5 proposed.** Rigidity-first ordering is in the tree and compiles;
everything else is unbuilt.

**Priority: 1.** This is the cheap path. [Plan 4](TOPOLOGY_MAPPER_PLAN_4_SAT_JOINT_PLACEMENT.md) is the
expensive one, and §2 alone may make it unnecessary.

Sibling plans: [Plan 3 — connectivity-aware PGD placement](TOPOLOGY_MAPPER_PLAN_3_CONNECTIVITY_AWARE_PGD_PLACEMENT.md)
(the search being improved), [Plan 4 — two-layer SAT joint placement](TOPOLOGY_MAPPER_PLAN_4_SAT_JOINT_PLACEMENT.md).

> **Relationship to Plan 4.** Plan 4 hoists the inner CSP solve out of the search so that true MRV and
> real forward checking become affordable. This plan gets a large fraction of the same benefit from
> static approximations that cost nothing, and is strictly compatible: every change here survives Plan 4
> and gets *better* under it, because the heuristics stop being approximations.

---

## 1. The evidence, and which fault each section attacks

Gemma (70 meshes) on SC36 Rev C, 36 ranks, instrumented build, ~20 minutes and 850+ search nodes:

| Observation | Value | Attacked by |
| --- | --- | --- |
| Deepest partial assignment | 19 of 70 | — |
| `S4x4` placements ever attempted | **0** | §2 rigidity ordering |
| Distinct depth-1 commits | **1** | §3 restarts |
| Backtracks at depth 19 | **895** | §4 backjumping |
| Backtracks at depth 1 | **0** | §3 restarts |
| Fresh CSP solve per search node | every node | §5 constant factors, Plan 4 |

**Run §0 before any of it.**

## 0. The one-line diagnostic

```cpp
// physical_grouping_descriptor_matching.cpp:2474
constexpr std::size_t kMaxPlacementsPerVariant = 10;
```

The comment on this constant says a truncated pool can in principle hide the only solution and is the
first thing to suspect when a descriptor comes back unplaced. Gemma comes back unplaced, and nobody has
ever run this path with the cap raised.

Set it to 50, rerun, and read the depth. If the search moves past 19, the failure is truncation and §2–§4
are treating a symptom. If it stalls in the same place with the same 0 candidates for mesh 19, the
failure is ordering and this plan is aimed correctly. Either answer is worth twenty minutes.

---

## 2. Rigidity-first ordering within the connected frontier — **implemented**

### The fault

`select_next_mesh` ranked unplaced meshes by placed-neighbour count alone, ties to lowest id. That grows
one connected blob outward from mesh 0. Gemma's mesh graph chains `S4x2 → S4x1 ×5-6 → S4x2 → …`, so the
search threads flexible 4-chip strips through the fabric while the rigid 16-chip `S4x4` blocks sit at
the back of the queue behind every strip that happens to be closer to the frontier. It never reached
one.

### Why not real MRV

Fail-first ordering wants the mesh with the fewest *live* candidates. Counting those means one
`enumerate_distinct_placements_for_grouping` per unplaced mesh per search node — 70× the cost of the
thing that is already the bottleneck. Plan 4 makes it affordable; nothing else does.

### The static proxy

Two numbers, both available from `global_mesh_groupings` before the search starts, neither requiring a
solve: a mesh with a **larger footprint** and **fewer accepted grouping variants** has fewer legal
seatings. An `S4x4` (16 nodes) beats a 4x1 strip (4 nodes) on the first, decisively.

```cpp
// How few ways a mesh can be seated, as a static stand-in for its domain size. Fail-first ordering wants
// the mesh with the fewest live candidates, but counting those is one CSP enumeration per unplaced mesh
// per search node -- more expensive than the enumeration that already dominates the search. Both numbers
// here come from the grouping list alone, so the whole map is computed once before the search starts.
struct MeshRigidity {
    std::size_t footprint_asics = 0;  ///< Largest variant's node count; big shapes have fewer seatings
    std::size_t variant_count = 0;    ///< Accepted grouping variants; fewer variants, fewer seatings
};

std::map<GlobalMeshId, MeshRigidity> compute_mesh_rigidity(
    const std::map<GlobalMeshId, std::vector<GroupingInfo>>& global_mesh_groupings) {
    std::map<GlobalMeshId, MeshRigidity> rigidity;
    for (const auto& [mesh_id, variants] : global_mesh_groupings) {
        MeshRigidity entry;
        entry.variant_count = variants.size();
        for (const GroupingInfo& variant : variants) {
            // Node count rather than asic_count: the nodes are what next_step_pool actually solves over,
            // so this cannot disagree with the shape that gets placed.
            entry.footprint_asics = std::max(entry.footprint_asics, variant.adjacency_graph.get_nodes().size());
        }
        rigidity.emplace(mesh_id, entry);
    }
    return rigidity;
}
```

### The catch this design exists to avoid

Plan 3's central property is that seams are satisfied *by construction*: a mesh is only ever placed next
to something already placed, so its inter-mesh edges are checked at the moment it is committed. Ranking
by rigidity alone destroys that. Seat all the `S4x4`s first and, if they are not mutually adjacent, no
seam constraint binds while they are placed — they go down freely and the search discovers they are in
the wrong relative positions many levels later, which is a strictly worse failure mode than the one
being fixed.

So frontier membership **dominates** the ranking and rigidity only orders within it. Among meshes that
already touch the assignment, take the most rigid. `S4x4`s are pulled forward as soon as they are
reachable, and never before.

```cpp
    // Compared lexicographically, larger wins.
    struct SelectionKey {
        bool on_frontier = false;
        std::size_t footprint_asics = 0;
        std::size_t variant_count = 0;
        std::size_t placed_neighbor_count = 0;

        bool operator>(const SelectionKey& other) const {
            if (on_frontier != other.on_frontier) {
                return on_frontier;
            }
            if (footprint_asics != other.footprint_asics) {
                return footprint_asics > other.footprint_asics;
            }
            if (variant_count != other.variant_count) {
                return variant_count < other.variant_count;  // fewer variants is more constrained
            }
            return placed_neighbor_count > other.placed_neighbor_count;
        }
    };
```

Neighbour count survives as the last tiebreak, so among equally rigid choices the behaviour is exactly
what it was. The empty-frontier path is unchanged and now does double duty: it picks the seed on the
very first call, so the search opens on the most rigid mesh in the graph, and it re-seeds whenever a
disconnected component is exhausted.

`rigidity` is computed once in `start_adjacency_guided_dfs` and threaded through `place_remaining_meshes`
as a `const&`.

*Validated:* `ninja fabric` clean. Not yet run against Gemma or the PGD test suite.

---

## 3. Luby restarts

### The fault

Zero backtracks at depth 1 and one distinct depth-1 commit across the whole run. The DFS is a deep local
search that will never revisit its first decision, so if the seed placement is wrong the remaining
compute is spent proving it in ever-finer detail.

### Design

Restarting is only useful if something differs between runs, and the thing that must differ is the
*shallow* decision. Randomising the variable ordering would fight §2 directly, so randomise **value**
ordering — which candidate is tried for the mesh — and only near the root. Below the limit the heuristic
order is kept, because that is where it is doing useful work.

Budgets follow the Luby sequence (1,1,2,1,1,2,4,…), which is the standard choice when the runtime
distribution is unknown: short budgets keep being resampled so a lucky seed is found fast, while the
occasional long budget still lets a genuinely deep search finish.

```cpp
// luby(1)=1, luby(2)=1, luby(3)=2, luby(4)=1, luby(5)=1, luby(6)=2, luby(7)=4, ...
std::size_t luby(std::size_t i) {
    for (std::size_t k = 1; k < 64; ++k) {
        const std::size_t bound = (std::size_t{1} << k) - 1;
        if (i == bound) {
            return std::size_t{1} << (k - 1);
        }
        if (i < bound) {
            return luby(i - (std::size_t{1} << (k - 1)) + 1);
        }
    }
    return 1;
}

// Shuffle the candidate pool only near the root. That is where the trace says the search is stuck (one
// distinct depth-1 commit in 850 nodes); deeper levels keep the heuristic order so a restart re-roots
// the search without discarding the value ordering everywhere else.
constexpr std::size_t kRandomizedDepthLimit = 3;

void randomize_shallow_pool(std::vector<PlacementCandidate>& pool, std::size_t depth, std::mt19937_64& rng) {
    if (depth <= kRandomizedDepthLimit && pool.size() > 1) {
        std::shuffle(pool.begin(), pool.end(), rng);
    }
}
```

Reproducibility is the constraint that shapes the driver. A failure has to be reproducible from the MGD
and PSD alone, which is why the existing budget counts search nodes rather than wall clock (line 2576).
Restarts keep that: the seed is a hash of the problem, and `node_budget` becomes the total across all
restarts rather than a per-restart figure.

```cpp
constexpr std::size_t kRestartBaseBudget = 200;  // search nodes in the shortest restart
constexpr std::size_t kMaxRestarts = 64;

// Seeded from the problem, never from wall clock or address layout, so a failing run reproduces.
std::size_t restart_seed(const std::map<GlobalMeshId, std::vector<GroupingInfo>>& global_mesh_groupings) {
    std::size_t seed = 1469598103934665603ull;  // FNV-1a offset basis
    for (const auto& [mesh_id, variants] : global_mesh_groupings) {
        seed = (seed ^ static_cast<std::size_t>(*mesh_id)) * 1099511628211ull;
        for (const GroupingInfo& variant : variants) {
            seed = (seed ^ std::hash<std::string>{}(variant.name)) * 1099511628211ull;
        }
    }
    return seed;
}

AssignedMeshes start_adjacency_guided_dfs(/* ... */ std::size_t node_budget, PlacementSolveStats* stats /* ... */) {
    const std::map<GlobalMeshId, MeshRigidity> rigidity = compute_mesh_rigidity(global_mesh_groupings);
    const std::size_t seed = restart_seed(global_mesh_groupings);

    std::size_t nodes_expanded = 0;  // accumulates across restarts, so node_budget bounds the whole run
    for (std::size_t restart = 1; restart <= kMaxRestarts; ++restart) {
        std::size_t restart_nodes = 0;
        const std::size_t restart_budget = kRestartBaseBudget * luby(restart);
        std::mt19937_64 rng(seed ^ (restart * 0x9e3779b97f4a7c15ull));

        AssignedMeshes assignment = place_remaining_meshes(
            /*assignment=*/{}, global_mesh_groupings, rigidity, mesh_level_graph, physical_graph,
            physical_system_descriptor, relaxed_inter_mesh_policy, restart_nodes, restart_budget, rng, stats);

        nodes_expanded += restart_nodes;
        if (assignment.size() == mesh_level_graph.get_nodes().size()) {
            return assignment;
        }
        // Restart 1 is the pure-heuristic run (shuffling a pool the heuristic already ordered is only a
        // loss if the heuristic was right), so an unshuffled attempt is always tried first.
        if (node_budget != 0 && nodes_expanded >= node_budget) {
            break;
        }
    }
    return {};
}
```

**Open question.** Restart 1 should skip the shuffle so the deterministic heuristic run is always
attempted first; the snippet above needs a `restart == 1` guard threaded into
`randomize_shallow_pool`. Left explicit rather than silently folded in, because it changes what "the
default behaviour" means for every existing test.

---

## 4. Conflict-directed backjumping

### The fault

`S4x2` mesh 19 returns 0 candidates. The DFS responds by re-seating mesh 18, 895 times, when the ASICs
mesh 19 needed were consumed by a much shallower decision. Every one of those 895 retries rediscovers
the same conflict.

### Design

On failure, return *which placed meshes caused it* alongside the failure. A level whose own mesh is not
in that set cannot fix the conflict by re-seating, so it skips its remaining candidates and forwards the
set — that is the jump.

```cpp
// The placed meshes responsible for a failure below this node. A level whose mesh is absent from the set
// cannot repair the conflict by trying a different candidate, so it stops and forwards the set upward.
using ConflictSet = std::set<GlobalMeshId>;

struct SearchOutcome {
    AssignedMeshes assignment;  ///< Non-empty only on success
    ConflictSet conflict;       ///< Meaningful only on failure
};
```

**Soundness is the whole difficulty.** The conflict set must be a *superset* of the true causes. Too
large and the jumps get short (degenerating to today's chronological backtracking, which is merely slow);
too small and the search prunes branches that contained real solutions, which is silently wrong. Every
blame rule below is therefore built to over-approximate, and the fallback is "everything placed".

`next_step_pool` is where the information lives, so it grows an out-parameter:

```cpp
// Why the pool came back empty, populated only on failure. Blame only meshes that actually removed
// something this mesh needed: blaming every placed mesh is sound but makes the set universal, which
// turns backjumping back into chronological backtracking.
std::vector<PlacementCandidate> next_step_pool(/* ... */, ConflictSet* blame);
```

Two blame rules cover the two ways the pool empties:

```cpp
    // (a) A placed neighbour is walled in -- collect_seams_to_placed_neighbors returned nullopt because
    //     nothing free borders its region. Blame the neighbour itself and everyone squatting on its
    //     border, since those are the meshes that ate the chips this seam needed.
    if (!seams_or_blocked.has_value()) {
        if (blame != nullptr) {
            for (const auto& [neighbor_id, _] : placed_neighbors_of(mesh_id, assignment, mesh_level_graph)) {
                blame->insert(neighbor_id);
                const PlacedMesh& neighbor = *find_placed(assignment, neighbor_id);
                for (const AsicID& region_chip : neighbor.placement.asics) {
                    for (const AsicID& adjacent : physical_graph.get_neighbors(region_chip)) {
                        if (const PlacedMesh* owner = owner_of_asic(assignment, adjacent)) {
                            blame->insert(owner->mesh_id);
                        }
                    }
                }
            }
        }
        return {};
    }

    // (b) Every variant solved to nothing against the free graph. There is no cheap way to attribute
    //     that to specific occupied chips without the master candidate list from Plan 4, so blame the
    //     placed neighbours plus every mesh occupying a chip reachable from the seam boundary. Coarse
    //     but sound; Plan 4 replaces it with the exact set (the meshes whose footprints intersect the
    //     candidates that were filtered out).
```

The propagation rule at each level is the standard one:

```cpp
        SearchOutcome child = place_remaining_meshes(/* ... */);
        if (!child.assignment.empty()) {
            return child;
        }
        if (!child.conflict.contains(*next_mesh)) {
            // This level is not implicated. Re-seating it cannot help, so skip the remaining candidates
            // and let the caller keep unwinding -- this is the jump.
            return SearchOutcome{{}, std::move(child.conflict)};
        }
        // Implicated: absorb the child's conflict (minus ourselves) and keep trying siblings.
        child.conflict.erase(*next_mesh);
        accumulated_conflict.merge(child.conflict);
```

*Validation:* every descriptor that places today must still place, with an identical assignment. CBJ may
only remove work, never solutions — so a differing result is a soundness bug in a blame rule, not a
heuristic difference. That test is the reason to do §4 last.

---

## 5. Constant factors

None of these change which nodes are visited. They change how many fit in a budget, which is why they
are worth doing before measuring anything else.

**(a) Rebuilding the fabric graph at every node.** `filter_mapped_placements_in_physical_graph` (2086)
constructs a fresh `AdjacencyGraph` over all ~1150 ASICs — a `std::map` insert and a `std::vector`
allocation per free node — once per search node.

```cpp
// Check first whether MappingConstraints can express forbidden global values. If it can, occupancy
// becomes a constraint on the shared graph and the per-node rebuild disappears entirely. If it cannot,
// this is the single largest constant-factor win available and is worth adding for.
```

**(b) Recomputing occupancy at every node.** `collect_occupied_asics` (2072) rebuilds the set from
scratch. Carry it instead, as a bitset over the dense ASIC index from
[Plan 4 §3.1](TOPOLOGY_MAPPER_PLAN_4_SAT_JOINT_PLACEMENT.md) — the one piece of Plan 4 worth pulling
forward, since it is useful on its own.

**(c) Deep-copying the assignment per candidate.** Line 2586 is `AssignedMeshes branch = assignment;`,
and each `PlacedMesh` holds an `unordered_set<AsicID>` plus a `std::map`. At depth 19 with ~10 candidates
per node that is ~190 deep copies of a growing vector, per node. The copy is currently the undo
mechanism; make/undo replaces it:

```cpp
    for (PlacementCandidate& candidate : candidates) {
        assignment.push_back(PlacedMesh{*next_mesh, std::move(candidate.placement)});
        occupied.or_with(candidate.footprint);

        SearchOutcome child = place_remaining_meshes(assignment, occupied, /* ... */);
        if (!child.assignment.empty()) {
            return child;
        }

        // Undo. Footprints are disjoint from `occupied` by construction, so xor clears exactly the bits
        // that were set above. The placement moves back into the candidate so the pool stays intact.
        occupied.xor_with(candidate.footprint);
        candidate.placement = std::move(assignment.back().placement);
        assignment.pop_back();
    }
```

This makes `assignment` a mutable reference rather than a by-value parameter, which is a signature change
across the recursion.

**(d) Debug instrumentation.** 53 `PGD_DFS_DEBUG` sites, including two `std::string` copies per candidate
at 2587–2588 and a formatted `log_info` per expand, commit, and backtrack. Removal checklist at 2034.
Keep them until §2–§4 are validated — they are how the faults were found — then delete.

---

## 6. Order of work

| # | Change | Size | Why here |
| --- | --- | --- | --- |
| 0 | Raise `kMaxPlacementsPerVariant` to 50, rerun | 1 line | Decides whether the rest is aimed correctly |
| 1 | §2 rigidity ordering | ~90 lines | **Done.** Highest value per line; may fix Gemma alone |
| 2 | §5(d) delete debug logs, §5(b) incremental occupancy | ~60 lines | Buys throughput for measuring everything after |
| 3 | §3 restarts | ~60 lines | Cheap insurance against a bad seed; independent of §4 |
| 4 | §5(c) make/undo, §5(a) graph rebuild | ~80 lines | Larger refactor, no behaviour change |
| 5 | §4 backjumping | ~150 lines | Most targeted, most invasive, hardest to prove sound |

Stop as soon as Gemma places. Each step is independently revertible and none of them block Plan 4.

---

## 7. Risks

| Risk | Mitigation |
| --- | --- |
| §2 changes placements on descriptors that already work | Ordering affects which solution is found, not whether one exists. Existing PGD tests must still place all meshes; assignments may differ and expectations may need updating |
| §2 helps Gemma but the failure was truncation all along | §0 answers this first, in twenty minutes |
| §3 makes failures irreproducible | Seed is a hash of the descriptor; `node_budget` counts nodes across all restarts, not wall clock |
| §3 restart 1 shuffles and loses the deterministic run | Explicit open question in §3; must be resolved before merge |
| §4 blame rule too narrow, prunes real solutions | Every rule over-approximates, fallback is "all placed meshes". Validation is exact-assignment equality against pre-§4 on all passing descriptors |
| §5(c) make/undo leaves stale state on an early return | The budget-exhausted return at 2579 sits inside the candidate loop and must undo before returning, or unwinding corrupts every ancestor's assignment |

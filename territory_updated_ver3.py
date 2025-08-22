# 23 - 12 - 23 
import sys
import math
import random
import time
import numpy as np
try:
    from scipy.optimize import linprog
except Exception:
    linprog = None

MAX = 1005
INF = int(1e9)
MOD = int(1e9 + 7)
N = 50

def Rand(l, r):
    x = random.randint(0, int(1e18))
    return x % (r - l + 1) + l

# Input
n = p = m = 0
adj = [[] for _ in range(MAX)]
w = [[0, 0] for _ in range(MAX)]
average_w = [0.0, 0.0]
u = [0.0, 0.0]

d = [[0.0 for _ in range(MAX)] for _ in range(MAX)]
fami = [[0 for _ in range(MAX)] for _ in range(N)]
t = [0.05, 0.05]

node = [[] for _ in range(N)]
p_centroid = [0 for _ in range(N)]
territory = [0 for _ in range(MAX)]
total_sum = [[0.0, 0.0] for _ in range(MAX)]


# Added for new input format
x_coord = [0.0 for _ in range(MAX)]
y_coord = [0.0 for _ in range(MAX)]
instance_name = "instance"

def input_data():
    import sys
    global n, m, p, average_w, u, t, instance_name

    if len(sys.argv) >= 2:
        instance_name = sys.argv[1]

    # ---- Đọc dòng 1: số node ----
    n_local = int(sys.stdin.readline().strip())
    n = n_local

    average_w[0] = 0.0
    average_w[1] = 0.0
    for i in range(MAX):
        adj[i].clear()
        territory[i] = 0
        if i < N:
            p_centroid[i] = 0

    # ---- Đọc thông tin từng node ----
    for _ in range(n):
        toks = sys.stdin.readline().strip().split()
        idx = int(toks[0])
        x_coord[idx] = float(toks[1])
        y_coord[idx] = float(toks[2])
        w[idx][0] = float(toks[3])  # customers
        w[idx][1] = float(toks[4])  # orders
        average_w[0] += w[idx][0]
        average_w[1] += w[idx][1]

    # ---- Đọc số cạnh ----
    l = int(sys.stdin.readline().strip())

    # ---- Đọc cạnh ----
    for _ in range(l):
        a, b = map(int, sys.stdin.readline().strip().split())
        adj[a].append(b)
        adj[b].append(a)

    # ---- Đọc drivers, clusters, tolerances ----
    toks = sys.stdin.readline().strip().split()
    number_drivers = int(toks[0])
    number_clusters = int(toks[1])
    tol1 = float(toks[2])
    tol2 = float(toks[3])

    p_local = number_clusters
    global p
    p = p_local

    t[0] = tol1
    t[1] = tol2

    u[0] = average_w[0] / float(p)
    u[1] = average_w[1] / float(p)

    average_w[0] /= float(n)
    average_w[1] /= float(n)

    # ---- Đọc familiarity matrix ----
    for i_dr in range(1, p+1):
        vals = []
        while len(vals) < n:
            vals.extend(sys.stdin.readline().strip().split())
        for j in range(1, n+1):
            fami[i_dr][j] = int(float(vals[j-1]))

    # ---- Tính ma trận khoảng cách Euclid ----
    for i in range(1, n+1):
        for j in range(1, n+1):
            dx = x_coord[i] - x_coord[j]
            dy = y_coord[i] - y_coord[j]
            d[i][j] = math.hypot(dx, dy)


def merit_function(gamma=1000, beta=500):
    f = 0.0
    dmax = 0.0
    for i in range(1, n+1):
        f += d[i][p_centroid[territory[i]]]
        dmax = max(dmax, d[i][p_centroid[territory[i]]])
    g = [0.0, 0.0]
    for ic in range(2):
        for i in range(1, p+1):
            total_sum[i][ic] = 0.0
        for i in range(1, n+1):
            total_sum[territory[i]][ic] += w[i][ic]
        for i in range(1, p+1):
            g[ic] += max(
                total_sum[i][ic] - (1 + t[ic]) * u[ic],
                (1 - t[ic]) * u[ic] - total_sum[i][ic],
                0.0
            )
    g_ = g[0] + g[1]
    familarity = 0.0
    expected_f = 1
    for i in range(1, n+1):
        familarity += fami[p_centroid[territory[i]]][i] - expected_f
    return f + gamma * g_ + beta * familarity

def location():
    leftover = p
    for i in range(1, n+1):
        x = 0
        if leftover: 
            x = Rand(0, 1)
        if i + leftover - 1 == n: 
            x = 1
        if x:
            p_centroid[leftover] = x
        leftover -= x
    for i in range(1, n+1):
        res = -1
        mindist = INF
        for j in range(1, p+1):
            if d[i][p_centroid[j]] < mindist:
                mindist = d[i][p_centroid[j]]
                res = j
        territory[i] = res

    time_change = 0
    cnt = 0
    while True:
        time_change = 0
        for i in range(1, p+1):
            node[i].clear()
        for i in range(1, n+1):
            node[territory[i]].append(i)
        for i in range(1, p+1):
            best = INF
            res = 0
            for ix in range(len(node[i])):
                distant = 0
                for j in range(len(node[i])):
                    distant += d[node[i][ix]][node[i][j]]
                if distant < best:
                    best = distant
                    res = ix
            if node[i]:
                p_centroid[i] = node[i][res]
        for i in range(1, n+1):
            res = p_centroid[territory[i]]
            mindist = d[i][territory[i]]
            for j in range(1, p+1):
                if d[i][p_centroid[j]] < mindist:
                    time_change += 1
                    mindist = d[i][p_centroid[j]]
                    res = j
            territory[i] = res
        cnt += 1
        if not time_change or cnt > 100:
            break


def allocation():
    # Phase 4.2: LP-based allocation solved via simplex-style LP (HiGHS). 
    # Variables x[i,j] for territory i (1..p), node j (1..n), 0<=x<=1.
    # Objective: minimize sum_{i,j} d[j][centroid_i] * x[i,j].
    # Constraints:
    #   (a) For each node j: sum_i x[i,j] = 1
    #   (b) For each territory i and activity c in {0,1}:
    #       (1 - t[c]) * u[c] <= sum_j w[j][c] * x[i,j] <= (1 + t[c]) * u[c]
    # After solving, round by argmax and refine by greedy balance + local search.
    any_centroid = any(p_centroid[i] > 0 for i in range(1, p+1))
    if not any_centroid:
        _nearest_centroid_assignment()
        _greedy_balance_fix()
        local_search()
        return

    # cost vector
    def vidx(i, j):  # 0-based
        return (i-1)*n + (j-1)
    c = np.zeros(p*n, dtype=float)
    for i in range(1, p+1):
        ci = p_centroid[i]
        for j in range(1, n+1):
            c[vidx(i,j)] = float(d[j][ci]) if ci > 0 else 1e6

    # equality: each node assigned fully
    A_eq = []
    b_eq = []
    for j in range(1, n+1):
        row = np.zeros(p*n, dtype=float)
        for i in range(1, p+1):
            row[vidx(i,j)] = 1.0
        A_eq.append(row)
        b_eq.append(1.0)

    # inequalities: balance per territory and activity
    A_ub = []
    b_ub = []
    for i in range(1, p+1):
        for c_act in range(2):
            upper = (1.0 + t[c_act]) * u[c_act]
            lower = (1.0 - t[c_act]) * u[c_act]
            row_up = np.zeros(p*n, dtype=float)
            for j in range(1, n+1):
                row_up[vidx(i,j)] = w[j][c_act]
            A_ub.append(row_up)
            b_ub.append(upper)
            A_ub.append(-row_up)
            b_ub.append(-lower)

    A_eq_np = np.vstack(A_eq) if A_eq else None
    b_eq_np = np.array(b_eq, dtype=float) if b_eq else None
    A_ub_np = np.vstack(A_ub) if A_ub else None
    b_ub_np = np.array(b_ub, dtype=float) if b_ub else None
    bounds = [(0.0, 1.0) for _ in range(p*n)]

    x_sol = None
    if linprog is not None:
        res = linprog(c, A_ub=A_ub_np, b_ub=b_ub_np, A_eq=A_eq_np, b_eq=b_eq_np, bounds=bounds, method="highs")
        if res.success and res.x is not None:
            x_sol = res.x.reshape((p, n))

    if x_sol is None:
        _nearest_centroid_assignment()
        _greedy_balance_fix()
        local_search()
        return

    # round by argmax
    for j in range(1, n+1):
        col = x_sol[:, j-1]
        territory[j] = int(np.argmax(col)) + 1

    _greedy_balance_fix(max_rounds=8)
    local_search(max_iters=5)


def _territory_nodes(tid):
    return [i for i in range(1, n+1) if territory[i] == tid]

def _is_connected_subset(nodes_list):
    if not nodes_list:
        return True
    nodes_set = set(nodes_list)
    start = next(iter(nodes_set))
    from collections import deque
    q = deque([start])
    vis = {start}
    while q:
        u0 = q.popleft()
        for v0 in adj[u0]:
            if v0 in nodes_set and v0 not in vis:
                vis.add(v0)
                q.append(v0)
    return len(vis) == len(nodes_set)

def _check_move_connectivity(node_id, src_tid, dst_tid):
    # source after removing node must stay connected
    src_nodes = _territory_nodes(src_tid)
    if node_id in src_nodes:
        src_nodes.remove(node_id)
    if not _is_connected_subset(src_nodes):
        return False
    # destination with node should be connected (light check: neighbor exists or do full check)
    dst_nodes = _territory_nodes(dst_tid)
    if len(dst_nodes) == 0:
        return True
    has_neighbor = any(territory[v] == dst_tid for v in adj[node_id])
    if has_neighbor:
        return True
    # fallback full check
    return _is_connected_subset(dst_nodes + [node_id])

def _compute_totals():
    totals = [[0.0, 0.0] for _ in range(p+1)]
    for i in range(1, n+1):
        tid = territory[i]
        for c in range(2):
            totals[tid][c] += w[i][c]
    return totals

def _balance_bounds():
    lower = [[(1.0 - t[c]) * u[c] for c in range(2)] for _ in range(p+1)]
    upper = [[(1.0 + t[c]) * u[c] for c in range(2)] for _ in range(p+1)]
    return lower, upper

def _nearest_centroid_assignment():
    for i_node in range(1, n+1):
        best_tid = 1
        best_d = float('inf')
        for tid in range(1, p+1):
            c_node = p_centroid[tid]
            if c_node <= 0:
                continue
            if d[i_node][c_node] < best_d:
                best_d = d[i_node][c_node]
                best_tid = tid
        territory[i_node] = best_tid

def _greedy_balance_fix(max_rounds=5):
    # move boundary nodes from overfull to neighboring underfull territories if it improves merit
    for _ in range(max_rounds):
        improved = False
        totals = _compute_totals()
        low, up = _balance_bounds()

        # find overfull territory by how much it exceeds
        def over_amount(tid):
            val = 0.0
            for c in range(2):
                if totals[tid][c] > up[tid][c]:
                    val += (totals[tid][c] - up[tid][c])
            return val

        worst_tid = max(range(1, p+1), key=lambda tid: over_amount(tid))
        if over_amount(worst_tid) <= 1e-9:
            break

        # boundary nodes in worst_tid (touch another territory)
        candidates = []
        for j in range(1, n+1):
            if territory[j] != worst_tid:
                continue
            for v in adj[j]:
                if territory[v] != worst_tid:
                    candidates.append(j)
                    break

        base_merit = merit_function()
        best_move = None
        best_delta = 0.0
        for j in candidates:
            for v in adj[j]:
                dst_tid = territory[v]
                if dst_tid == worst_tid:
                    continue
                if not _check_move_connectivity(j, worst_tid, dst_tid):
                    continue
                # try move
                old_tid = territory[j]
                territory[j] = dst_tid
                new_merit = merit_function()
                delta = base_merit - new_merit
                territory[j] = old_tid
                if delta > best_delta + 1e-9:
                    best_delta = delta
                    best_move = (j, dst_tid)

        if best_move:
            j, dst_tid = best_move
            territory[j] = dst_tid
            improved = True

        if not improved:
            break

def local_search(max_iters=3):
    # steepest descent local search using merit function, preserve connectivity
    for _ in range(max_iters):
        base = merit_function()
        best_delta = 0.0
        best_move = None
        for j in range(1, n+1):
            src = territory[j]
            for dst in range(1, p+1):
                if dst == src:
                    continue
                if not _check_move_connectivity(j, src, dst):
                    continue
                territory[j] = dst
                new_m = merit_function()
                delta = base - new_m
                territory[j] = src
                if delta > best_delta + 1e-9:
                    best_delta = delta
                    best_move = (j, dst)
        if best_move is None:
            break
        j, dst = best_move
        territory[j] = dst



def print_output(algorithm_name="local_search", iter_idx=0, running_time=0.0):
    # best_objective: sum distances of each node to its assigned centroid (same as f-part of merit)
    f = 0.0
    print(merit_function())
    radius_per_tid = [0.0 for _ in range(p+1)]
    act1 = [0.0 for _ in range(p+1)]
    act2 = [0.0 for _ in range(p+1)]
    for i in range(1, n+1):
        tid = territory[i]
        cnode = p_centroid[tid]
        dij = d[i][cnode] if cnode>0 else 0.0
        f += dij
        radius_per_tid[tid] = max(radius_per_tid[tid], dij)
        act1[tid] += w[i][0]
        act2[tid] += w[i][1]

    # Header
    print("=========================================================================================================")
    print("description line: instance_name algorithm_name iter_idx best_objective running_time")
    print("values :", instance_name, algorithm_name, iter_idx, f, running_time)
    print("description line: nb_riders nb_polygons")
    print("values :", p, n)

    # Per rider
    for tid in range(1, p+1):
        # node list
        nodes = [i for i in range(1, n+1) if territory[i] == tid]
        nodes_str = ", ".join(str(z) for z in nodes)
        print(f"node list assigned to rider: rider {tid-1} = {{{nodes_str}}}")
        # stats line
        centroid_idx = p_centroid[tid]
        print("description line: centroid - radius - orders - customers")
        # In absence of area / act_3, set to 0.0
        print("values :", centroid_idx, radius_per_tid[tid], act1[tid], act2[tid])


def main():
    start_t = time.time()
    input_data()
    location()
    allocation()
    try:
        local_search()
    except Exception:
        pass
    elapsed = time.time() - start_t
    print_output(algorithm_name="local_search", iter_idx=0, running_time=elapsed)

if __name__ == "__main__":
    main()

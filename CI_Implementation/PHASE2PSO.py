import numpy as np

# ----------------- Projection -----------------
def project_row_to_simplex_leq1(row):
    r = np.clip(row, 0, 1)
    s = r.sum()
    if s <= 1: return r
    u = np.sort(r)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u)+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1.0)
    return np.maximum(r - theta, 0)

def project_matrix(X):
    return np.array([project_row_to_simplex_leq1(row) for row in X])

# ----------------- Fitness -----------------
def fitness(X, U, R, C, idle, slope, Plimit, penalty_w=1e6):
    reward = np.sum(R * (U.reshape(-1,1) * X))
    server_load = (U.reshape(-1,1) * X).sum(axis=0)
    frac_util = server_load / C
    power = idle + slope * (frac_util**2) * 100
    total_power = np.sum(power)
    tasks_per_server = (X > 1e-6).sum(axis=0)
    latency_penalty = 10.0 * np.sum(tasks_per_server)

    penalty = penalty_w * np.sum(np.maximum(server_load - C, 0))
    penalty += penalty_w * max(total_power - Plimit, 0)
    penalty += penalty_w * np.sum(np.maximum(X.sum(axis=1) - 1, 0))

    return -(reward - latency_penalty) + penalty

# ----------------- PSO Allocation -----------------
def pso_allocate(U, R, C, idle, slope, Plimit,
                 swarm_size=30, iters=200,
                 w=0.72, c1=1.4, c2=1.4, seed=42):
    rng = np.random.default_rng(seed)
    n_tasks, n_servers = R.shape

    # Initialize swarm
    X = rng.random((swarm_size, n_tasks, n_servers))
    for k in range(swarm_size):
        X[k] = project_matrix(X[k])
    V = rng.normal(0, 0.1, size=(swarm_size, n_tasks, n_servers))

    # Personal/global bests
    pbest = X.copy()
    pbest_val = np.array([fitness(X[k], U, R, C, idle, slope, Plimit) for k in range(swarm_size)])
    g_idx = np.argmin(pbest_val)
    gbest = pbest[g_idx].copy()
    gbest_val = pbest_val[g_idx]

    # Main loop
    for t in range(iters):
        r1 = rng.random((swarm_size, n_tasks, n_servers))
        r2 = rng.random((swarm_size, n_tasks, n_servers))
        V = w*V + c1*r1*(pbest - X) + c2*r2*(gbest - X)
        X = np.clip(X + V, 0, 1)
        for k in range(swarm_size):
            X[k] = project_matrix(X[k])
        vals = np.array([fitness(X[k], U, R, C, idle, slope, Plimit) for k in range(swarm_size)])
        improved = vals < pbest_val
        pbest[improved] = X[improved]
        pbest_val[improved] = vals[improved]
        if pbest_val.min() < gbest_val:
            g_idx = np.argmin(pbest_val)
            gbest = pbest[g_idx].copy()
            gbest_val = pbest_val[g_idx]

    # Metrics
    reward = np.sum(R * (U.reshape(-1,1) * gbest))
    server_load = (U.reshape(-1,1) * gbest).sum(axis=0)
    frac_util = server_load / C
    power = idle + slope * (frac_util**2) * 100
    total_power = np.sum(power)

    return gbest, reward, server_load, total_power

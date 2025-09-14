import numpy as np
#Triangular Membership
def triangular(x, a, b, c):
    return max(min((x - a) / (b - a + 1e-9), (c - x) / (c - b + 1e-9)), 0)
def reward_low(x): return triangular(x, 0, 0, 40)
def reward_med(x): return triangular(x, 30, 60, 90)
def reward_high(x): return triangular(x, 70, 100, 120)
def power_low(x): return triangular(x, 0, 0, 15)
def power_med(x): return triangular(x, 10, 25, 40)
def power_high(x): return triangular(x, 30, 50, 50)
def util_low(x): return triangular(x, 0, 0, 30)
def util_med(x): return triangular(x, 20, 50, 80)
def util_high(x): return triangular(x, 60, 100, 100)
priority_values = {'reject': 20, 'moderate': 50, 'strong': 90}

#Fuzzy Priority
def fuzzy_priority(reward, power, util):
    rL, rM, rH = reward_low(reward), reward_med(reward), reward_high(reward)
    pL, pM, pH = power_low(power), power_med(power), power_high(power)
    uL, uM, uH = util_low(util), util_med(util), util_high(util)
    rules = []
    rules.append(min(rH, pL) * priority_values['strong'])
    rules.append(min(rH, pH) * priority_values['moderate'])
    rules.append(min(rL, pH) * priority_values['reject'])
    rules.append(uH * priority_values['reject'])
    rules.append(min(rM, pM) * priority_values['moderate'])
    rules.append(min(rL, pL) * priority_values['moderate'])
    weights = [min(rH, pL), min(rH, pH), min(rL, pH), uH, min(rM, pM), min(rL, pL)]
    return 0 if sum(weights) == 0 else sum(rules) / sum(weights)

#Fuzzy Phase Algorithm
def fuzzy_phase(tasks, es_dict, power_budget):
    n_tasks = len(tasks)
    n_servers = len(es_dict)
    alloc_matrix = np.zeros((n_tasks, n_servers))
    unassigned_tasks = set(tasks)
    for es in es_dict.values():
        es.current_utilization = 0
    total_power = sum([es.power(0) for es in es_dict.values()])
    while unassigned_tasks and total_power <= power_budget:
        best, best_score = None, -float('inf')
        for task in unassigned_tasks:
            for es_name in task.candidates:
                es = es_dict[es_name]
                next_util = es.current_utilization + task.cpu
                if next_util > es.capacity:
                    continue

                p_before = es.power(es.current_utilization / es.capacity)
                p_after = es.power(next_util / es.capacity)
                delta_power = max(p_after - p_before, 1e-3)
                if total_power - p_before + p_after > power_budget:
                    continue

                score = fuzzy_priority(task.reward, delta_power, es.current_utilization)
                if score > best_score:
                    best = (task, es, es_name)
                    best_score = score
        if best is None:
            break
        task, es, es_name = best
        es.current_utilization += task.cpu
        alloc_matrix[task.id, es.id] = task.cpu
        total_power = sum([srv.power(srv.current_utilization / srv.capacity) for srv in es_dict.values()])
        unassigned_tasks.remove(task)
    # Outputs
    Y = {es.id: es.current_utilization for es in es_dict.values()}
    reward = sum(
        task.rewards[es.id] * alloc_matrix[task.id, es.id] / task.cpu
        for task in tasks for es in es_dict.values()
        if alloc_matrix[task.id, es.id] > 0
    )
    server_load = alloc_matrix.sum(axis=0)
    frac_util = server_load / np.array([es.capacity for es in es_dict.values()])
    power = np.array([
        es.power_idle + list(es.power_active_coeffs.values())[-1] * (u**2) * 100
        for es, u in zip(es_dict.values(), frac_util)
    ])
    total_power = power.sum()
    return Y, alloc_matrix, reward, server_load, total_power

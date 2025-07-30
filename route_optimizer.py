import pandas as pd
import numpy as np
from pyomo.environ import *
from math import radians, cos, sin, asin, sqrt

# Haversine Distance in KM
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * asin(sqrt(a))

# Load dealer data
data = pd.read_csv('dealer_data.csv')
dealer_ids = data['dealer_id'].tolist()

# Distance matrix (in minutes, assuming avg speed = 30 km/h)
dist_matrix = {}
for i in dealer_ids:
    for j in dealer_ids:
        if i != j:
            dist_km = haversine(data.loc[i, 'latitude'], data.loc[i, 'longitude'],
                                data.loc[j, 'latitude'], data.loc[j, 'longitude'])
            dist_minutes = (dist_km / 30) * 60
            dist_matrix[(i, j)] = dist_minutes
        else:
            dist_matrix[(i, j)] = 0

# Pyomo Model
model = ConcreteModel()
model.N = Set(initialize=dealer_ids)
model.x = Var(model.N, model.N, within=Binary)
model.u = Var(model.N, within=NonNegativeReals)  # For subtour elimination

# Priority reward
priority_map = dict(zip(data['dealer_id'], data['priority']))
meeting_time = dict(zip(data['dealer_id'], data['meeting_time']))

# Objective: Minimize total time (travel + meetings) - high reward for visiting high-priority dealers
model.obj = Objective(
    expr=sum(model.x[i, j] * dist_matrix[i, j] for i in model.N for j in model.N if i != j)
         - sum((4 - priority_map[i]) * sum(model.x[i, j] for j in model.N if j != i) * 10 for i in model.N),
    sense=minimize
)

# Constraints
model.constraints = ConstraintList()

# Visit each dealer at most once
for i in model.N:
    model.constraints.add(sum(model.x[i, j] for j in model.N if j != i) <= 1)
    model.constraints.add(sum(model.x[j, i] for j in model.N if j != i) <= 1)

# Start and end at depot (id 0)
model.constraints.add(sum(model.x[0, j] for j in model.N if j != 0) == 1)
model.constraints.add(sum(model.x[i, 0] for i in model.N if i != 0) == 1)

# Total time constraint
model.constraints.add(
    sum(model.x[i, j] * dist_matrix[i, j] for i in model.N for j in model.N if i != j)
    + sum(meeting_time[i] * sum(model.x[i, j] for j in model.N if i != j) for i in model.N)
    <= 480  # 8 hours
)

# Subtour elimination (MTZ constraints)
for i in model.N:
    for j in model.N:
        if i != j and i != 0 and j != 0:
            model.constraints.add(model.u[i] - model.u[j] + len(dealer_ids) * model.x[i, j] <= len(dealer_ids) - 1)

# Solve
solver = SolverFactory('glpk')  # or use 'cbc' if installed
results = solver.solve(model, tee=True)

# Extract route
route = []
current = 0
visited = set()
while True:
    for j in model.N:
        if j != current and value(model.x[current, j]) > 0.5:
            route.append((current, j))
            visited.add(current)
            current = j
            break
    if current == 0 or len(visited) == len(dealer_ids):
        break

# Display route
print("Optimal Route (with priorities):")
for i, j in route:
    print(f"{i} ➝ {j} (Priority: {priority_map[j]}, Meeting time: {meeting_time[j]} mins)")

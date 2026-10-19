"""Bounded task assignment, time-expanded grid reservations and consensus fixtures."""
from collections import deque
import itertools
import json
import math


class NoPlan(ValueError):pass


def scalar(value):
    if type(value) not in (int,float):raise ValueError('finite numeric value required')
    try:valid=math.isfinite(value)
    except OverflowError:valid=False
    if not valid:raise ValueError('finite numeric value required')
    return float(value)


def assignment(costs,optimal=True):
    n=len(costs)
    if not 1<=n<=8 or any(len(row)!=n for row in costs):raise ValueError('use a square matrix of size1..8')
    costs=[[None if value is None else scalar(value) for value in row] for row in costs]
    if any(value is not None and value<0 for row in costs for value in row):raise ValueError('negative cost')
    if optimal:
        options=[(sum(costs[i][j] for i,j in enumerate(order)),order)
                 for order in itertools.permutations(range(n))
                 if all(costs[i][j] is not None for i,j in enumerate(order))]
        if not options:raise NoPlan('no complete capability-compatible assignment')
        cost,order=min(options)
    else:
        available=set(range(n));order=[];cost=0.
        for i in range(n):
            options=[(costs[i][j],j) for j in available if costs[i][j] is not None]
            if not options:raise NoPlan('greedy assignment stranded a robot')
            value,j=min(options);order.append(j);available.remove(j);cost+=value
        order=tuple(order)
    return {'task_per_robot':order,'total_cost':cost}


class Grid:
    def __init__(self,width,height,blocked=()):
        if any(type(v) is not int or not 1<=v<=30 for v in (width,height)):
            raise ValueError('grid dimensions must be integers in1..30')
        self.width,self.height=width,height;self.blocked=frozenset(blocked)
        for cell in self.blocked:self.check(cell,allow_blocked=True)

    def check(self,cell,allow_blocked=False):
        if (not isinstance(cell,tuple) or len(cell)!=2 or any(type(v) is not int for v in cell)
                or not 0<=cell[0]<self.width or not 0<=cell[1]<self.height
                or (not allow_blocked and cell in self.blocked)):
            raise ValueError('invalid or blocked cell')

    def neighbors(self,cell):
        x,y=cell
        for dx,dy in [(1,0),(-1,0),(0,-1),(0,1),(0,0)]:
            candidate=(x+dx,y+dy)
            if 0<=candidate[0]<self.width and 0<=candidate[1]<self.height and candidate not in self.blocked:
                yield candidate


def horizon_check(horizon):
    if type(horizon) is not int or not 0<=horizon<=100:raise ValueError('horizon must be in0..100')


def validate_path(grid,path,horizon):
    horizon_check(horizon)
    if not path or len(path)>horizon+1:raise ValueError('path must fit the horizon')
    for cell in path:grid.check(cell)
    if any(abs(a[0]-b[0])+abs(a[1]-b[1])>1 for a,b in zip(path,path[1:])):
        raise ValueError('path jumps over grid cells')


def padded(path,horizon):return tuple(path)+tuple([path[-1]]*(horizon+1-len(path)))


def conflicts(paths,horizon):
    """Independent pairwise checker. Robots hold the last cell through the horizon."""
    horizon_check(horizon)
    if any(not path or len(path)>horizon+1 for path in paths.values()):raise ValueError('invalid path horizon')
    full={name:padded(path,horizon) for name,path in paths.items()};found=[]
    for a,b in itertools.combinations(sorted(full),2):
        for t in range(horizon+1):
            if full[a][t]==full[b][t]:found.append({'kind':'vertex','time':t,'robots':[a,b]})
            if t<horizon and full[a][t]==full[b][t+1] and full[b][t]==full[a][t+1] and full[a][t]!=full[a][t+1]:
                found.append({'kind':'edge_swap','time':t,'robots':[a,b]})
    return found


class Reservations:
    def __init__(self,grid,horizon):
        horizon_check(horizon);self.grid,self.horizon=grid,horizon
        self.vertices=set();self.edges=set()

    def add(self,path):
        validate_path(self.grid,path,self.horizon)
        full=padded(path,self.horizon)
        vertices={(t,cell) for t,cell in enumerate(full)}
        edges={(t,a,b) for t,(a,b) in enumerate(zip(full,full[1:]))}
        if self.vertices & vertices or any((t,b,a) in self.edges for t,a,b in edges):
            raise NoPlan('reservation conflict')
        self.vertices.update(vertices);self.edges.update(edges)

    def plan(self,start,goal):
        self.grid.check(start);self.grid.check(goal)
        if (0,start) in self.vertices:raise NoPlan('start occupied at time0')
        queue=deque([(start,0)]);previous={(start,0):None}
        while queue:
            cell,t=queue.popleft()
            if cell==goal and all((future,goal) not in self.vertices for future in range(t,self.horizon+1)):
                path=[];current=(cell,t)
                while current is not None:
                    path.append(current[0]);current=previous[current]
                return tuple(reversed(path))
            if t==self.horizon:continue
            for next_cell in self.grid.neighbors(cell):
                state=(next_cell,t+1)
                if state in previous or (t+1,next_cell) in self.vertices or (t,next_cell,cell) in self.edges:continue
                previous[state]=(cell,t);queue.append(state)
        raise NoPlan('no path under these priorities/reservations/horizon')


def plan_fleet(grid,requests,horizon):
    """Plan in given order; publish no partial plan when a later robot fails."""
    if len({name for name,start,goal in requests})!=len(requests):raise ValueError('duplicate robot name')
    reservations=Reservations(grid,horizon);plans={}
    for name,start,goal in requests:
        path=reservations.plan(start,goal);reservations.add(path);plans[name]=path
    assert not conflicts(plans,horizon)
    return plans


def consensus_step(values,edges,gain):
    values=[scalar(value) for value in values];gain=scalar(gain)
    if gain<0:raise ValueError('negative consensus gain')
    if not values:raise ValueError('empty team')
    unique=set();result=values.copy()
    for a,b in edges:
        if any(type(i) is not int or not 0<=i<len(values) for i in (a,b)) or a==b:
            raise ValueError('invalid undirected edge')
        key=tuple(sorted((a,b)))
        if key in unique:raise ValueError('duplicate edge')
        unique.add(key)
        difference=gain*(values[b]-values[a])
        result[a]+=difference;result[b]-=difference
    return [scalar(value) for value in result]


def demo():
    grid=Grid(5,3)
    requests=[('A',(0,1),(4,1)),('B',(4,1),(0,1))]
    independent={name:Reservations(grid,12).plan(start,goal) for name,start,goal in requests}
    coordinated=plan_fleet(grid,requests,12)
    branch=Grid(3,2,blocked=[(0,1),(2,1)])
    witness={'A':((0,0),(1,0),(1,1),(1,0)), 'B':((2,0),(2,0),(1,0),(0,0))}
    for path in witness.values():validate_path(branch,path,6)
    assert not conflicts(witness,6)
    try:plan_fleet(branch,[('A',(0,0),(1,0)),('B',(2,0),(0,0))],6)
    except NoPlan:priority_failure=True
    else:raise AssertionError('fixture should expose greedy priority incompleteness')
    values=[0.,2.];stable=[values]
    for _ in range(4):values=consensus_step(values,[(0,1)],.25);stable.append(values)
    return {'scope':'discrete synthetic point robots, shared integer clock, static known grid',
            'greedy_assignment':assignment([[1,2],[2,100]],False),
            'optimal_assignment':assignment([[1,2],[2,100]],True),
            'independent_paths':independent,'independent_conflicts':conflicts(independent,12),
            'coordinated_paths':coordinated,'coordinated_conflicts':conflicts(coordinated,12),
            'priority_failure_despite_joint_solution':priority_failure,'joint_solution_witness':witness,
            'consensus_gain_quarter':stable,'consensus_gain_one':[consensus_step([0.,2.],[(0,1)],1.)],
            'limitations':['no finite footprint, continuous dynamics or braking','no delayed/lost messages or clock skew',
                           'prioritized planning is incomplete; grid success does not establish physical collision avoidance']}


if __name__=='__main__':print(json.dumps(demo(),indent=2,allow_nan=False))

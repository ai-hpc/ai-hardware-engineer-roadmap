import unittest
from coordination import (assignment,Grid,NoPlan,Reservations,conflicts,plan_fleet,
                           validate_path,consensus_step,demo)


class AllocationTests(unittest.TestCase):
    def test_greedy_is_not_global_optimum(self):
        costs=[[1,2],[2,100]]
        self.assertEqual(assignment(costs,False)['total_cost'],101)
        self.assertEqual(assignment(costs,True),{'task_per_robot':(1,0),'total_cost':4.})

    def test_capability_constraints_and_incomplete_greedy(self):
        costs=[[1,2],[1,None]]
        with self.assertRaises(NoPlan):assignment(costs,False)
        self.assertEqual(assignment(costs)['task_per_robot'],(1,0))
        with self.assertRaises(NoPlan):assignment([[None]])

    def test_invalid_costs(self):
        for value in [[],[[1,2]],[[float('nan')]],[[True]],[[-1]]]:
            with self.assertRaises(ValueError):assignment(value)


class ReservationTests(unittest.TestCase):
    def test_independent_shortest_paths_collide_and_joint_plan_avoids_it(self):
        result=demo()
        self.assertTrue(result['independent_conflicts'])
        self.assertEqual(result['coordinated_conflicts'],[])
        self.assertEqual(result['coordinated_paths']['B'][-1],(0,1))

    def test_opposite_edges_conflict_without_vertex_overlap(self):
        result=conflicts({'A':[(0,0),(1,0)],'B':[(1,0),(0,0)]},1)
        self.assertEqual([row['kind'] for row in result],['edge_swap'])

    def test_goal_is_occupied_after_short_path_ends(self):
        result=conflicts({'A':[(0,0),(1,0)],'B':[(2,1),(2,0),(1,0)]},3)
        self.assertTrue(any(row['kind']=='vertex' and row['time']==2 for row in result))

    def test_start_at_goal_must_leave_if_reserved_later(self):
        table=Reservations(Grid(3,2),4);table.add([(0,0),(1,0),(2,0)])
        path=table.plan((1,0),(1,0))
        self.assertGreater(len(path),1)
        self.assertNotEqual(path[1],(1,0))
        self.assertEqual(path[-1],(1,0))
        table.add(path)

    def test_reservation_rejection_does_not_mutate_table(self):
        table=Reservations(Grid(2,1),2);table.add([(0,0),(1,0)])
        before=(table.vertices.copy(),table.edges.copy())
        with self.assertRaises(NoPlan):table.add([(1,0),(0,0)])
        self.assertEqual((table.vertices,table.edges),before)

    def test_narrow_corridor_cannot_swap(self):
        with self.assertRaises(NoPlan):plan_fleet(Grid(3,1),[('A',(0,0),(2,0)),('B',(2,0),(0,0))],6)

    def test_priority_failure_is_not_proof_no_joint_solution_exists(self):
        result=demo();self.assertTrue(result['priority_failure_despite_joint_solution'])
        witness=result['joint_solution_witness']
        self.assertEqual(conflicts(witness,6),[])

    def test_invalid_path_grid_start_and_horizon(self):
        grid=Grid(3,2,[(2,1)])
        for path in [[(0,0),(2,0)],[(2,1)],[]]:
            with self.assertRaises(ValueError):validate_path(grid,path,3)
        with self.assertRaises(ValueError):Reservations(grid,True)
        with self.assertRaises(ValueError):Grid(0,2)
        with self.assertRaises(ValueError):plan_fleet(grid,[('A',(0,0),(1,0)),('A',(1,0),(2,0))],3)

    def test_zero_horizon_only_allows_existing_goal(self):
        table=Reservations(Grid(2,1),0)
        self.assertEqual(table.plan((0,0),(0,0)),((0,0),))
        with self.assertRaises(NoPlan):table.plan((0,0),(1,0))


class ConsensusTests(unittest.TestCase):
    def test_simultaneous_update_preserves_average(self):
        values=[0.,2.]
        updated=consensus_step(values,[(0,1)],.25)
        self.assertEqual(updated,[.5,1.5]);self.assertEqual(values,[0.,2.])
        self.assertEqual(sum(updated),sum(values))

    def test_gain_one_swaps_without_convergence(self):
        first=consensus_step([0.,2.],[(0,1)],1.)
        self.assertEqual(first,[2.,0.])
        self.assertEqual(consensus_step(first,[(0,1)],1.),[0.,2.])

    def test_disconnected_agent_does_not_join_consensus(self):
        values=[0.,2.,10.]
        for _ in range(20):values=consensus_step(values,[(0,1)],.25)
        self.assertAlmostEqual(values[0],1.,places=5)
        self.assertEqual(values[2],10.)

    def test_duplicate_edge_and_nonfinite_rejected(self):
        for edges in [[(0,0)],[(0,2)],[(0,1),(1,0)],[(True,0)]]:
            with self.assertRaises(ValueError):consensus_step([0,1],edges,.25)
        with self.assertRaises(ValueError):consensus_step([0,float('inf')],[],.1)


if __name__=='__main__':unittest.main()

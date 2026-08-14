import unittest
from functools import partial

import networkx as nx

from Metrics import GraphMetricCollector
from RandomFailure import ProgressiveRandomFailure
from SimulationRunner import SimulationRunner


class ConnectivityMetricTests(unittest.TestCase):
    def test_giant_component_majority_is_not_treated_as_collapse(self):
        graph = nx.path_graph(10)
        graph.add_edge(10, 11)

        metrics = GraphMetricCollector(graph).collect_metrics()

        self.assertTrue(metrics["connectivity"])

    def test_failure_stops_at_collapse_stage_without_processing_another_stage(self):
        graph = nx.path_graph(10)
        failure_runner = ProgressiveRandomFailure(graph, failure_rate=0.0)
        failure_stages = 0

        def collapse_on_first_stage():
            nonlocal failure_stages
            failure_stages += 1
            if failure_stages > 1:
                self.fail("Failure processing continued after connectivity collapsed")
            graph.remove_edges_from([(2, 3), (6, 7)])
            return 2

        failure_runner.apply_single_stage_failure = collapse_on_first_stage

        _, metrics = failure_runner.apply_overall_failure(
            keep_graph_snapshots=False,
        )

        self.assertEqual(failure_stages, 1)
        self.assertEqual(len(metrics), 2)
        self.assertFalse(metrics[-1]["connectivity"])

    def test_parallel_runs_use_process_workers(self):
        runs = SimulationRunner().run_experiment(
            graph_factory=partial(nx.path_graph, 8),
            num_runs=2,
            failure_rate=0.5,
            max_stages=1,
            run_parallel=True,
            max_workers=2,
        )

        self.assertEqual(len(runs), 2)
        self.assertTrue(all(run for run in runs))


if __name__ == "__main__":
    unittest.main()

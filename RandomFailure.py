
import numpy as np
from Metrics import GraphMetricCollector


class ProgressiveRandomFailure:
    """Handles progressive random edge failure of a network."""

    def __init__(self, graph, failure_rate, max_stages=None):
        """
        Constructor for the ProgressiveRandomFailure class.

        :param graph: The graph on which to perform random failure.
        :param failure_rate: The rate at which edges will fail (between 0 and 1).
        :param max_stages: Optional safety cap on number of degradation stages.
        """
        self.graph = graph
        self.failure_rate = failure_rate
        self.max_stages = max_stages

    def apply_overall_failure(self, should_stop=None, keep_graph_snapshots=True):
        """
        Progressively apply random edge failures and collect metrics at each step.
        Graph snapshots are retained only for the initial state and the first
        collapse state (or final state if collapse never occurs).

        :param keep_graph_snapshots: Whether to retain initial/collapse graph copies.
        :return: (graphs, metrics) where graphs has two entries: [initial, collapse_or_final].
        """
        metric_collector = GraphMetricCollector(self.graph)
        initial_graph = self.graph.copy() if keep_graph_snapshots else None
        metrics_at_stages = [metric_collector.collect_metrics()]
        collapse_graph = None
        stages_completed = 0

        if not metrics_at_stages[0]["connectivity"]:
            collapse_graph = self.graph.copy() if keep_graph_snapshots else None

        while collapse_graph is None:
            if should_stop and should_stop():
                break

            if self.max_stages is not None and stages_completed >= self.max_stages:
                break

            removed_edges = self.apply_single_stage_failure()
            if removed_edges == 0:
                break

            stages_completed += 1
            stage_metrics = metric_collector.collect_metrics()
            metrics_at_stages.append(stage_metrics)
            if not stage_metrics["connectivity"]:
                collapse_graph = self.graph.copy() if keep_graph_snapshots else None

        if collapse_graph is None and keep_graph_snapshots:
            collapse_graph = self.graph.copy()

        return [initial_graph, collapse_graph], metrics_at_stages

    def apply_single_stage_failure(self):
        """
        Function applies a single stage of random failure to the graph based on the specified failure rate. 
        Failure occurs on edges only and not on nodes. 
        :return: Number of edges removed in this stage.
        """
        edges = list(self.graph.edges())
        edge_count = len(edges)
        if edge_count == 0:
            return 0

        # Equivalent to independent Bernoulli edge removals but much faster.
        remove_count = int(np.random.binomial(edge_count, self.failure_rate))
        if remove_count <= 0:
            return 0

        if remove_count >= edge_count:
            edges_to_remove = edges
        else:
            indices = np.random.choice(edge_count, size=remove_count, replace=False)
            edges_to_remove = [edges[i] for i in indices]

        self.graph.remove_edges_from(edges_to_remove)
        return len(edges_to_remove)
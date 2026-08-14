
import numpy as np
from Metrics import GraphMetricCollector


class ProgressiveRandomFailure:
    """
    Class that applies progressive random edge failure to a network and collects metrics at each stage.
    """

    def __init__(self, graph, failure_rate, max_stages=None):
        """
        Constructor for ProgressiveRandomFailure.

        @param graph The NetworkX graph on which to perform random failure.
        @param failure_rate The probability of each edge being removed at every stage (between 0 and 1).
        @param max_stages Optional upper bound on the number of degradation stages.
        """
        self.graph = graph
        self.failure_rate = failure_rate
        self.max_stages = max_stages

    def apply_overall_failure(self, should_stop=None, keep_graph_snapshots=True, stop_on_collapse=True):
        """
        Method that iteratively applies random edge failures until overall connectivity collapses.
        Metrics are collected after every stage. Graph snapshots are retained only for the initial
        state and the first collapse state (or final state if collapse never occurs).

        @param should_stop Optional callable that returns True when the caller requests early termination.
        @param keep_graph_snapshots Whether to retain deep copies of the initial and collapse graphs.
        @param stop_on_collapse Whether to stop immediately once overall connectivity collapses.
        @return A tuple (graphs, metrics) where graphs is [initial, collapse_or_final] and metrics
                is the list of metric dicts collected at each stage.
        """
        metric_collector = GraphMetricCollector(self.graph)
        initial_graph = self.graph.copy() if keep_graph_snapshots else None
        metrics_at_stages = [metric_collector.collect_metrics()]
        collapse_graph = None
        has_collapsed = not metrics_at_stages[0]["connectivity"]
        stages_completed = 0

        if has_collapsed and keep_graph_snapshots:
            collapse_graph = self.graph.copy()

        while not has_collapsed:
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

            if stop_on_collapse and not stage_metrics["connectivity"]:
                has_collapsed = True
                if keep_graph_snapshots:
                    collapse_graph = self.graph.copy()
                break

            if self.graph.number_of_edges() == 0:
                break

        if collapse_graph is None and keep_graph_snapshots:
            collapse_graph = self.graph.copy()

        return [initial_graph, collapse_graph], metrics_at_stages

    def apply_single_stage_failure(self):
        """
        Method that removes a Binomially-sampled number of edges from the graph in a single stage.
        Only edges are removed; nodes are never deleted.

        @return The number of edges removed in this stage.
        """
        edges = list(self.graph.edges())
        edge_count = len(edges)
        if edge_count == 0:
            return 0

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
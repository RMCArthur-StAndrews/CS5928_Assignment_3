
from collections import Counter

import networkx as nx


class GraphMetricCollector:
    """
    Class that computes and returns structural metrics for a NetworkX graph at a single point in time.
    Large graphs use sampling to keep runtime bounded.
    """

    # Constant for the threshold above which exact shortest path calculations are replaced with sampling.
    EXACT_SHORTEST_PATH_THRESHOLD = 1200
    # Constant for the threshold above which exact diameter calculations are replaced with sampling.
    EXACT_DIAMETER_THRESHOLD = 400
    # Constant for the number of source nodes to sample when estimating average shortest path and diameter.
    SAMPLED_SOURCE_COUNT = 8

    def __init__(self, graph):
        """
        Constructor for GraphMetricCollector.

        @param graph The NetworkX graph to collect metrics from.
        """
        self.graph = graph

    def _distance_metrics_for_connected_graph(self, graph, node_count):
        """
        Method that computes average shortest path length and diameter for a fully connected graph,
        using exact algorithms for small graphs and sampling for large ones.

        @param graph A connected NetworkX graph.
        @param node_count The number of nodes in the graph.
        @return A tuple (average_shortest_path, diameter).
        """
        if node_count <= 1:
            return 0.0, 0

        sampled_stats = None
        if (
            node_count > self.EXACT_SHORTEST_PATH_THRESHOLD
            or node_count > self.EXACT_DIAMETER_THRESHOLD
        ):
            sampled_stats = self._sample_distance_stats(graph)

        if node_count <= self.EXACT_SHORTEST_PATH_THRESHOLD:
            average_shortest_path = nx.average_shortest_path_length(graph)
        else:
            average_shortest_path = sampled_stats["average"]

        if node_count <= self.EXACT_DIAMETER_THRESHOLD:
            diameter = nx.diameter(graph)
        else:
            diameter = sampled_stats["diameter"]

        return average_shortest_path, diameter

    def _component_stats(self):
        """
        Method that computes connected-component statistics for the current graph state.

        @return A dict containing is_connected, connectivity, cluster_sizes_and_counts,
                and largest_component_nodes.
        """
        node_count = self.graph.number_of_nodes()
        if node_count == 0:
            return {
                "is_connected": False,
                "connectivity": False,
                "cluster_sizes_and_counts": [],
                "largest_component_nodes": set(),
            }

        cluster_counts = Counter()
        component_count = 0
        largest_component_size = 0
        largest_component_nodes = set()

        for component in nx.connected_components(self.graph):
            component_count += 1
            component_size = len(component)
            cluster_counts[component_size] += 1
            if component_size > largest_component_size:
                largest_component_size = component_size
                largest_component_nodes = component

        connectivity = component_count == 1

        return {
            "is_connected": connectivity,
            "connectivity": connectivity,
            "cluster_sizes_and_counts": sorted(cluster_counts.items()),
            "largest_component_nodes": largest_component_nodes,
        }

    def _sample_distance_stats(self, graph, sample_size=None):
        """
        Method that approximates average shortest path and diameter via BFS from a small sample of nodes.

        @param graph The NetworkX graph to sample distances from.
        @param sample_size Optional number of source nodes to sample; defaults to SAMPLED_SOURCE_COUNT.
        @return A dict with keys "average" (estimated mean path length) and "diameter" (estimated diameter).
        """
        node_count = graph.number_of_nodes()
        if node_count == 0:
            return {"average": 0.0, "diameter": 0}

        sample_size = min(sample_size or self.SAMPLED_SOURCE_COUNT, node_count)
        nodes = list(graph.nodes())
        step = max(node_count // sample_size, 1)
        sampled_nodes = nodes[::step][:sample_size]

        distance_sum = 0
        pair_count = 0
        diameter_estimate = 0

        for source in sampled_nodes:
            lengths = nx.single_source_shortest_path_length(graph, source)
            if lengths:
                diameter_estimate = max(diameter_estimate, max(lengths.values()))
            distance_sum += sum(lengths.values())
            pair_count += max(len(lengths) - 1, 0)

        average_estimate = 0.0 if pair_count == 0 else (distance_sum / pair_count)
        return {"average": average_estimate, "diameter": diameter_estimate}
        
    
    def collect_metrics(self):
        """
        Method that collects all structural metrics for the current graph state and returns them as a dict.

        @return A dict containing connectivity, cluster_sizes_and_counts, average_shortest_path,
                diameter, largest_component_diameter, average_degree, average_clustering,
                and degree_distribution.
        """
        component_stats = self._component_stats()

        is_connected = component_stats["is_connected"]

        average_shortest_path = float("inf")
        diameter = float("inf")
        largest_component_diameter = 0

        if is_connected:
            node_count = self.graph.number_of_nodes()
            average_shortest_path, diameter = self._distance_metrics_for_connected_graph(
                self.graph,
                node_count,
            )
            largest_component_diameter = diameter
        else:
            largest_nodes = component_stats.get("largest_component_nodes", set())
            if largest_nodes:
                largest_component_graph = self.graph.subgraph(largest_nodes)
                largest_count = largest_component_graph.number_of_nodes()
                average_shortest_path, largest_component_diameter = (
                    self._distance_metrics_for_connected_graph(
                        largest_component_graph,
                        largest_count,
                    )
                )

        diameter = largest_component_diameter

        average_degree, degree_distribution = self._degree_stats()
        average_clustering = self._average_clustering()

        metrics = {
            "connectivity": component_stats["connectivity"],
            "cluster_sizes_and_counts": component_stats["cluster_sizes_and_counts"],
            "average_shortest_path": average_shortest_path,
            "diameter": diameter,
            "largest_component_diameter": largest_component_diameter,
            "average_degree": average_degree,
            "average_clustering": average_clustering,
            "degree_distribution": degree_distribution,
        }
        return metrics

    def _average_clustering(self):
        """
        Method that computes the average clustering coefficient, using a node sample for large graphs
        to keep runtime bounded.

        @return The average clustering coefficient as a float between 0 and 1.
        """
        node_count = self.graph.number_of_nodes()
        if node_count == 0:
            return 0.0
        if node_count <= 1000:
            return nx.average_clustering(self.graph)
        nodes = list(self.graph.nodes())
        step = max(node_count // 500, 1)
        sample = nodes[::step][:500]
        clustering = nx.clustering(self.graph, nodes=sample)
        return sum(clustering.values()) / len(clustering) if clustering else 0.0

    def _degree_stats(self):
        """
        Method that computes average degree and the full degree distribution in a single pass over the graph.

        @return A tuple (average_degree, degree_distribution) where degree_distribution is a dict
                mapping degree value to node count.
        """
        degree_sum = 0
        node_count = 0
        degree_count = Counter(d for _, d in self.graph.degree())
        for degree, count in degree_count.items():
            degree_sum += degree * count
            node_count += count

        average_degree = 0.0 if node_count == 0 else degree_sum / node_count
        return average_degree, dict(sorted(degree_count.items()))
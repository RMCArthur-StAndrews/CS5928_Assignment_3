
from collections import Counter

import networkx as nx


class GraphMetricCollector:
    EXACT_SHORTEST_PATH_THRESHOLD = 1200
    EXACT_DIAMETER_THRESHOLD = 400
    SAMPLED_SOURCE_COUNT = 8

    def __init__(self, graph):
        self.graph = graph

    def _distance_metrics_for_connected_graph(self, graph, node_count):
        """Return average shortest path and diameter for a connected graph."""
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
        """Compute connected-component-derived stats once per stage."""
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

        connectivity = (largest_component_size / node_count) >= 0.5

        return {
            "is_connected": component_count == 1,
            "connectivity": connectivity,
            "cluster_sizes_and_counts": sorted(cluster_counts.items()),
            "largest_component_nodes": largest_component_nodes,
        }

    def _sample_distance_stats(self, graph, sample_size=None):
        """Approximate distance stats from a bounded set of BFS source nodes."""
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
        """Collect and return graph metrics as a dictionary."""
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

        # Always use the largest-component diameter as the "network diameter"
        # (end-to-end width of the network at this stage).
        diameter = largest_component_diameter

        average_degree, degree_distribution = self._degree_stats()

        metrics = {
            "connectivity": component_stats["connectivity"],
            "cluster_sizes_and_counts": component_stats["cluster_sizes_and_counts"],
            "average_shortest_path": average_shortest_path,
            "diameter": diameter,
            "largest_component_diameter": largest_component_diameter,
            "average_degree": average_degree,
            "degree_distribution": degree_distribution,
        }
        return metrics

    def _degree_stats(self):
        """Return average degree and degree distribution in one graph pass."""
        degree_sum = 0
        node_count = 0
        degree_count = Counter(d for _, d in self.graph.degree())
        for degree, count in degree_count.items():
            degree_sum += degree * count
            node_count += count

        average_degree = 0.0 if node_count == 0 else degree_sum / node_count
        return average_degree, dict(sorted(degree_count.items()))
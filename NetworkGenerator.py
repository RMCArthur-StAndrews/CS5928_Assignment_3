import networkx as nx


def generate_er_network(node_count, edge_probability):
    return nx.erdos_renyi_graph(n=node_count, p=edge_probability)


def generate_power_law_network(node_count, exponent):
    return nx.powerlaw_cluster_graph(n=node_count, m=int(exponent), p=0.0)



class CreatePowerLawNetwork:
    """
    Class for generating Power-Law (scale-free) networks via preferential attachment.
    """

    def __init__(self, node_count, exponent):
        """
        Constructor for CreatePowerLawNetwork.

        @param node_count The number of nodes in the network.
        @param exponent The number of edges each new node attaches to existing nodes (m value).
        """
        self.node_count = node_count
        self.exponent = exponent

    def generate(self):
        """
        Function that generates a Power-Law network using the powerlaw_cluster_graph algorithm.

        @return A NetworkX undirected Power-Law graph.
        """
        return generate_power_law_network(self.node_count, self.exponent)

class CreateERNetwork:
    """
    Class for generating Erdős–Rényi random networks.
    """

    def __init__(self, node_count, edge_probability):
        """
        Constructor for CreateERNetwork.

        @param node_count The number of nodes in the network.
        @param edge_probability The probability of an edge existing between any pair of nodes.
        """
        self.node_count = node_count
        self.edge_probability = edge_probability


    def generate(self):
        """
        Function that generates an Erdős–Rényi random network.

        @return A NetworkX undirected Erdős–Rényi graph.
        """
        return generate_er_network(self.node_count, self.edge_probability)


class CreateGridNetwork:
    """
    Class for generating regular grid (lattice) networks of arbitrary dimension.
    """

    def __init__(self, node_count, dimensions):
        """
        Constructor for CreateGridNetwork.

        @param node_count The target number of nodes; the actual count may differ slightly due to rounding.
        @param dimensions The number of dimensions for the grid (e.g. 2 for a 2D lattice).
        """
        self.node_count = node_count
        self.dimensions = dimensions

    def _grid_dimensions(self):
        """
        Function that computes the per-axis size list required by networkx.grid_graph.

        @return A list of range objects, one per dimension, each of length side = round(node_count^(1/d)).
        """
        if isinstance(self.dimensions, int):
            side = int(round(self.node_count ** (1 / self.dimensions)))
            return [range(side) for _ in range(self.dimensions)]
        return [range(d) for d in self.dimensions]

    def generate(self):
        """
        Function that generates a non-periodic grid network with the configured dimensions.

        @return A NetworkX undirected grid graph.
        """
        return nx.grid_graph(dim=self._grid_dimensions(), periodic=False)

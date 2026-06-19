import networkx as nx



class CreatePowerLawNetwork:
    def __init__(self, node_count, exponent):
        """
        Constructor for the CreatePowerLawNetwork class.
        
        :param node_count: The number of nodes in the power-law network.
        :param exponent: The exponent for the power-law degree distribution.
        """
        self.node_count = node_count
        self.exponent = exponent

    def generate(self):
        """
        Generates a power-law network.
        
        :return: A NetworkX power-law network graph.
        """
        return nx.powerlaw_cluster_graph(n=self.node_count, m=int(self.exponent), p=0.0)

class CreateERNetwork:
    def __init__(self, node_count, edge_probability):
        """
        Constructor for the CreateERNetwork class.
        
        :param node_count: The number of nodes in the ER network.
        :param edge_probability: The probability of edge creation between any two nodes.
        """
        self.node_count = node_count
        self.edge_probability = edge_probability


    def generate(self):
        """
        Generates an Erdős-Rényi (ER) network.
        
        :return: A NetworkX ER network graph.
        """
        return nx.erdos_renyi_graph(n=self.node_count, p=self.edge_probability)


class CreateGridNetwork:
    def __init__(self, node_count, dimensions):
        """
        Constructor for the CreateGridNetwork class.
        
        :param node_count: The number of nodes in the grid network.
        :param dimensions: The dimensions of the grid network (e.g., 2 for a 2D grid).
        """
        self.node_count = node_count
        self.dimensions = dimensions

    def _grid_dimensions(self):
        """
        Build the dimension argument for networkx.grid_graph.
        """
        if isinstance(self.dimensions, int):
            side = int(round(self.node_count ** (1 / self.dimensions)))
            return [range(side) for _ in range(self.dimensions)]
        return [range(d) for d in self.dimensions]

    def generate(self):
        """
        Generates a grid network.
        
        :return: A NetworkX grid network graph.
        """
        return nx.grid_graph(dim=self._grid_dimensions(), periodic=False)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _collapse_stage(run_metrics):
    """
    Function that finds the first stage index at which connectivity drops to False.

    @param run_metrics A list of metric dicts for a single simulation run.
    @return The integer stage index of first disconnection, or None if connectivity never drops.
    """
    for i, m in enumerate(run_metrics):
        if not m["connectivity"]:
            return i
    return None


def _largest_cluster(cluster_sizes_and_counts):
    """
    Function that extracts the size of the largest connected component from a cluster summary list.

    @param cluster_sizes_and_counts A list of (size, count) tuples as produced by GraphMetricCollector.
    @return The size of the largest cluster as an integer.
    """
    if not cluster_sizes_and_counts:
        return 0
    return max(size for size, _ in cluster_sizes_and_counts)


def _total_cluster_count(cluster_sizes_and_counts):
    """
    Function that sums the total number of connected components from a cluster summary list.

    @param cluster_sizes_and_counts A list of (size, count) tuples as produced by GraphMetricCollector.
    @return The total number of clusters as an integer.
    """
    return sum(count for _, count in cluster_sizes_and_counts)


def _finite_points(stages, values):
    """
    Function that filters out infinite metric values, returning only the finite stage/value pairs.

    @param stages A list of stage indices.
    @param values A list of metric values corresponding to each stage.
    @return A tuple (finite_stages, finite_values) with infinite entries removed.
    """
    finite_stages = [s for s, v in zip(stages, values) if v != float("inf")]
    finite_values = [v for v in values if v != float("inf")]
    return finite_stages, finite_values



class SimulationVisualiser:
    """
    Class that produces matplotlib plots for metrics collected across multiple simulation runs.
    """

    def __init__(self, simulation_runs, labels=None):
        """
        Constructor for SimulationVisualiser.

        @param simulation_runs List of runs, where each run is a list of metric dicts
                               as returned by GraphMetricCollector.collect_metrics().
        @param labels Optional list of display labels, one per run.
        """
        self.runs = simulation_runs
        n = len(simulation_runs)
        self.labels = labels if labels is not None else [f"Run {i + 1}" for i in range(n)]
        self._colours = [cm.tab10(i / max(n, 1)) for i in range(n)]

    def _style_axis(self, ax, title, ylabel):
        """
        Method that applies standard styling (title, axis labels, grid, background) to a matplotlib Axes.

        @param ax The matplotlib Axes object to style.
        @param title The chart title string.
        @param ylabel The y-axis label string.
        """
        ax.set_title(title)
        ax.set_xlabel("Random Edge Failure Stage")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
        ax.set_facecolor("#fbfbfd")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    def _add_legend(self, ax):
        """
        Method that adds a legend to a matplotlib Axes if labelled series are present.

        @param ax The matplotlib Axes object to add the legend to.
        """
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend(fontsize="small", frameon=True, loc="best")



    def plot_connectivity(self, ax=None, title="Network Connectivity over Failure Stages"):
        """
        Method that plots a binary connectivity series (1 = connected, 0 = disconnected) for each run,
        with a red marker at the first stage where connectivity collapses.

        @param ax Optional matplotlib Axes to draw on; a new figure is created if None.
        @param title Chart title string.
        @return The matplotlib Axes containing the plot.
        """
        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=(9, 4))

        for run, colour, label in zip(self.runs, self._colours, self.labels):
            stages = list(range(len(run)))
            values = [1 if m["connectivity"] else 0 for m in run]
            ax.plot(stages, values, color=colour, label=label, linewidth=1.8, alpha=0.9)

            collapse = _collapse_stage(run)
            if collapse is not None:
                ax.scatter(collapse, 0, color="red", zorder=5, s=70)
                ax.axvline(collapse, color="red", linestyle=":", linewidth=0.9, alpha=0.35)

        self._style_axis(ax, title, "Connected  (1 = yes,  0 = no)")
        ax.set_yticks([0, 1])
        self._add_legend(ax)

        if standalone:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_average_shortest_path(self, ax=None,
                                    title="Average Shortest Path over Failure Stages"):
        """
        Method that plots the average shortest path length for each run, omitting infinite values
        that arise when the graph is disconnected.

        @param ax Optional matplotlib Axes to draw on; a new figure is created if None.
        @param title Chart title string.
        @return The matplotlib Axes containing the plot.
        """
        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=(9, 4))

        for run, colour, label in zip(self.runs, self._colours, self.labels):
            stages = list(range(len(run)))
            values = [m["average_shortest_path"] for m in run]
            finite_stages, finite_values = _finite_points(stages, values)
            if finite_stages:
                ax.plot(finite_stages, finite_values, color=colour, label=label,
                        linewidth=1.8, alpha=0.9)

        self._style_axis(ax, title, "Average Shortest Path Length")
        self._add_legend(ax)

        if standalone:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_average_degree(self, ax=None, title="Average Degree over Failure Stages"):
        """
        Method that plots the average node degree for each run across all failure stages.

        @param ax Optional matplotlib Axes to draw on; a new figure is created if None.
        @param title Chart title string.
        @return The matplotlib Axes containing the plot.
        """
        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=(9, 4))

        for run, colour, label in zip(self.runs, self._colours, self.labels):
            stages = list(range(len(run)))
            values = [m["average_degree"] for m in run]
            ax.plot(stages, values, color=colour, label=label, linewidth=1.8, alpha=0.9)

        self._style_axis(ax, title, "Average Degree")
        self._add_legend(ax)

        if standalone:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_largest_cluster_size(self, ax=None,
                                   title="Largest Cluster Size over Failure Stages"):
        """
        Method that plots the size of the largest connected component for each run,
        with a red marker at the first connectivity collapse stage.

        @param ax Optional matplotlib Axes to draw on; a new figure is created if None.
        @param title Chart title string.
        @return The matplotlib Axes containing the plot.
        """
        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=(9, 4))

        for run, colour, label in zip(self.runs, self._colours, self.labels):
            stages = list(range(len(run)))
            values = [_largest_cluster(m["cluster_sizes_and_counts"]) for m in run]
            ax.plot(stages, values, color=colour, label=label, linewidth=1.8, alpha=0.9)

            collapse = _collapse_stage(run)
            if collapse is not None:
                ax.scatter(collapse, values[collapse], color="red", zorder=5, s=70)
                ax.axvline(collapse, color="red", linestyle=":", linewidth=0.9, alpha=0.35)

        self._style_axis(ax, title, "Largest Cluster Size (nodes)")
        self._add_legend(ax)

        if standalone:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_cluster_count(self, ax=None, title="Number of Clusters over Failure Stages"):
        """
        Method that plots the total number of connected components for each run across all failure stages.

        @param ax Optional matplotlib Axes to draw on; a new figure is created if None.
        @param title Chart title string.
        @return The matplotlib Axes containing the plot.
        """
        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=(9, 4))

        for run, colour, label in zip(self.runs, self._colours, self.labels):
            stages = list(range(len(run)))
            values = [_total_cluster_count(m["cluster_sizes_and_counts"]) for m in run]
            ax.plot(stages, values, color=colour, label=label, linewidth=1.8, alpha=0.9)

        self._style_axis(ax, title, "Number of Clusters")
        self._add_legend(ax)

        if standalone:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_diameter(self, ax=None, title="Network Diameter over Failure Stages"):
        """
        Method that plots the network diameter for each run, omitting infinite values
        that arise when the graph is disconnected.

        @param ax Optional matplotlib Axes to draw on; a new figure is created if None.
        @param title Chart title string.
        @return The matplotlib Axes containing the plot.
        """
        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=(9, 4))

        for run, colour, label in zip(self.runs, self._colours, self.labels):
            stages = list(range(len(run)))
            values = [m.get("diameter", float("inf")) for m in run]
            finite_stages, finite_values = _finite_points(stages, values)
            if finite_stages:
                ax.plot(finite_stages, finite_values, color=colour, label=label,
                        linewidth=1.8, alpha=0.9)

        self._style_axis(ax, title, "Diameter")
        self._add_legend(ax)

        if standalone:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_largest_component_diameter(self, ax=None,
                                         title="Largest-Component Diameter over Failure Stages"):
        """
        Method that plots the diameter of the largest connected component for each run across all stages.

        @param ax Optional matplotlib Axes to draw on; a new figure is created if None.
        @param title Chart title string.
        @return The matplotlib Axes containing the plot.
        """
        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=(9, 4))

        for run, colour, label in zip(self.runs, self._colours, self.labels):
            stages = list(range(len(run)))
            values = [m.get("largest_component_diameter", 0) for m in run]
            ax.plot(stages, values, color=colour, label=label, linewidth=1.8, alpha=0.9)

        self._style_axis(ax, title, "Largest-Component Diameter")
        self._add_legend(ax)

        if standalone:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_clustering(self, ax=None, title="Average Clustering Coefficient over Failure Stages"):
        """
        Method that plots the average clustering coefficient for each run across all failure stages.

        @param ax Optional matplotlib Axes to draw on; a new figure is created if None.
        @param title Chart title string.
        @return The matplotlib Axes containing the plot.
        """
        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=(9, 4))

        for run, colour, label in zip(self.runs, self._colours, self.labels):
            stages = list(range(len(run)))
            values = [m.get("average_clustering", 0.0) for m in run]
            ax.plot(stages, values, color=colour, label=label, linewidth=1.8, alpha=0.9)

        self._style_axis(ax, title, "Average Clustering Coefficient")
        self._add_legend(ax)

        if standalone:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_degree_distribution(self, stage=-1, ax=None, title=None):
        """
        Method that plots the degree distribution as a bar chart at a specific failure stage.

        @param stage Stage index to visualise; -1 selects the final recorded stage.
        @param ax Optional matplotlib Axes to draw on; a new figure is created if None.
        @param title Optional chart title; a default is generated from the stage index if None.
        @return The matplotlib Axes containing the plot.
        """
        standalone = ax is None
        if standalone:
            _, ax = plt.subplots(figsize=(9, 4))

        if title is None and stage == -1:
            title = "Degree Distribution at Final Failure Stage"
        if title is None:
            title = f"Degree Distribution at Stage {stage}"

        for run, colour, label in zip(self.runs, self._colours, self.labels):
            if not run:
                continue

            stage_index = stage if stage >= 0 else len(run) + stage
            if stage_index < 0 or stage_index >= len(run):
                continue
            dist = run[stage_index]["degree_distribution"]
            degrees = sorted(dist.keys())
            counts = [dist[d] for d in degrees]
            ax.bar(
                degrees,
                counts,
                color=colour,
                label=label,
                alpha=0.65,
                width=0.85,
                edgecolor="white",
                linewidth=0.4,
            )

        ax.set_xlabel("Degree")
        ax.set_ylabel("Node Count")
        ax.set_title(title)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
        ax.set_facecolor("#fbfbfd")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self._add_legend(ax)

        if standalone:
            plt.tight_layout()
            plt.show()
        return ax

    def plot_all(self, stage_for_degree_dist=-1):
        """
        Method that renders all metric panels in a single combined dashboard figure.

        @param stage_for_degree_dist Stage index used for the degree distribution panel; -1 = final stage.
        """
        fig, axes = plt.subplots(5, 2, figsize=(15, 19))
        fig.suptitle("Network Random Edge Failure Simulation — All Metrics",
                     fontsize=13, fontweight="bold")

        self.plot_connectivity(ax=axes[0, 0])
        self.plot_average_shortest_path(ax=axes[0, 1])
        self.plot_average_degree(ax=axes[1, 0])
        self.plot_largest_cluster_size(ax=axes[1, 1])
        self.plot_cluster_count(ax=axes[2, 0])
        self.plot_diameter(ax=axes[2, 1])
        self.plot_largest_component_diameter(ax=axes[3, 0])
        self.plot_clustering(ax=axes[3, 1])
        self.plot_degree_distribution(stage=stage_for_degree_dist, ax=axes[4, 0])
        axes[4, 1].set_visible(False)

        plt.tight_layout()
        plt.show()
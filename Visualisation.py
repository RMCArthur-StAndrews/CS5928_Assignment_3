import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _collapse_stage(run_metrics):
    """Return the first stage index where connectivity drops to False, or None."""
    for i, m in enumerate(run_metrics):
        if not m["connectivity"]:
            return i
    return None


def _largest_cluster(cluster_sizes_and_counts):
    """Return the size of the largest cluster from a cluster_sizes_and_counts list."""
    if not cluster_sizes_and_counts:
        return 0
    return max(size for size, _ in cluster_sizes_and_counts)


def _total_cluster_count(cluster_sizes_and_counts):
    """Return the total number of clusters."""
    return sum(count for _, count in cluster_sizes_and_counts)


def _finite_points(stages, values):
    """Return stages/values pairs where metric is finite."""
    finite_stages = [s for s, v in zip(stages, values) if v != float("inf")]
    finite_values = [v for v in values if v != float("inf")]
    return finite_stages, finite_values



class SimulationVisualiser:
    """
    Visualises metrics from multiple Progressive Random Failure simulation runs.

    :param simulation_runs: List of runs, where each run is a list of metric dicts
                            as returned by GraphMetricCollector.collect_metrics()
                            (i.e. the metrics_at_stages list from apply_overall_failure).
    :param labels: Optional list of display labels, one per run.
    """

    def __init__(self, simulation_runs, labels=None):
        self.runs = simulation_runs
        n = len(simulation_runs)
        self.labels = labels if labels is not None else [f"Run {i + 1}" for i in range(n)]
        self._colours = [cm.tab10(i / max(n, 1)) for i in range(n)]

    def _style_axis(self, ax, title, ylabel):
        ax.set_title(title)
        ax.set_xlabel("Random Edge Failure Stage")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
        ax.set_facecolor("#fbfbfd")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    def _add_legend(self, ax):
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend(fontsize="small", frameon=True, loc="best")



    def plot_connectivity(self, ax=None, title="Network Connectivity over Failure Stages"):
        """
        Plot connectivity (1 = connected, 0 = disconnected) for each run.
        A red dot marks the first stage where connectivity collapses.
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
        Plot average shortest path length for each run.
        Infinite values (graph disconnected) are omitted.
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
        """Plot average node degree for each run across all failure stages."""
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
        Plot the size of the largest connected component for each run.
        A red dot marks the stage where connectivity collapses.
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
        """Plot the total number of connected components for each run."""
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
        Plot network diameter for each run.
        Infinite values (graph disconnected) are omitted.
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
        """Plot largest-component diameter for each run across all failure stages."""
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

    def plot_degree_distribution(self, stage=-1, ax=None, title=None):
        """
        Plot the degree distribution at a specific failure stage for each run.

        :param stage: Stage index to visualise (default -1 = final stage).
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

    # -----------------------------------------------------------------------
    # Combined dashboard
    # -----------------------------------------------------------------------

    def plot_all(self, stage_for_degree_dist=-1):
        """
        Render all metric panels in a single figure.

        :param stage_for_degree_dist: Stage index used for the degree distribution panel.
        """
        fig, axes = plt.subplots(4, 2, figsize=(15, 15))
        fig.suptitle("Network Random Edge Failure Simulation — All Metrics",
                     fontsize=13, fontweight="bold")

        self.plot_connectivity(ax=axes[0, 0])
        self.plot_average_shortest_path(ax=axes[0, 1])
        self.plot_average_degree(ax=axes[1, 0])
        self.plot_largest_cluster_size(ax=axes[1, 1])
        self.plot_cluster_count(ax=axes[2, 0])
        self.plot_diameter(ax=axes[2, 1])
        self.plot_largest_component_diameter(ax=axes[3, 0])
        self.plot_degree_distribution(stage=stage_for_degree_dist, ax=axes[3, 1])

        plt.tight_layout()
        plt.show()
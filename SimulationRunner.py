from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import gc
import json
from pathlib import Path
from threading import Event
from typing import Callable, Dict, List, Any

import RandomFailure as rf


def _copy_graph(graph):
    return graph.copy()


def _run_single_process(graph_factory, failure_rate, max_stages):
    graph = graph_factory()
    failure_runner = rf.ProgressiveRandomFailure(
        graph, failure_rate, max_stages=max_stages
    )
    _, metrics = failure_runner.apply_overall_failure(
        keep_graph_snapshots=False,
    )
    return metrics


class SimulationRunner:
    """
    Class that orchestrates repeated random-failure simulation runs across one or more network topologies.
    """

    FULL_GC_INTERVAL = 5

    def export_results(self, runs: List[list], output_path: str,
                       typology: str, parameters: Dict[str, Any]) -> None:
        """
        Write completed simulation runs and their parameters to a JSON file.

        @param runs Simulation results returned by run_experiment.
        @param output_path Destination JSON path.
        @param typology Display name for the network architecture.
        @param parameters Parameters used to generate and degrade the networks.
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "typology": typology,
            "parameters": parameters,
            "num_runs": len(runs),
            "runs": runs,
        }
        output_file.write_text(json.dumps(payload), encoding="utf-8")

    def _copy_graph(self, graph):
        """
        Method that returns a shallow copy of a template graph for use in one simulation run.

        @param graph The NetworkX graph to copy.
        @return A new NetworkX graph with the same nodes and edges.
        """
        return _copy_graph(graph)

    def _collect_after_run(self, run_index: int):
        """
        Method that triggers garbage collection after each run, with a full collection every FULL_GC_INTERVAL runs.

        @param run_index The 1-based index of the run that just completed.
        """
        gc.collect(0)
        if run_index % self.FULL_GC_INTERVAL == 0:
            gc.collect()

    def _run_single(self, graph_factory: Callable[[], Any], failure_rate: float,
                    max_stages: int = None, stop_event: Event = None):
        """
        Method that executes a single simulation run: builds the graph, applies progressive failure,
        and returns the collected metrics.

        @param graph_factory Callable that returns a fresh NetworkX graph for this run.
        @param failure_rate Edge removal probability applied at each degradation stage.
        @param max_stages Optional cap on the number of stages before the run is terminated.
        @param stop_event Optional threading Event; if set, the run is aborted early.
        @return A list of metric dicts, one per failure stage including the initial state.
        """
        if stop_event and stop_event.is_set():
            return []

        graph = None
        failure_runner = None
        try:
            graph = graph_factory()
            failure_runner = rf.ProgressiveRandomFailure(
                graph, failure_rate, max_stages=max_stages
            )
            _, metrics = failure_runner.apply_overall_failure(
                should_stop=(lambda: stop_event.is_set()) if stop_event else None,
                keep_graph_snapshots=False,
            )
            return metrics
        finally:
            del failure_runner
            del graph

    def _shutdown_executor_on_caller_stop(self, executor: ThreadPoolExecutor, futures: List[Any]):
        """
        Method that performs best-effort cancellation of pending futures when the caller terminates early.

        @param executor The ThreadPoolExecutor whose futures should be cancelled.
        @param futures The list of Future objects to cancel.
        """
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)

    def run_experiment(self, graph_factory: Callable[[], Any], num_runs: int,
                       failure_rate: float, max_stages: int = None,
                       run_parallel: bool = False, max_workers: int = None,
                       stop_event: Event = None) -> List[list]:
        """
        Method that runs a complete experiment for a single topology by executing num_runs independent simulations.

        @param graph_factory Callable that returns a fresh NetworkX graph for each run.
        @param num_runs Number of independent simulation runs to perform.
        @param failure_rate Edge removal probability applied at each degradation stage.
        @param max_stages Optional cap on the number of stages per run.
        @param run_parallel Whether to execute runs concurrently using worker processes.
        @param max_workers Optional limit on the number of worker processes when run_parallel is True.
        @param stop_event Optional threading Event for sequential or caller-level termination.
        @return A list of runs, where each run is a list of metric dicts.
        """
        if num_runs <= 0:
            return []

        if not run_parallel:
            runs = []
            for run_index in range(1, num_runs + 1):
                if stop_event and stop_event.is_set():
                    break
                runs.append(
                    self._run_single(
                        graph_factory,
                        failure_rate,
                        max_stages,
                        stop_event=stop_event,
                    )
                )
                print(f"Completed run {run_index}/{num_runs}")
                self._collect_after_run(run_index)
            return runs

        executor = ProcessPoolExecutor(max_workers=max_workers)
        futures = []
        caller_stopped = False
        try:
            futures = [
                executor.submit(
                    _run_single_process,
                    graph_factory,
                    failure_rate,
                    max_stages,
                )
                for _ in range(num_runs)
            ]
            runs = []
            for run_index, future in enumerate(futures, start=1):
                runs.append(future.result())
                self._collect_after_run(run_index)
            return runs
        except (KeyboardInterrupt, SystemExit):
            caller_stopped = True
            self._shutdown_executor_on_caller_stop(executor, futures)
            raise
        finally:
            if not caller_stopped:
                executor.shutdown(wait=True, cancel_futures=False)

    def run_experiment_from_template(self, template_graph: Any, num_runs: int,
                                      failure_rate: float, max_stages: int = None,
                                      run_parallel: bool = False,
                                      max_workers: int = None,
                                      stop_event: Event = None) -> List[list]:
        """
        Method that runs repeated simulations by copying a pre-built template graph for each run,
        avoiding the cost of regenerating a deterministic topology on every iteration.

        @param template_graph A pre-built NetworkX graph used as the starting point for every run.
        @param num_runs Number of independent simulation runs to perform.
        @param failure_rate Edge removal probability applied at each degradation stage.
        @param max_stages Optional cap on the number of stages per run.
        @param run_parallel Whether to execute runs concurrently using a thread pool.
        @param max_workers Optional limit on the number of worker threads when run_parallel is True.
        @param stop_event Optional threading Event that signals early termination.
        @return A list of runs, where each run is a list of metric dicts.
        """
        return self.run_experiment(
            graph_factory=partial(_copy_graph, template_graph),
            num_runs=num_runs,
            failure_rate=failure_rate,
            max_stages=max_stages,
            run_parallel=run_parallel,
            max_workers=max_workers,
            stop_event=stop_event,
        )

    def run_experiments_parallel(self, experiments: Dict[str, Callable[[], Any]], num_runs: int,
                                 failure_rate: float, max_stages: int = None,
                                 max_workers: int = None) -> Dict[str, List[list]]:
        """
        Method that runs multiple named experiments concurrently, one thread per experiment.

        @param experiments Mapping of experiment name to a graph factory callable.
        @param num_runs Number of runs per experiment.
        @param failure_rate Edge removal probability applied at each degradation stage.
        @param max_stages Optional cap on the number of stages per run.
        @param max_workers Optional limit on the number of experiment-level worker threads.
        @return A dict mapping each experiment name to its list of simulation run results.
        """
        if not experiments:
            return {}

        executor = ThreadPoolExecutor(max_workers=max_workers)
        futures = {}
        caller_stopped = False
        stop_event = Event()
        try:
            futures = {
                name: executor.submit(
                    self.run_experiment,
                    factory,
                    num_runs,
                    failure_rate,
                    max_stages,
                    False,
                    None,
                    stop_event,
                )
                for name, factory in experiments.items()
            }
            return {name: future.result() for name, future in futures.items()}
        except (KeyboardInterrupt, SystemExit):
            caller_stopped = True
            stop_event.set()
            self._shutdown_executor_on_caller_stop(executor, list(futures.values()))
            raise
        finally:
            if not caller_stopped:
                executor.shutdown(wait=True, cancel_futures=False)

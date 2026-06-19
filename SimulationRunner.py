from concurrent.futures import ThreadPoolExecutor
import gc
from threading import Event
from typing import Callable, Dict, List, Any

import RandomFailure as rf


class SimulationRunner:
    """Runs random-failure simulations for one or more network topologies."""

    FULL_GC_INTERVAL = 5

    def _copy_graph(self, graph):
        """Return a fresh copy of a template graph for one simulation run."""
        return graph.copy()

    def _collect_after_run(self, run_index: int):
        """Collect young-generation garbage every run; full GC periodically."""
        gc.collect(0)
        if run_index % self.FULL_GC_INTERVAL == 0:
            gc.collect()

    def _run_single(self, graph_factory: Callable[[], Any], failure_rate: float,
                    max_stages: int = None, stop_event: Event = None):
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
        """Best-effort cancellation path when the caller terminates early."""
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)

    def run_experiment(self, graph_factory: Callable[[], Any], num_runs: int,
                       failure_rate: float, max_stages: int = None,
                       run_parallel: bool = False, max_workers: int = None,
                       stop_event: Event = None) -> List[list]:
        """Run one topology experiment for num_runs simulations.

        :param graph_factory: Callable that returns a fresh NetworkX graph.
        :param num_runs: Number of independent simulation runs.
        :param failure_rate: Edge failure probability per stage.
        :param max_stages: Optional cap for maximum degradation stages.
        :param run_parallel: Whether to parallelize runs with threads.
        :param max_workers: Optional worker cap when run_parallel=True.
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
                self._collect_after_run(run_index)
            return runs

        executor = ThreadPoolExecutor(max_workers=max_workers)
        futures = []
        caller_stopped = False
        local_stop_event = stop_event or Event()
        try:
            futures = [
                executor.submit(
                    self._run_single,
                    graph_factory,
                    failure_rate,
                    max_stages,
                    local_stop_event,
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
            local_stop_event.set()
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
        """Run repeated simulations from a pre-built graph template.

        This avoids regenerating an expensive topology for every run. Each
        simulation starts from a copy of the provided template graph.
        """
        return self.run_experiment(
            graph_factory=lambda: self._copy_graph(template_graph),
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
        """Run each named experiment on its own thread.

        :param experiments: Mapping of experiment name to graph factory callable.
        :param num_runs: Number of runs per experiment.
        :param failure_rate: Edge failure probability per stage.
        :param max_stages: Optional cap for maximum degradation stages.
        :param max_workers: Optional worker cap for experiment-level threads.
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

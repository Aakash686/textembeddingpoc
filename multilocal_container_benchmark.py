import argparse
import logging
import time
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import docker
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


@dataclass
class ContainerStats:
    name: str
    cpu_history: List[float] = field(default_factory=list)
    ram_history: List[float] = field(default_factory=list)

    @property
    def avg_cpu(self) -> float:
        return statistics.mean(self.cpu_history) if self.cpu_history else 0.0

    @property
    def max_cpu(self) -> float:
        return max(self.cpu_history) if self.cpu_history else 0.0

    @property
    def avg_ram(self) -> float:
        return statistics.mean(self.ram_history) if self.ram_history else 0.0


@dataclass
class BenchmarkResult:
    successful_requests: int
    failed_requests: int
    total_time: float
    rps: float
    avg_latency: float
    p95_latency: float
    p99_latency: float
    container_stats: Dict[str, ContainerStats]


class SingleContainerMonitor(threading.Thread):
    """
    Monitors a single container's stream in a dedicated thread.
    """

    def __init__(self, container_name: str, stats_object: ContainerStats):
        super().__init__()
        self.container_name = container_name
        self.stats = stats_object
        self.client = docker.from_env()
        self._stop_event = threading.Event()
        self.daemon = True  # Daemon thread ensures it dies if main program crashes

    def run(self):
        try:
            container = self.client.containers.get(self.container_name)
            stats_stream = container.stats(stream=True, decode=True)

            for stat in stats_stream:
                if self._stop_event.is_set():
                    break
                self._process_stat(stat)
        except docker.errors.NotFound:
            logger.error(f"Container {self.container_name} not found!")
        except Exception as e:
            logger.error(f"Monitor error for {self.container_name}: {e}")

    def _process_stat(self, d: dict):
        # CPU Calculation
        try:
            cpu_count = d["cpu_stats"]["online_cpus"]
            cpu_delta = d["cpu_stats"]["cpu_usage"]["total_usage"] - \
                        d["precpu_stats"]["cpu_usage"]["total_usage"]
            system_delta = d["cpu_stats"]["system_cpu_usage"] - \
                           d["precpu_stats"]["system_cpu_usage"]

            cpu_pct = 0.0
            if system_delta > 0.0 and cpu_delta > 0.0:
                cpu_pct = (cpu_delta / system_delta) * cpu_count * 100.0

            # RAM Calculation (MB)
            mem_usage = d["memory_stats"].get("usage", 0) / (1024 * 1024)

            self.stats.cpu_history.append(cpu_pct)
            self.stats.ram_history.append(mem_usage)
        except KeyError:
            pass

    def stop(self):
        self._stop_event.set()


class ClusterMonitor:
    """
    Orchestrates monitoring for a list of containers.
    """

    def __init__(self, container_names: List[str]):
        self.monitors = []
        self.stats_map = {}

        for name in container_names:
            stats = ContainerStats(name=name)
            self.stats_map[name] = stats
            monitor = SingleContainerMonitor(name, stats)
            self.monitors.append(monitor)

    def start(self):
        logger.info(f"Starting monitors for cluster: {list(self.stats_map.keys())}")
        for m in self.monitors:
            m.start()

    def stop(self):
        for m in self.monitors:
            m.stop()
        for m in self.monitors:
            m.join(timeout=2.0)


class LoadTester:
    def __init__(self, url: str, concurrency: int, total_requests: int, containers: List[str]):
        self.url = url
        self.concurrency = concurrency
        self.total_requests = total_requests
        self.monitor = ClusterMonitor(containers)

        self.payloads = [
            "Distributed systems require robust observability.",
            "Horizontal scaling distributes load across nodes.",
            "The quick brown fox jumps over the lazy dog.",
            "Nginx is a high-performance load balancer.",
            "Docker Compose simplifies multi-container orchestration."
        ]

    def _send_request(self) -> Optional[float]:
        import random
        payload = {"inputs": random.choice(self.payloads)}
        try:
            start = time.perf_counter()
            # Timeout slightly increased for load spikes
            resp = requests.post(self.url, json=payload, timeout=10)
            resp.raise_for_status()
            return (time.perf_counter() - start) * 1000
        except Exception:
            return None

    def run(self) -> BenchmarkResult:
        logger.info(f"Starting Load Test: {self.total_requests} reqs @ {self.concurrency} threads")

        self.monitor.start()
        latencies = []
        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            futures = [executor.submit(self._send_request) for _ in range(self.total_requests)]

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    latencies.append(result)

        total_time = time.perf_counter() - start_time
        self.monitor.stop()

        successful = len(latencies)
        failed = self.total_requests - successful

        if not latencies:
            logger.error("No successful requests recorded.")
            return None

        return BenchmarkResult(
            successful_requests=successful,
            failed_requests=failed,
            total_time=total_time,
            rps=successful / total_time,
            avg_latency=statistics.mean(latencies),
            p95_latency=statistics.quantiles(latencies, n=20)[18],
            p99_latency=statistics.quantiles(latencies, n=100)[98],
            container_stats=self.monitor.stats_map
        )


def print_report(res: BenchmarkResult):
    print("\n" + "=" * 60)
    print(f"🚀 CLUSTER BENCHMARK REPORT")
    print("=" * 60)
    print(f"Requests:     {res.successful_requests} ok / {res.failed_requests} failed")
    print(f"Duration:     {res.total_time:.2f}s")
    print(f"Throughput:   {res.rps:.2f} req/s")
    print("-" * 60)
    print(f"Latency (Avg): {res.avg_latency:.2f}ms")
    print(f"Latency (P95): {res.p95_latency:.2f}ms")
    print(f"Latency (P99): {res.p99_latency:.2f}ms")
    print("-" * 60)
    print(f"📦 CONTAINER RESOURCE USAGE")

    total_avg_cpu = 0.0

    for name, stats in res.container_stats.items():
        print(
            f"  🔹 {name:<10} | CPU: Avg {stats.avg_cpu:6.1f}% (Max {stats.max_cpu:6.1f}%) | RAM: {stats.avg_ram:.1f} MB")
        total_avg_cpu += stats.avg_cpu

    print("-" * 60)
    print(f"🔥 Cluster Total CPU Load: {total_avg_cpu:.1f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster Load Tester")
    parser.add_argument("--url", default="http://localhost:8080/embed", help="Load Balancer URL")
    parser.add_argument("--c", type=int, default=20, help="Concurrency level")
    parser.add_argument("--n", type=int, default=500, help="Total requests")
    # Allows passing multiple container names: --containers tei-1 tei-2
    parser.add_argument("--containers", nargs='+', default=["load_test-tei-1-1", "load_test-tei-2-1"], help="List of containers to monitor")

    args = parser.parse_args()

    tester = LoadTester(args.url, args.c, args.n, args.containers)
    try:
        result = tester.run()
        if result:
            print_report(result)
    except KeyboardInterrupt:
        logger.info("Benchmark stopped by user.")
    except Exception as e:
        logger.critical(f"Benchmark crashed: {e}")
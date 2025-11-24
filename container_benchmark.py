import argparse
import logging
import time
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional

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
class BenchmarkResult:
    successful_requests: int
    failed_requests: int
    total_time: float
    rps: float
    avg_latency: float
    p95_latency: float
    p99_latency: float
    avg_cpu_percent: float
    max_cpu_percent: float
    avg_ram_mb: float


class ContainerMonitor:
    """
    Monitors a specific Docker container's stats stream.
    """

    def __init__(self, container_name: str):
        self.container_name = container_name
        self.client = docker.from_env()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor_loop)

        self.cpu_stats: List[float] = []
        self.mem_stats: List[float] = []

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()

    def _calculate_cpu_percent(self, d: dict) -> float:
        """
        Calculates CPU usage % from Docker stats JSON (Linux/cgroup standard).
        """
        try:
            cpu_count = d["cpu_stats"]["online_cpus"]
            cpu_delta = d["cpu_stats"]["cpu_usage"]["total_usage"] - \
                        d["precpu_stats"]["cpu_usage"]["total_usage"]
            system_delta = d["cpu_stats"]["system_cpu_usage"] - \
                           d["precpu_stats"]["system_cpu_usage"]

            if system_delta > 0.0 and cpu_delta > 0.0:
                return (cpu_delta / system_delta) * cpu_count * 100.0
            return 0.0
        except KeyError:
            return 0.0

    def _monitor_loop(self):
        try:
            container = self.client.containers.get(self.container_name)
            # Stream stats (blocking call per update)
            stats_stream = container.stats(stream=True, decode=True)

            for stat in stats_stream:
                if self._stop_event.is_set():
                    break

                cpu_pct = self._calculate_cpu_percent(stat)
                mem_usage = stat["memory_stats"].get("usage", 0) / (1024 * 1024)  # Convert to MB

                self.cpu_stats.append(cpu_pct)
                self.mem_stats.append(mem_usage)
        except Exception as e:
            logger.error(f"Monitor failed: {e}")


class LoadTester:
    def __init__(self, url: str, concurrency: int, total_requests: int, container_name: str):
        self.url = url
        self.concurrency = concurrency
        self.total_requests = total_requests
        self.monitor = ContainerMonitor(container_name)

        # Realistic test data
        self.payloads = [
            "Optimization of vector search is critical for RAG pipelines.",
            "Distributed systems require eventual consistency checks.",
            "The quick brown fox jumps over the lazy dog.",
            "Latency constraints in high-frequency trading are strict.",
            "Deployment of LLMs on edge devices is a growing field."
        ]

    def _send_request(self) -> Optional[float]:
        import random
        payload = {"inputs": random.choice(self.payloads)}
        try:
            start = time.perf_counter()
            resp = requests.post(self.url, json=payload, timeout=5)
            resp.raise_for_status()
            return (time.perf_counter() - start) * 1000
        except Exception:
            return None

    def run(self) -> BenchmarkResult:
        logger.info(f"Starting benchmark: {self.total_requests} reqs @ {self.concurrency} concurrency")

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

        # Metrics calculation
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
            avg_cpu_percent=statistics.mean(self.monitor.cpu_stats) if self.monitor.cpu_stats else 0,
            max_cpu_percent=max(self.monitor.cpu_stats) if self.monitor.cpu_stats else 0,
            avg_ram_mb=statistics.mean(self.monitor.mem_stats) if self.monitor.mem_stats else 0
        )


def print_report(res: BenchmarkResult):
    print("\n" + "=" * 50)
    print(f"BENCHMARK REPORT")
    print("=" * 50)
    print(f"Requests:     {res.successful_requests} ok / {res.failed_requests} failed")
    print(f"Duration:     {res.total_time:.2f}s")
    print(f"Throughput:   {res.rps:.2f} req/s")
    print("-" * 50)
    print(f"Latency (Avg): {res.avg_latency:.2f}ms")
    print(f"Latency (P95): {res.p95_latency:.2f}ms")
    print(f"Latency (P99): {res.p99_latency:.2f}ms")
    print("-" * 50)
    print(f"Container CPU: Avg {res.avg_cpu_percent:.1f}% / Max {res.max_cpu_percent:.1f}%")
    print(f"Container RAM: Avg {res.avg_ram_mb:.1f} MB")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TEI Load Tester")
    parser.add_argument("--url", default="http://localhost:8080/embed", help="Endpoint URL")
    parser.add_argument("--c", type=int, default=10, help="Concurrency level")
    parser.add_argument("--n", type=int, default=500, help="Total requests")
    parser.add_argument("--container", default="hf-embedding", help="Docker container name")

    args = parser.parse_args()

    tester = LoadTester(args.url, args.c, args.n, args.container)
    try:
        result = tester.run()
        if result:
            print_report(result)
    except Exception as e:
        logger.critical(f"Benchmark crashed: {e}")
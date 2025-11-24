import requests
import time
import concurrent.futures
import statistics
import random
import psutil
import threading

# --- Configuration ---
API_URL = "http://localhost:8080/embed"
CONCURRENCY_LEVEL = 10  # Parallel requests
TOTAL_REQUESTS = 200  # Increased to ensure test lasts long enough to measure
SAMPLE_TEXTS = [
    "Machine learning engineering requires robust monitoring.",
    "Docker containers on M1 Macs use virtualization frameworks.",
    "Latency and throughput are trade-offs in system design.",
    "The quick brown fox jumps over the lazy dog.",
    "Embeddings are dense vector representations of text.",
    "Horizontal scaling is preferred over vertical scaling."
]


class ResourceMonitor(threading.Thread):
    """
    Monitors System CPU and RAM usage in a separate thread.
    """

    def __init__(self, interval=0.2):
        super().__init__()
        self.interval = interval
        self.stop_event = threading.Event()
        self.cpu_readings = []
        self.ram_readings = []

    def run(self):
        # Prime psutil (first call is always 0.0)
        psutil.cpu_percent(interval=None)

        while not self.stop_event.is_set():
            # Get system-wide CPU and RAM
            cpu = psutil.cpu_percent(interval=self.interval)
            ram = psutil.virtual_memory().percent

            self.cpu_readings.append(cpu)
            self.ram_readings.append(ram)

    def stop(self):
        self.stop_event.set()


def send_request(request_id):
    text = random.choice(SAMPLE_TEXTS)
    try:
        start = time.perf_counter()
        response = requests.post(API_URL, json={"inputs": text})
        response.raise_for_status()
        latency = (time.perf_counter() - start) * 1000
        return latency
    except Exception as e:
        return None


def run_load_test():
    print(f"🚀 Starting Load Test with Monitoring")
    print(f"   Requests: {TOTAL_REQUESTS} | Concurrency: {CONCURRENCY_LEVEL}")
    print("-" * 50)

    # 1. Start Monitoring
    monitor = ResourceMonitor()
    monitor.start()

    # 2. Run Load Test
    latencies = []
    start_test_time = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY_LEVEL) as executor:
        futures = [executor.submit(send_request, i) for i in range(TOTAL_REQUESTS)]

        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res: latencies.append(res)

    total_test_time = time.perf_counter() - start_test_time

    # 3. Stop Monitoring
    monitor.stop()
    monitor.join()

    # --- Results ---
    if not latencies:
        print("❌ All requests failed.")
        return

    rps = len(latencies) / total_test_time
    avg_lat = statistics.mean(latencies)
    p99_lat = statistics.quantiles(latencies, n=100)[98]

    # Resource Stats
    avg_cpu = statistics.mean(monitor.cpu_readings) if monitor.cpu_readings else 0
    max_cpu = max(monitor.cpu_readings) if monitor.cpu_readings else 0
    avg_ram = statistics.mean(monitor.ram_readings) if monitor.ram_readings else 0

    print(f"\n📊 --- Performance Results ---")
    print(f"⚡ Throughput:     {rps:.2f} req/sec")
    print(f"⏱️  Avg Latency:    {avg_lat:.2f} ms")
    print(f"⏱️  P99 Latency:    {p99_lat:.2f} ms")

    print(f"\n🖥️  --- System Resources (Host) ---")
    print(f"🔥 CPU Usage:      Avg {avg_cpu:.1f}% | Max {max_cpu:.1f}%")
    print(f"💾 RAM Usage:      Avg {avg_ram:.1f}%")
    print("-" * 50)


if __name__ == "__main__":
    run_load_test()
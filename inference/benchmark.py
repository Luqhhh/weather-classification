"""
CPU Inference Benchmark

Measures inference performance on CPU:
- Per-image latency (mean, median, p95, p99)
- Throughput (images/second)
- Memory usage
- Total time estimate for the scoring set
"""

import logging
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch

from models.base import WeatherClassifier

logger = logging.getLogger(__name__)


class CpuBenchmark:
    """Benchmark CPU inference performance for weather classification models.

    Measures:
    - Latency statistics (mean, median, p95, p99)
    - Throughput (images/second)
    - Memory footprint (RAM)
    - Estimated total time for scoring set
    """

    def __init__(
        self,
        model: WeatherClassifier,
        input_size: int = 224,
        batch_sizes: Optional[List[int]] = None,
    ):
        """
        Args:
            model: Trained WeatherClassifier.
            input_size: Input image size.
            batch_sizes: List of batch sizes to test.
        """
        self.model = model.to("cpu")
        self.model.eval()
        self.input_size = input_size
        self.batch_sizes = batch_sizes or [1, 4, 8, 16, 32, 64]

    def run(self, num_warmup: int = 10, num_iterations: int = 100) -> Dict:
        """Run the full CPU benchmark.

        Returns:
            Dict with all benchmark results.
        """
        logger.info("Running CPU inference benchmark...")
        logger.info(f"Model: {self.model.backbone_name}, "
                     f"Params: {self.model.get_param_count()['total_millions']}M")

        results = {
            "model_name": self.model.backbone_name,
            "params_millions": self.model.get_param_count()["total_millions"],
            "weight_size_mb": round(self.model.get_weight_size_mb(), 1),
            "input_size": self.input_size,
            "batch_results": {},
            "memory_mb": None,
        }

        # Test each batch size
        for bs in self.batch_sizes:
            logger.info(f"  Benchmarking batch_size={bs}...")
            batch_result = self._benchmark_batch_size(bs, num_warmup, num_iterations)
            results["batch_results"][str(bs)] = batch_result

        # Measure memory
        results["memory_mb"] = self._measure_memory()

        # Find optimal batch size (best throughput within 70-min limit)
        optimal = self._find_optimal_batch(results)
        results["optimal_batch_size"] = optimal["batch_size"]
        results["optimal_throughput"] = optimal["throughput"]

        # Estimate for scoring set (assume 3000 images)
        scoring_estimate = self._estimate_scoring_time(
            results, num_scoring_images=3000
        )
        results["scoring_estimate"] = scoring_estimate

        self._log_results(results)
        return results

    def _benchmark_batch_size(
        self, batch_size: int, num_warmup: int, num_iterations: int
    ) -> Dict:
        """Benchmark a specific batch size."""
        dummy_input = torch.randn(batch_size, 3, self.input_size, self.input_size)

        # Warmup
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = self.model(dummy_input)

        # Timed iterations
        latencies_ms = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            with torch.no_grad():
                _ = self.model(dummy_input)
            latencies_ms.append((time.perf_counter() - start) * 1000)

        latencies = np.array(latencies_ms)
        per_image = latencies / batch_size

        return {
            "batch_size": batch_size,
            "batch_latency_mean_ms": round(float(np.mean(latencies)), 2),
            "batch_latency_median_ms": round(float(np.median(latencies)), 2),
            "batch_latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
            "batch_latency_p99_ms": round(float(np.percentile(latencies, 99)), 2),
            "per_image_mean_ms": round(float(np.mean(per_image)), 2),
            "throughput_imgs_per_sec": round(batch_size * 1000 / float(np.mean(latencies)), 1),
        }

    def _measure_memory(self) -> Dict:
        """Measure model memory footprint."""
        # Model parameter memory
        param_size_mb = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        ) / (1024 * 1024)

        # Process memory (RSS)
        process = psutil.Process()
        rss_mb = process.memory_info().rss / (1024 * 1024)

        return {
            "model_params_mb": round(param_size_mb, 1),
            "process_rss_mb": round(rss_mb, 1),
        }

    def _find_optimal_batch(self, results: Dict) -> Dict:
        """Find the batch size with best throughput that fits within constraints."""
        best = {"batch_size": 1, "throughput": 0}
        for bs_str, batch_result in results["batch_results"].items():
            throughput = batch_result["throughput_imgs_per_sec"]
            if throughput > best["throughput"]:
                best = {"batch_size": int(bs_str), "throughput": throughput}
        return best

    def _estimate_scoring_time(
        self,
        results: Dict,
        num_scoring_images: int = 3000,
    ) -> Dict:
        """Estimate total time to process the scoring set."""
        optimal_bs = results["optimal_batch_size"]
        best_result = results["batch_results"][str(optimal_bs)]
        per_image_ms = best_result["per_image_mean_ms"]

        # Model time
        total_model_time_sec = per_image_ms * num_scoring_images / 1000

        # I/O overhead estimate: ~5ms per image for disk read + preprocessing
        io_overhead_sec = num_scoring_images * 0.005

        total_sec = total_model_time_sec + io_overhead_sec

        return {
            "num_scoring_images": num_scoring_images,
            "estimated_model_time_min": round(total_model_time_sec / 60, 1),
            "estimated_io_time_min": round(io_overhead_sec / 60, 1),
            "estimated_total_time_min": round(total_sec / 60, 1),
            "within_70min": total_sec < 70 * 60,
            "margin_min": round(70 - total_sec / 60, 1),
        }

    def _log_results(self, results: Dict) -> None:
        """Log a summary of benchmark results."""
        se = results["scoring_estimate"]
        logger.info("=" * 50)
        logger.info("CPU Benchmark Summary")
        logger.info(f"  Model: {results['model_name']} "
                     f"({results['params_millions']}M params, "
                     f"{results['weight_size_mb']}MB)")
        logger.info(f"  Optimal batch size: {results['optimal_batch_size']}")
        logger.info(f"  Optimal throughput: {results['optimal_throughput']} imgs/s")
        logger.info(f"  Memory: {results['memory_mb']['process_rss_mb']} MB RSS")
        logger.info(f"  Scoring estimate ({se['num_scoring_images']} images):")
        logger.info(f"    Model time: {se['estimated_model_time_min']} min")
        logger.info(f"    I/O time:   {se['estimated_io_time_min']} min")
        logger.info(f"    Total:      {se['estimated_total_time_min']} min")
        logger.info(f"    Within 70min: {'✅ YES' if se['within_70min'] else '❌ NO — must optimize'}")
        logger.info("=" * 50)

    def save_csv(self, results: Dict, path: str) -> None:
        """Save benchmark results to CSV."""
        import csv
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "batch_size", "batch_latency_mean_ms", "batch_latency_p95_ms",
                "per_image_mean_ms", "throughput_imgs_per_sec",
            ])
            for bs_str, br in results["batch_results"].items():
                writer.writerow([
                    bs_str,
                    br["batch_latency_mean_ms"],
                    br["batch_latency_p95_ms"],
                    br["per_image_mean_ms"],
                    br["throughput_imgs_per_sec"],
                ])
        logger.info(f"Benchmark results saved to {path}")

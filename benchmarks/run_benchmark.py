#!/usr/bin/env python3
"""
Benchmark script for ldcov standard workflows.
Runs various ldcov workflows and captures performance metrics.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import psutil
import numpy as np
import tempfile
import shutil
import argparse

# Test configurations
TEST_CONFIGS = [
    # (samples, variants, compression, bits, name)
    (1000, 1000, "zlib", 8, "small"),
    (5000, 5000, "zlib", 8, "medium"),
    (10000, 10000, "zlib", 8, "large"),
    (100000, 10000, "zlib", 8, "extra_large"),
    (5000, 5000, "nocomp", 8, "medium_nocomp"),
    (10000, 5000, "zlib", 8, "wide"),
    (5000, 10000, "zlib", 8, "tall"),
]

WORKFLOWS = [
    "standard_workflow",  # The standard ldcov workflow with projection and z-file
    "precompute_projection",  # Need this to create projection matrix first
]


class BenchmarkRunner:
    def __init__(self, data_dir="/data/test_data", output_dir="/results", num_runs=3):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.num_runs = num_runs
        self.commit = os.environ.get("GIT_COMMIT", "unknown")
        self.process = psutil.Process()
        
        # Create temp directory for outputs
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def __del__(self):
        # Cleanup temp directory
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def get_system_info(self):
        """Get system information."""
        return {
            "cpu": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "platform": os.uname().sysname,
            "threads": int(os.environ.get("OMP_NUM_THREADS", 1)),
        }
    
    def create_synthetic_covariates(self, sample_file, output_file):
        """Create synthetic covariate file for testing."""
        # Read sample IDs from .sample file
        with open(sample_file, 'r') as f:
            lines = f.readlines()
            # Skip header lines
            sample_ids = [line.split()[0] for line in lines[2:]]
        
        # Create synthetic covariates (PC1-PC10 + age + sex)
        n_samples = len(sample_ids)
        
        with open(output_file, 'w') as f:
            # Header
            f.write("FID\tIID\tPC1\tPC2\tPC3\tPC4\tPC5\tPC6\tPC7\tPC8\tPC9\tPC10\tage\tsex\n")
            
            # Generate random covariates
            np.random.seed(42)  # For reproducibility
            for sample_id in sample_ids:
                pcs = np.random.randn(10) * 0.1  # Small values for PCs
                age = np.random.uniform(20, 80)
                sex = np.random.choice([0, 1])
                
                values = [sample_id, sample_id] + [f"{x:.6f}" for x in pcs] + [f"{age:.1f}", str(sex)]
                f.write("\t".join(values) + "\n")
    
    def create_z_file(self, num_variants, output_file, proportion=0.5):
        """Create a z-file with variant IDs for filtering."""
        # Generate variant IDs to include (select a proportion of variants)
        np.random.seed(42)
        selected_indices = np.random.choice(num_variants, int(num_variants * proportion), replace=False)
        selected_indices.sort()
        
        with open(output_file, 'w') as f:
            # Write header
            f.write("rsid\tchromosome\tposition\tallele1\tallele2\n")
            
            for idx in selected_indices:
                # Create variant info matching test data format
                # Test data uses rs1000000, rs1000001, etc. with positions 1000, 2000, etc.
                rsid = f"rs{1000000 + idx}"
                chrom = "chr1"  # Test data uses "chr1" not just "1"
                pos = (idx + 1) * 1000  # Positions are 1000, 2000, 3000, etc.
                allele1 = "A"
                allele2 = "G"
                f.write(f"{rsid}\t{chrom}\t{pos}\t{allele1}\t{allele2}\n")
    
    def measure_command(self, cmd, name):
        """Run command and measure performance."""
        metrics = {
            "name": name,
            "command": " ".join(cmd),
            "runs": []
        }
        
        for run in range(self.num_runs):
            # Clear caches (best effort)
            try:
                subprocess.run(["sync"], check=False)
                with open("/proc/sys/vm/drop_caches", "w") as f:
                    f.write("1")
            except:
                pass  # May not have permission
            
            # Record initial state
            self.process.cpu_percent()  # Initialize
            start_time = time.time()
            start_memory = self.process.memory_info().rss / (1024**2)  # MB
            
            # Run command
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=self.temp_dir
                )
                success = result.returncode == 0
                error = result.stderr if not success else None
            except Exception as e:
                success = False
                error = str(e)
            
            # Record final state
            end_time = time.time()
            peak_memory = self.process.memory_info().rss / (1024**2)  # MB
            cpu_percent = self.process.cpu_percent()
            
            run_metrics = {
                "run": run + 1,
                "success": success,
                "total_time": end_time - start_time,
                "peak_memory_mb": peak_memory,
                "memory_delta_mb": peak_memory - start_memory,
                "cpu_percent": cpu_percent,
            }
            
            if error:
                run_metrics["error"] = error
            
            metrics["runs"].append(run_metrics)
            
            # Sleep between runs
            time.sleep(2)
        
        # Calculate aggregate metrics
        if all(r["success"] for r in metrics["runs"]):
            times = [r["total_time"] for r in metrics["runs"]]
            memories = [r["peak_memory_mb"] for r in metrics["runs"]]
            
            metrics["aggregate"] = {
                "median_time": np.median(times),
                "mean_time": np.mean(times),
                "std_time": np.std(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
                "median_memory_mb": np.median(memories),
                "success_rate": 1.0,
            }
        else:
            metrics["aggregate"] = {
                "success_rate": sum(1 for r in metrics["runs"] if r["success"]) / len(metrics["runs"])
            }
        
        return metrics
    
    def run_precompute_projection(self, sample_file):
        """Run projection matrix precomputation."""
        # Create synthetic covariates
        cov_file = self.temp_dir / "covariates.txt"
        self.create_synthetic_covariates(sample_file, cov_file)
        
        output_prefix = self.temp_dir / "projection"
        cmd = [
            "ldcov",
            "--precompute-projection",
            "-c", str(cov_file),
            "--sample", str(sample_file),
            "--out", str(output_prefix)
        ]
        return self.measure_command(cmd, "precompute_projection")
    
    def run_standard_workflow(self, bgen_file, sample_file, num_variants):
        """Run the standard ldcov workflow with projection, z-file, and BCOR output."""
        # First ensure projection matrix exists
        proj_file = self.temp_dir / "projection.proj.npz"
        if not proj_file.exists():
            # Run projection precomputation
            proj_result = self.run_precompute_projection(sample_file)
            if not proj_file.exists():
                return {"name": "standard_workflow", "error": "Failed to create projection matrix"}
        
        # Create z-file for variant filtering (50% of variants)
        z_file = self.temp_dir / "variants.z"
        self.create_z_file(num_variants, z_file, proportion=0.5)
        
        # Get BGI file path
        bgi_file = Path(str(bgen_file) + ".bgi")
        
        output_prefix = self.temp_dir / "ld_standard"
        cmd = [
            "ldcov",
            "--bgen", str(bgen_file),
            "--bgi", str(bgi_file),
            "--sample", str(sample_file),
            "--projection", str(proj_file),
            "--z", str(z_file),
            "--compute-ld",
            "--output-format", "bcor",
            "--nan-action", "mean",
            "--out", str(output_prefix)
        ]
        return self.measure_command(cmd, "standard_workflow")
    
    def run_benchmark(self):
        """Run all benchmarks."""
        results = {
            "commit": self.commit,
            "timestamp": datetime.now().isoformat(),
            "system": self.get_system_info(),
            "benchmarks": []
        }
        
        # Read commit info if available
        commit_info_file = Path("/app/COMMIT_INFO.txt")
        if commit_info_file.exists():
            results["commit_info"] = commit_info_file.read_text().strip()
        
        for samples, variants, compression, bits, config_name in TEST_CONFIGS:
            print(f"\nRunning benchmarks for {config_name} ({samples}s × {variants}v {compression} {bits}bit)")
            
            # Find test files
            pattern = f"test_{samples}s_{variants}v_{compression}_{bits}bit"
            bgen_file = self.data_dir / f"{pattern}.bgen"
            sample_file = self.data_dir / f"{pattern}.sample"
            
            if not bgen_file.exists():
                print(f"  Skipping - {bgen_file} not found")
                continue
            
            config_results = {
                "config": config_name,
                "samples": samples,
                "variants": variants,
                "compression": compression,
                "bits": bits,
                "file": str(bgen_file.name),
                "file_size_mb": bgen_file.stat().st_size / (1024**2),
                "workflows": {}
            }
            
            # Run workflows
            for workflow in WORKFLOWS:
                print(f"  Running {workflow}...")
                
                if workflow == "precompute_projection":
                    result = self.run_precompute_projection(sample_file)
                elif workflow == "standard_workflow":
                    result = self.run_standard_workflow(bgen_file, sample_file, variants)
                
                config_results["workflows"][workflow] = result
                
                # Clean temp directory between workflows
                for f in self.temp_dir.glob("*"):
                    if f.is_file():
                        f.unlink()
            
            # Calculate derived metrics
            if "standard_workflow" in config_results["workflows"]:
                std_result = config_results["workflows"]["standard_workflow"]
                if "aggregate" in std_result and "median_time" in std_result["aggregate"]:
                    median_time = std_result["aggregate"]["median_time"]
                    # Note: using half the variants since z-file filters 50%
                    effective_variants = variants * 0.5
                    config_results["derived_metrics"] = {
                        "variants_per_second": effective_variants / median_time if median_time > 0 else 0,
                        "mb_per_second": config_results["file_size_mb"] / median_time if median_time > 0 else 0,
                        "effective_variants": effective_variants
                    }
            
            results["benchmarks"].append(config_results)
        
        return results
    
    def save_results(self, results):
        """Save results to JSON file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename with commit and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{self.commit[:8]}_{timestamp}.json"
        output_file = self.output_dir / filename
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        # Also save as latest for easy access
        latest_file = self.output_dir / f"benchmark_{self.commit[:8]}_latest.json"
        with open(latest_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return output_file


def main():
    parser = argparse.ArgumentParser(description="Run ldcov benchmarks")
    parser.add_argument("--data-dir", default="/data/test_data",
                        help="Directory containing test data")
    parser.add_argument("--output-dir", default="/results",
                        help="Directory for output results")
    parser.add_argument("--num-runs", type=int, default=3,
                        help="Number of runs per benchmark")
    args = parser.parse_args()
    
    print(f"Running ldcov benchmarks for commit: {os.environ.get('GIT_COMMIT', 'unknown')}")
    
    runner = BenchmarkRunner(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_runs=args.num_runs
    )
    
    results = runner.run_benchmark()
    output_file = runner.save_results(results)
    
    # Print summary
    print("\n=== Benchmark Summary ===")
    print(f"Commit: {results['commit']}")
    print(f"Timestamp: {results['timestamp']}")
    print(f"Total configurations: {len(results['benchmarks'])}")
    
    # Print performance summary
    for config in results['benchmarks']:
        print(f"\n{config['config']}:")
        for workflow, result in config['workflows'].items():
            if "aggregate" in result and "median_time" in result["aggregate"]:
                print(f"  {workflow}: {result['aggregate']['median_time']:.2f}s")


if __name__ == "__main__":
    main()
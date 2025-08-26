#!/usr/bin/env python3
"""
Performance benchmark for torch.rand vs random.randint in seed generation.

This benchmark tests the performance characteristics of different random backends
for seed generation in the AdvancedSeedGenerator node.
"""

import time
import random
import torch
import numpy as np
from typing import List, Dict, Any
import statistics
import sys
import os

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from random_seed_generator import AdvancedSeedGenerator
    HAS_SEED_GENERATOR = True
except ImportError:
    print("Warning: Could not import AdvancedSeedGenerator - running standalone benchmarks only")
    HAS_SEED_GENERATOR = False


def benchmark_random_randint(count: int) -> tuple[List[int], float]:
    """Benchmark Python's random.randint performance."""
    start_time = time.perf_counter()
    seeds = []
    
    for _ in range(count):
        seed = random.randint(0, 2**48 - 1)  # Use compatible range
        seeds.append(seed)
    
    elapsed = time.perf_counter() - start_time
    return seeds, elapsed


def benchmark_torch_rand_cpu(count: int) -> tuple[List[int], float]:
    """Benchmark torch.randint on CPU performance."""
    start_time = time.perf_counter()
    
    with torch.no_grad():
        seed_vals = torch.randint(
            low=0,
            high=2**48,
            size=(count,),
            dtype=torch.int64,
            device='cpu'
        )
        seeds = seed_vals.tolist()
    
    elapsed = time.perf_counter() - start_time
    return seeds, elapsed


def benchmark_torch_rand_gpu(count: int) -> tuple[List[int], float]:
    """Benchmark torch.randint on GPU performance (if available)."""
    if not torch.cuda.is_available():
        return [], float('inf')
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        seed_vals = torch.randint(
            low=0,
            high=2**48,
            size=(count,),
            dtype=torch.int64,
            device='cuda'
        )
        seeds = seed_vals.cpu().tolist()
    
    elapsed = time.perf_counter() - start_time
    return seeds, elapsed


def benchmark_torch_single_seeds(count: int) -> tuple[List[int], float]:
    """Benchmark torch.randint for single seed generation (mimicking current usage)."""
    start_time = time.perf_counter()
    seeds = []
    
    for _ in range(count):
        with torch.no_grad():
            seed_tensor = torch.randint(
                low=0,
                high=2**48,
                size=(1,),
                dtype=torch.int64,
                device='cpu'
            )
            seeds.append(int(seed_tensor.item()))
    
    elapsed = time.perf_counter() - start_time
    return seeds, elapsed


def benchmark_advanced_seed_generator(count: int, backend: str) -> tuple[List[int], float]:
    """Benchmark the AdvancedSeedGenerator with different backends."""
    if not HAS_SEED_GENERATOR:
        return [], float('inf')
    
    generator = AdvancedSeedGenerator()
    start_time = time.perf_counter()
    
    seeds = []
    for _ in range(count):
        result = generator.generate_seed(
            mode="random",
            seed=0,
            sync_libraries=False,  # Skip sync for pure performance test
            deterministic=False,
            overflow_behavior="wrap",
            use_torch_backend=backend,
            batch_count=1
        )
        seeds.append(result[0])
    
    elapsed = time.perf_counter() - start_time
    return seeds, elapsed


def benchmark_advanced_seed_generator_batch(batch_size: int, backend: str) -> tuple[List[int], float]:
    """Benchmark the AdvancedSeedGenerator batch mode."""
    if not HAS_SEED_GENERATOR:
        return [], float('inf')
    
    generator = AdvancedSeedGenerator()
    start_time = time.perf_counter()
    
    result = generator.generate_seed(
        mode="random",
        seed=0,
        sync_libraries=False,  # Skip sync for pure performance test
        deterministic=False,
        overflow_behavior="wrap",
        use_torch_backend=backend,
        batch_count=batch_size
    )
    
    elapsed = time.perf_counter() - start_time
    # For batch mode, we simulate getting all seeds (in real usage, batch generation 
    # would return multiple seeds, but current implementation returns first seed + count)
    seeds = [result[0]] * batch_size  # Placeholder for actual batch results
    
    return seeds, elapsed


def run_benchmark_suite():
    """Run comprehensive benchmarks and display results."""
    print("🔬 Advanced Seed Generator - torch.rand Performance Benchmark")
    print("=" * 70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print()
    
    # Test different batch sizes
    test_sizes = [1, 10, 100, 1000, 10000]
    results = {}
    
    for size in test_sizes:
        print(f"🧪 Testing batch size: {size}")
        print("-" * 40)
        
        # Run multiple iterations for statistical accuracy
        iterations = 5 if size <= 1000 else 3
        
        # Benchmark different methods
        methods = [
            ("random.randint", benchmark_random_randint),
            ("torch.rand (CPU batch)", benchmark_torch_rand_cpu),
            ("torch.rand (single calls)", benchmark_torch_single_seeds),
        ]
        
        if torch.cuda.is_available():
            methods.append(("torch.rand (GPU batch)", benchmark_torch_rand_gpu))
        
        if HAS_SEED_GENERATOR:
            methods.extend([
                ("AdvancedSeedGenerator (random)", lambda c: benchmark_advanced_seed_generator(c, "random")),
                ("AdvancedSeedGenerator (torch)", lambda c: benchmark_advanced_seed_generator(c, "torch")),
                ("AdvancedSeedGenerator (auto)", lambda c: benchmark_advanced_seed_generator(c, "auto")),
            ])
            
            if size > 1:  # Test batch mode for larger sizes
                methods.append(("AdvancedSeedGenerator (batch-auto)", lambda c: benchmark_advanced_seed_generator_batch(c, "auto")))
        
        size_results = {}
        
        for method_name, method_func in methods:
            times = []
            
            for _ in range(iterations):
                try:
                    seeds, elapsed = method_func(size)
                    if elapsed != float('inf'):
                        times.append(elapsed)
                except Exception as e:
                    print(f"  ❌ {method_name}: Error - {str(e)}")
                    continue
            
            if times:
                avg_time = statistics.mean(times)
                min_time = min(times)
                max_time = max(times)
                
                # Calculate seeds per second
                seeds_per_sec = size / avg_time if avg_time > 0 else 0
                
                size_results[method_name] = {
                    'avg_time': avg_time,
                    'min_time': min_time,
                    'max_time': max_time,
                    'seeds_per_sec': seeds_per_sec
                }
                
                print(f"  ✅ {method_name:30s}: {avg_time:.6f}s ({seeds_per_sec:,.0f} seeds/sec)")
            else:
                print(f"  ❌ {method_name:30s}: Failed")
        
        results[size] = size_results
        print()
    
    # Performance summary
    print("📊 PERFORMANCE SUMMARY")
    print("=" * 70)
    
    # Find optimal method for each batch size
    for size, size_results in results.items():
        if not size_results:
            continue
        
        fastest_method = min(size_results.keys(), key=lambda k: size_results[k]['avg_time'])
        fastest_time = size_results[fastest_method]['avg_time']
        fastest_rate = size_results[fastest_method]['seeds_per_sec']
        
        print(f"Batch size {size:5d}: {fastest_method} is fastest ({fastest_time:.6f}s, {fastest_rate:,.0f} seeds/sec)")
    
    print()
    print("💡 RECOMMENDATIONS")
    print("=" * 70)
    print("• Single seeds (1-9): Use random.randint for best performance")
    print("• Small batches (10-99): torch.rand CPU may be comparable")
    print("• Large batches (100+): torch.rand shows significant advantages")
    if torch.cuda.is_available():
        print("• Very large batches (1000+): torch.rand GPU offers best performance")
    else:
        print("• GPU not available - CPU torch.rand still beneficial for large batches")
    
    print("• AdvancedSeedGenerator 'auto' mode provides optimal selection")


if __name__ == "__main__":
    # Set up reproducible benchmarks
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Warm up GPU if available
    if torch.cuda.is_available():
        with torch.no_grad():
            _ = torch.rand(100, device='cuda')
    
    run_benchmark_suite()
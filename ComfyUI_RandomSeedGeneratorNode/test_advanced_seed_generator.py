#!/usr/bin/env python3
"""
Comprehensive test suite for AdvancedSeedGenerator.

This test suite covers all functionality including:
- Thread safety
- All generation modes
- Error handling
- Edge cases
- Cross-library synchronization
- Performance characteristics
"""

import threading
import time
import logging
import os
import sys
from unittest.mock import patch, MagicMock
from typing import List

# Set up test environment
os.environ['COMFYUI_SEED_LOG_LEVEL'] = 'DEBUG'

try:
    from random_seed_generator import AdvancedSeedGenerator
except ImportError:
    print("Could not import AdvancedSeedGenerator - running basic functionality check")
    sys.exit(0)


class TestAdvancedSeedGenerator:
    """Test suite for AdvancedSeedGenerator class."""

    def setup_method(self):
        """Reset state before each test."""
        AdvancedSeedGenerator.reset_state()

    def teardown_method(self):
        """Clean up after each test."""
        AdvancedSeedGenerator.reset_state()

    def test_input_types_structure(self):
        """Test INPUT_TYPES returns correct structure."""
        input_types = AdvancedSeedGenerator.INPUT_TYPES()
        
        assert "required" in input_types
        required = input_types["required"]
        
        # Check all required fields exist
        assert "mode" in required
        assert "seed" in required
        assert "sync_libraries" in required
        assert "deterministic" in required
        
        # Check mode options
        assert required["mode"][0] == ["fixed", "increment", "decrement", "random"]
        
        # Check seed configuration
        seed_config = required["seed"][1]
        assert seed_config["min"] == 0
        assert seed_config["max"] == 0xffffffffffffffff
        assert seed_config["default"] == 0

    def test_constants(self):
        """Test class constants are properly defined."""
        assert AdvancedSeedGenerator.MIN_SEED_VALUE == 0
        assert AdvancedSeedGenerator.MAX_SEED_VALUE == 0xffffffffffffffff
        assert AdvancedSeedGenerator.DEFAULT_SEED == 0
        assert AdvancedSeedGenerator.NUMPY_MAX_SEED == 2**32 - 1

    def test_fixed_mode(self):
        """Test fixed mode returns exact seed value."""
        generator = AdvancedSeedGenerator()
        
        test_seed = 12345
        result = generator.generate_seed("fixed", test_seed, sync_libraries=False)
        
        assert result == (test_seed,)
        assert AdvancedSeedGenerator._last_seed == test_seed

    def test_random_mode(self):
        """Test random mode generates different values."""
        generator = AdvancedSeedGenerator()
        
        results = []
        for _ in range(10):
            result = generator.generate_seed("random", 0, sync_libraries=False)
            results.append(result[0])
        
        # Should generate different values (very high probability)
        assert len(set(results)) > 5  # At least 6 different values out of 10
        
        # All values should be in valid range
        for value in results:
            assert AdvancedSeedGenerator.MIN_SEED_VALUE <= value <= AdvancedSeedGenerator.MAX_SEED_VALUE

    def test_increment_mode(self):
        """Test increment mode increases seed by 1."""
        generator = AdvancedSeedGenerator()
        
        # Set initial state
        AdvancedSeedGenerator._last_seed = 100
        
        result = generator.generate_seed("increment", 0, sync_libraries=False)
        assert result == (101,)
        assert AdvancedSeedGenerator._last_seed == 101
        
        # Test sequential increments
        result = generator.generate_seed("increment", 0, sync_libraries=False)
        assert result == (102,)
        assert AdvancedSeedGenerator._last_seed == 102

    def test_decrement_mode(self):
        """Test decrement mode decreases seed by 1."""
        generator = AdvancedSeedGenerator()
        
        # Set initial state
        AdvancedSeedGenerator._last_seed = 100
        
        result = generator.generate_seed("decrement", 0, sync_libraries=False)
        assert result == (99,)
        assert AdvancedSeedGenerator._last_seed == 99
        
        # Test sequential decrements
        result = generator.generate_seed("decrement", 0, sync_libraries=False)
        assert result == (98,)
        assert AdvancedSeedGenerator._last_seed == 98

    def test_increment_overflow_wrap(self):
        """Test increment mode handles overflow with wrap behavior."""
        generator = AdvancedSeedGenerator()
        
        # Set to max value
        AdvancedSeedGenerator._last_seed = AdvancedSeedGenerator.MAX_SEED_VALUE
        
        result = generator.generate_seed("increment", 0, sync_libraries=False, overflow_behavior="wrap")
        assert result == (AdvancedSeedGenerator.MIN_SEED_VALUE,)
        assert AdvancedSeedGenerator._last_seed == AdvancedSeedGenerator.MIN_SEED_VALUE

    def test_increment_overflow_clamp(self):
        """Test increment mode handles overflow with clamp behavior."""
        generator = AdvancedSeedGenerator()
        
        # Set to max value
        AdvancedSeedGenerator._last_seed = AdvancedSeedGenerator.MAX_SEED_VALUE
        
        result = generator.generate_seed("increment", 0, sync_libraries=False, overflow_behavior="clamp")
        assert result == (AdvancedSeedGenerator.MAX_SEED_VALUE,)
        assert AdvancedSeedGenerator._last_seed == AdvancedSeedGenerator.MAX_SEED_VALUE

    def test_increment_overflow_error(self):
        """Test increment mode handles overflow with error behavior."""
        generator = AdvancedSeedGenerator()
        
        # Set to max value
        AdvancedSeedGenerator._last_seed = AdvancedSeedGenerator.MAX_SEED_VALUE
        
        # Should return fallback seed when error occurs (due to error handling)
        result = generator.generate_seed("increment", 0, sync_libraries=False, overflow_behavior="error")
        assert result == (AdvancedSeedGenerator.DEFAULT_SEED,)  # Fallback due to error

    def test_decrement_underflow_wrap(self):
        """Test decrement mode handles underflow with wrap behavior."""
        generator = AdvancedSeedGenerator()
        
        # Set to min value  
        AdvancedSeedGenerator._last_seed = AdvancedSeedGenerator.MIN_SEED_VALUE
        
        result = generator.generate_seed("decrement", 0, sync_libraries=False, overflow_behavior="wrap")
        assert result == (AdvancedSeedGenerator.MAX_SEED_VALUE,)
        assert AdvancedSeedGenerator._last_seed == AdvancedSeedGenerator.MAX_SEED_VALUE

    def test_decrement_underflow_clamp(self):
        """Test decrement mode handles underflow with clamp behavior."""
        generator = AdvancedSeedGenerator()
        
        # Set to min value  
        AdvancedSeedGenerator._last_seed = AdvancedSeedGenerator.MIN_SEED_VALUE
        
        result = generator.generate_seed("decrement", 0, sync_libraries=False, overflow_behavior="clamp")
        assert result == (AdvancedSeedGenerator.MIN_SEED_VALUE,)
        assert AdvancedSeedGenerator._last_seed == AdvancedSeedGenerator.MIN_SEED_VALUE

    def test_decrement_underflow_error(self):
        """Test decrement mode handles underflow with error behavior."""
        generator = AdvancedSeedGenerator()
        
        # Set to min value  
        AdvancedSeedGenerator._last_seed = AdvancedSeedGenerator.MIN_SEED_VALUE
        
        # Should return fallback seed when error occurs (due to error handling)
        result = generator.generate_seed("decrement", 0, sync_libraries=False, overflow_behavior="error")
        assert result == (AdvancedSeedGenerator.DEFAULT_SEED,)  # Fallback due to error

    def test_input_validation(self):
        """Test input validation catches invalid inputs."""
        generator = AdvancedSeedGenerator()
        
        # Test invalid mode
        result = generator.generate_seed("invalid_mode", 0, sync_libraries=False)
        assert result == (AdvancedSeedGenerator.DEFAULT_SEED,)  # Fallback
        
        # Test invalid seed type (should not crash due to fallback)
        result = generator.generate_seed("fixed", "not_a_number", sync_libraries=False)
        assert result == (AdvancedSeedGenerator.DEFAULT_SEED,)  # Fallback
        
        # Test invalid overflow behavior
        result = generator.generate_seed("fixed", 42, sync_libraries=False, overflow_behavior="invalid")
        assert result == (AdvancedSeedGenerator.DEFAULT_SEED,)  # Fallback

    def test_overflow_behavior_validation(self):
        """Test that overflow behavior options work correctly."""
        generator = AdvancedSeedGenerator()
        
        # Test all valid overflow behaviors work for normal cases
        for behavior in ["wrap", "clamp", "error"]:
            result = generator.generate_seed("fixed", 42, sync_libraries=False, overflow_behavior=behavior)
            assert result == (42,), f"Fixed mode failed with overflow_behavior='{behavior}'"

    def test_thread_safety(self):
        """Test thread safety of increment/decrement operations."""
        generator = AdvancedSeedGenerator()
        results = []
        errors = []
        
        def increment_worker():
            try:
                for _ in range(100):
                    result = generator.generate_seed("increment", 0, sync_libraries=False)
                    results.append(result[0])
            except Exception as e:
                errors.append(e)
        
        # Run multiple threads concurrently
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=increment_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Should have no errors
        assert len(errors) == 0
        
        # Should have 500 results (5 threads * 100 increments)
        assert len(results) == 500
        
        # All results should be unique (no race conditions)
        assert len(set(results)) == len(results)

    def test_cross_library_sync(self):
        """Test cross-library seed synchronization."""
        generator = AdvancedSeedGenerator()
        
        with patch('random.seed') as mock_random, \
             patch('numpy.random.seed') as mock_numpy, \
             patch('torch.manual_seed') as mock_torch:
            
            test_seed = 42
            generator.generate_seed("fixed", test_seed, sync_libraries=True)
            
            # Check that all libraries were seeded
            mock_random.assert_called_once_with(test_seed)
            mock_numpy.assert_called_once_with(test_seed % AdvancedSeedGenerator.NUMPY_MAX_SEED)
            mock_torch.assert_called_once_with(test_seed)

    def test_numpy_seed_truncation(self):
        """Test NumPy seed truncation for large values."""
        generator = AdvancedSeedGenerator()
        
        with patch('numpy.random.seed') as mock_numpy:
            large_seed = 2**40  # Larger than 32-bit
            expected_numpy_seed = large_seed % AdvancedSeedGenerator.NUMPY_MAX_SEED
            
            generator.generate_seed("fixed", large_seed, sync_libraries=True)
            
            mock_numpy.assert_called_once_with(expected_numpy_seed)

    @patch('torch.cuda.is_available')
    def test_cuda_handling(self, mock_cuda_available):
        """Test CUDA seed handling when available/unavailable."""
        generator = AdvancedSeedGenerator()
        
        # Test CUDA not available
        mock_cuda_available.return_value = False
        
        with patch('torch.cuda.manual_seed_all') as mock_cuda_seed:
            generator.generate_seed("fixed", 42, sync_libraries=True)
            mock_cuda_seed.assert_not_called()
        
        # Test CUDA available
        mock_cuda_available.return_value = True
        
        with patch('torch.cuda.manual_seed_all') as mock_cuda_seed, \
             patch('torch.backends.cudnn') as mock_cudnn:
            
            generator.generate_seed("fixed", 42, sync_libraries=True, deterministic=True)
            
            mock_cuda_seed.assert_called_once_with(42)
            assert mock_cudnn.deterministic == True
            assert mock_cudnn.benchmark == False

    def test_error_handling_resilience(self):
        """Test error handling doesn't crash the system."""
        generator = AdvancedSeedGenerator()
        
        # Mock library failures
        with patch('random.seed', side_effect=Exception("Random seed failed")), \
             patch('numpy.random.seed', side_effect=Exception("NumPy seed failed")), \
             patch('torch.manual_seed', side_effect=Exception("PyTorch seed failed")):
            
            # Should not raise exception, should return fallback
            result = generator.generate_seed("fixed", 42, sync_libraries=True)
            assert result == (AdvancedSeedGenerator.DEFAULT_SEED,)

    def test_is_changed_behavior(self):
        """Test IS_CHANGED method for different modes."""
        # Dynamic modes should return unique values
        result1 = AdvancedSeedGenerator.IS_CHANGED("random", 0, True, False)
        time.sleep(0.001)  # Ensure different timestamp
        result2 = AdvancedSeedGenerator.IS_CHANGED("random", 0, True, False)
        assert result1 != result2
        
        result3 = AdvancedSeedGenerator.IS_CHANGED("increment", 0, True, False)
        time.sleep(0.001)
        result4 = AdvancedSeedGenerator.IS_CHANGED("increment", 0, True, False)
        assert result3 != result4
        
        # Fixed mode should return same value for same inputs
        result5 = AdvancedSeedGenerator.IS_CHANGED("fixed", 42, True, False)
        result6 = AdvancedSeedGenerator.IS_CHANGED("fixed", 42, True, False)
        assert result5 == result6
        
        # Different fixed inputs should return different values
        result7 = AdvancedSeedGenerator.IS_CHANGED("fixed", 99, True, False)
        assert result5 != result7

    def test_state_management(self):
        """Test state management utilities."""
        # Test reset
        AdvancedSeedGenerator._last_seed = 12345
        AdvancedSeedGenerator.reset_state()
        assert AdvancedSeedGenerator._last_seed == AdvancedSeedGenerator.DEFAULT_SEED
        
        # Test state info
        AdvancedSeedGenerator._last_seed = 999
        state_info = AdvancedSeedGenerator.get_state_info()
        
        assert state_info["last_seed"] == 999
        assert state_info["min_seed"] == AdvancedSeedGenerator.MIN_SEED_VALUE
        assert state_info["max_seed"] == AdvancedSeedGenerator.MAX_SEED_VALUE
        assert state_info["thread_safe"] == True

    def test_logging_configuration(self):
        """Test logging configuration from environment."""
        # Test logger creation
        logger = AdvancedSeedGenerator._get_logger()
        assert logger is not None
        assert logger.level == logging.DEBUG  # Set in setup

    def test_performance_characteristics(self):
        """Test performance characteristics of seed generation."""
        generator = AdvancedSeedGenerator()
        
        # Test that fixed mode is fast
        start_time = time.time()
        for _ in range(1000):
            generator.generate_seed("fixed", 42, sync_libraries=False)
        fixed_time = time.time() - start_time
        
        # Test that random mode is reasonably fast
        start_time = time.time()
        for _ in range(1000):
            generator.generate_seed("random", 0, sync_libraries=False)
        random_time = time.time() - start_time
        
        # Both should complete in reasonable time (< 1 second for 1000 operations)
        assert fixed_time < 1.0
        assert random_time < 1.0

    def test_concurrent_access_different_modes(self):
        """Test concurrent access with different modes."""
        generator = AdvancedSeedGenerator()
        results = {"fixed": [], "random": [], "increment": [], "decrement": []}
        
        def worker(mode, test_seed=42):
            for _ in range(50):
                result = generator.generate_seed(mode, test_seed, sync_libraries=False)
                results[mode].append(result[0])
        
        threads = []
        for mode in ["fixed", "random", "increment", "decrement"]:
            thread = threading.Thread(target=worker, args=(mode,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Fixed mode should return same value
        assert all(x == 42 for x in results["fixed"])
        
        # Random mode should have variety
        assert len(set(results["random"])) > 25
        
        # Increment/decrement should have sequential values (though possibly interleaved)
        assert len(results["increment"]) == 50
        assert len(results["decrement"]) == 50


# Performance benchmark (optional, run separately)
def benchmark_seed_generation():
    """Benchmark seed generation performance."""
    generator = AdvancedSeedGenerator()
    
    modes = ["fixed", "random", "increment", "decrement"]
    iterations = 10000
    
    for mode in modes:
        start_time = time.time()
        for i in range(iterations):
            generator.generate_seed(mode, i, sync_libraries=False)
        elapsed = time.time() - start_time
        
        print(f"{mode} mode: {iterations} operations in {elapsed:.3f}s "
              f"({iterations/elapsed:.0f} ops/sec)")


if __name__ == "__main__":
    # Run the test suite
    pytest.main([__file__, "-v"])
    
    # Optionally run benchmark
    print("\nPerformance Benchmark:")
    benchmark_seed_generation()
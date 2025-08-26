import random
import time
import logging
import os
import threading
import numpy as np
import torch
from typing import Tuple, Union, Optional, Final, List

class AdvancedSeedGenerator:
    """
    An advanced node that generates and synchronizes seed values for reproducible or exploratory image generation.

    Features:
    - Multiple modes: fixed, increment, decrement, and random.
    - Robust state management for increment/decrement modes across executions.
    - Cross-library seed synchronization (random, numpy, torch).
    - Optional deterministic mode for complete reproducibility on CUDA devices.
    - Comprehensive error handling and input validation.
    - Configurable logging for debugging and monitoring.
    - Thread-safe operations for concurrent access.

    Generation Modes:
    - fixed: Returns the exact seed value provided by user
    - random: Generates a new random seed (0 to 2^64-1) on each execution
    - increment: Increments the last generated seed by 1
    - decrement: Decrements the last generated seed by 1

    Overflow Behavior (Configurable):
    Users can choose how increment/decrement operations handle boundary conditions:
    - \"wrap\" (default): Cycle around bounds (MAX -> MIN, MIN -> MAX)
    - \"clamp\": Stop at bounds (stay at MAX/MIN when limit reached)
    - \"error\": Raise ValueError exception when overflow would occur
    
    This provides flexibility for different use cases while maintaining predictable behavior.

    Cross-Library Compatibility:
    - Python random: Full 64-bit seed support
    - NumPy: Automatically truncates to 32-bit (logs when truncation occurs)
    - PyTorch: Full 64-bit seed support  
    - CUDA: Full 64-bit seed support when available

    Thread Safety:
    All state modifications are protected by threading.RLock() to ensure safe concurrent access.

    Environment Variables:
    - COMFYUI_SEED_LOG_LEVEL: Set logging level (DEBUG, INFO, WARNING, ERROR)

    Examples:
    >>> generator = AdvancedSeedGenerator()
    >>> result = generator.generate_seed("fixed", 42)  # Returns (42,)
    >>> result = generator.generate_seed("increment", 0)  # Returns (43,)
    >>> result = generator.generate_seed("random", 0)  # Returns (random_value,)
    """
    _last_seed = 0  # Class-level variable to store state across executions
    _logger = None  # Class-level logger instance
    _lock = threading.RLock()  # Thread-safe access to class state
    
    # Constants for validation - Using Final for immutability
    MIN_SEED_VALUE: Final[int] = 0
    # 64-bit maximum (18,446,744,073,709,551,615) chosen for:
    # - Compatibility with modern diffusion models (Stable Diffusion, SDXL, etc.)
    # - Full range support for PyTorch generators  
    # - Maximum entropy for random number generation
    # - Consistent with modern ML frameworks expecting 64-bit seeds
    MAX_SEED_VALUE: Final[int] = 0xffffffffffffffff  
    DEFAULT_SEED: Final[int] = 0
    
    # Configuration constants  
    NUMPY_MAX_SEED: Final[int] = 2**32 - 1  # NumPy limited to 32-bit seeds (4,294,967,295)
    MAX_BATCH_COUNT: Final[int] = 100000  # Maximum number of seeds that can be generated in batch mode
    
    # Backend selection thresholds for optimal performance
    TORCH_CUDA_BATCH_THRESHOLD: Final[int] = 1000  # Use GPU acceleration for batches >= 1000
    TORCH_CPU_BATCH_THRESHOLD: Final[int] = 100    # Use torch CPU for batches >= 100
    TORCH_BATCH_MIN_THRESHOLD: Final[int] = 10     # Minimum batch size to consider torch backend
    
    @classmethod
    def _get_logger(cls):
        """Get or create logger instance with configurable level."""
        if cls._logger is None:
            cls._logger = logging.getLogger(f"{__name__}.{cls.__name__}")
            log_level = os.environ.get('COMFYUI_SEED_LOG_LEVEL', 'WARNING')
            cls._logger.setLevel(getattr(logging, log_level.upper(), logging.WARNING))
            
            if not cls._logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
                handler.setFormatter(formatter)
                cls._logger.addHandler(handler)
                
        return cls._logger

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["fixed", "increment", "decrement", "random"],),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "step": 1,
                    "display": "number"
                }),
                "sync_libraries": ("BOOLEAN", {"default": True}),
                "deterministic": ("BOOLEAN", {"default": False}),
                "overflow_behavior": (["wrap", "clamp", "error"], {
                    "default": "wrap",
                    "tooltip": "How to handle overflow: wrap (cycle), clamp (stop at limits), error (raise exception)"
                }),
                "use_torch_backend": (["auto", "random", "torch"], {
                    "default": "auto",
                    "tooltip": "Random backend: auto (optimal), random (Python), torch (PyTorch)"
                }),
                "batch_count": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100000,  # Using literal value in INPUT_TYPES as class constants not accessible in classmethod
                    "step": 1,
                    "tooltip": "Number of seeds to generate (batch mode)"
                }),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("seed", "batch_count")
    FUNCTION = "generate_seed"
    CATEGORY = "utils"

    def generate_seed(self, mode: str, seed: int, sync_libraries: bool = True, deterministic: bool = False, overflow_behavior: str = "wrap", use_torch_backend: str = "auto", batch_count: int = 1) -> Tuple[int, int]:
        """
        Generate seed value(s) based on the selected mode and apply them if requested.

        Args:
            mode (str): The seed generation mode.
            seed (int): The user-defined seed (for 'fixed' mode).
            sync_libraries (bool): If True, synchronize the seed across Python, NumPy, and PyTorch.
            deterministic (bool): If True, enable full deterministic mode in PyTorch (may impact performance).
            overflow_behavior (str): How to handle overflow - "wrap", "clamp", or "error".
            use_torch_backend (str): Backend selection - "auto", "random", or "torch".
            batch_count (int): Number of seeds to generate (batch mode).

        Returns:
            A tuple containing the generated integer seed and batch count.
            
        Raises:
            ValueError: If mode is invalid, seed is out of bounds, or overflow occurs with "error" behavior.
            RuntimeError: If seed generation or library synchronization fails.
        """
        logger = self._get_logger()
        
        try:
            # Validate inputs
            self._validate_inputs(mode, seed, sync_libraries, deterministic, overflow_behavior, use_torch_backend, batch_count)
            
            logger.debug(f"Generating seed with mode='{mode}', seed={seed}, sync={sync_libraries}, deterministic={deterministic}, backend='{use_torch_backend}', batch={batch_count}")
            
            # Generate seed(s) based on mode and backend
            if batch_count == 1:
                final_seed = self._generate_seed_by_mode(mode, seed, overflow_behavior, use_torch_backend)
            else:
                # Batch mode - always returns the first seed for compatibility
                seeds = self._generate_seed_batch(mode, seed, batch_count, overflow_behavior, use_torch_backend)
                final_seed = seeds[0] if seeds else self.DEFAULT_SEED
                logger.info(f"Generated {len(seeds)} seeds in batch mode, using first seed: {final_seed}")
            
            # Validate and clamp final seed
            final_seed = self._validate_and_clamp_seed(final_seed)
            
            # Update class-level state (thread-safe)
            # Skip state update for batch increment/decrement as _generate_sequential_seed_batch already handled it
            if not (batch_count > 1 and mode in ['increment', 'decrement']):
                with self.__class__._lock:
                    self.__class__._last_seed = final_seed
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Updated _last_seed to {final_seed}")
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Skipped _last_seed update for batch {mode} mode (already handled by batch generator)")
            
            # Apply seed synchronization if requested
            if sync_libraries:
                self._apply_seed(final_seed, deterministic)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Applied seed {final_seed} to libraries")
            
            if logger.isEnabledFor(logging.INFO):
                logger.info(f"Successfully generated seed: {final_seed} (mode: {mode})")
            return (final_seed, batch_count)
            
        except Exception as e:
            logger.error(f"Failed to generate seed: {str(e)}")
            # Return fallback seed to prevent complete failure
            fallback_seed = self.DEFAULT_SEED
            logger.warning(f"Using fallback seed: {fallback_seed}")
            return (fallback_seed, 1)
    
    def _validate_inputs(self, mode: str, seed: int, sync_libraries: bool, deterministic: bool, overflow_behavior: str, use_torch_backend: str, batch_count: int) -> None:
        """Validate all input parameters."""
        valid_modes = ["fixed", "increment", "decrement", "random"]
        valid_overflow_behaviors = ["wrap", "clamp", "error"]
        valid_backends = ["auto", "random", "torch"]
        
        if not isinstance(mode, str) or mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")
        
        if not isinstance(seed, int):
            raise ValueError(f"Seed must be an integer, got {type(seed).__name__}")
        
        if seed < self.MIN_SEED_VALUE or seed > self.MAX_SEED_VALUE:
            raise ValueError(f"Seed {seed} out of valid range [{self.MIN_SEED_VALUE}, {self.MAX_SEED_VALUE}]")
        
        if not isinstance(sync_libraries, bool):
            raise ValueError(f"sync_libraries must be boolean, got {type(sync_libraries).__name__}")
        
        if not isinstance(deterministic, bool):
            raise ValueError(f"deterministic must be boolean, got {type(deterministic).__name__}")
        
        if not isinstance(overflow_behavior, str) or overflow_behavior not in valid_overflow_behaviors:
            raise ValueError(f"Invalid overflow_behavior '{overflow_behavior}'. Must be one of: {valid_overflow_behaviors}")
        
        if not isinstance(use_torch_backend, str) or use_torch_backend not in valid_backends:
            raise ValueError(f"Invalid use_torch_backend '{use_torch_backend}'. Must be one of: {valid_backends}")
        
        if not isinstance(batch_count, int) or batch_count < 1 or batch_count > self.MAX_BATCH_COUNT:
            raise ValueError(f"batch_count must be an integer between 1 and {self.MAX_BATCH_COUNT}, got {batch_count}")
    
    def _generate_seed_by_mode(self, mode: str, seed: int, overflow_behavior: str = "wrap", use_torch_backend: str = "auto") -> int:
        """
        Generate seed value based on the specified mode and backend.
        
        Thread-safe generation with configurable overflow handling:
        - "wrap": Cycle around bounds (MAX -> MIN, MIN -> MAX)
        - "clamp": Stop at bounds (stay at MAX/MIN)
        - "error": Raise exception on overflow
        
        Backend selection:
        - "auto": Use optimal backend (random for single seeds, torch for batches)
        - "random": Force Python random module
        - "torch": Force PyTorch backend
        """
        try:
            if mode == 'fixed':
                return seed
            elif mode == 'random':
                return self._generate_random_seed(use_torch_backend)
            elif mode == 'increment':
                with self.__class__._lock:
                    current_seed = self.__class__._last_seed
                    new_seed = current_seed + 1
                    return self._handle_overflow(new_seed, current_seed, "increment", overflow_behavior)
            elif mode == 'decrement':
                with self.__class__._lock:
                    current_seed = self.__class__._last_seed
                    new_seed = current_seed - 1
                    return self._handle_overflow(new_seed, current_seed, "decrement", overflow_behavior)
            else:
                raise ValueError(f"Unsupported mode: {mode}")
        except Exception as e:
            raise RuntimeError(f"Failed to generate seed for mode '{mode}': {str(e)}")
    
    def _generate_random_seed(self, use_torch_backend: str = "auto") -> int:
        """
        Generate a single random seed using the specified backend.
        
        Args:
            use_torch_backend (str): Backend preference - "auto", "random", or "torch"
            
        Returns:
            int: Random seed value in valid range
        """
        backend = self._select_optimal_backend(use_torch_backend, batch_size=1)
        logger = self._get_logger()
        
        try:
            if backend == "torch":
                # Use torch.randint for direct integer generation (more stable)
                with torch.no_grad():
                    # Use randint with safe range (PyTorch has limitations with very large ranges)
                    # Use 48-bit range for better compatibility while maintaining good entropy
                    torch_max = min(self.MAX_SEED_VALUE, 2**48 - 1)
                    seed_tensor = torch.randint(
                        low=self.MIN_SEED_VALUE, 
                        high=torch_max + 1, 
                        size=(1,), 
                        dtype=torch.int64,
                        device='cpu'
                    )
                    return int(seed_tensor.item())
            else:
                # Use Python random (default, most efficient for single values)
                return random.randint(self.MIN_SEED_VALUE, self.MAX_SEED_VALUE)
                
        except Exception as e:
            logger.warning(f"Failed to generate random seed with {backend} backend: {str(e)}")
            # Fallback to Python random
            return random.randint(self.MIN_SEED_VALUE, self.MAX_SEED_VALUE)
    
    def _generate_seed_batch(self, mode: str, seed: int, batch_count: int, overflow_behavior: str = "wrap", use_torch_backend: str = "auto") -> List[int]:
        """
        Generate multiple seeds efficiently using batch operations.
        
        Args:
            mode (str): The seed generation mode
            seed (int): Base seed value (for fixed mode)
            batch_count (int): Number of seeds to generate
            overflow_behavior (str): How to handle overflow
            use_torch_backend (str): Backend preference
            
        Returns:
            List[int]: List of generated seed values
        """
        logger = self._get_logger()
        
        try:
            if mode == 'fixed':
                return [seed] * batch_count
            elif mode == 'random':
                return self._generate_random_seed_batch(batch_count, use_torch_backend)
            elif mode in ['increment', 'decrement']:
                return self._generate_sequential_seed_batch(mode, batch_count, overflow_behavior)
            else:
                raise ValueError(f"Unsupported mode for batch generation: {mode}")
                
        except Exception as e:
            logger.error(f"Failed to generate seed batch: {str(e)}")
            # Fallback to single seed repeated
            fallback_seed = self._generate_seed_by_mode('fixed', self.DEFAULT_SEED, overflow_behavior, use_torch_backend)
            return [fallback_seed] * batch_count
    
    def _generate_random_seed_batch(self, batch_count: int, use_torch_backend: str = "auto") -> List[int]:
        """
        Generate multiple random seeds efficiently.
        
        Args:
            batch_count (int): Number of seeds to generate
            use_torch_backend (str): Backend preference
            
        Returns:
            List[int]: List of random seed values
        """
        backend = self._select_optimal_backend(use_torch_backend, batch_size=batch_count)
        logger = self._get_logger()
        
        try:
            if backend == "torch" and batch_count >= self.TORCH_BATCH_MIN_THRESHOLD:
                # Use torch for efficient batch generation
                with torch.no_grad():
                    # Use CPU to avoid GPU memory overhead for small batches
                    device = 'cpu'
                    if batch_count >= self.TORCH_CUDA_BATCH_THRESHOLD and torch.cuda.is_available():
                        device = 'cuda'
                    
                    # Use randint with safe range for batch generation
                    torch_max = min(self.MAX_SEED_VALUE, 2**48 - 1)
                    seed_vals = torch.randint(
                        low=self.MIN_SEED_VALUE,
                        high=torch_max + 1,
                        size=(batch_count,),
                        dtype=torch.int64,
                        device=device
                    )
                    
                    if device == 'cuda':
                        seed_vals = seed_vals.cpu()
                    
                    return seed_vals.tolist()
            else:
                # Use Python random for smaller batches or forced random backend
                return [random.randint(self.MIN_SEED_VALUE, self.MAX_SEED_VALUE) for _ in range(batch_count)]
                
        except Exception as e:
            logger.warning(f"Failed to generate batch with {backend} backend: {str(e)}")
            # Fallback to Python random
            return [random.randint(self.MIN_SEED_VALUE, self.MAX_SEED_VALUE) for _ in range(batch_count)]
    
    def _generate_sequential_seed_batch(self, mode: str, batch_count: int, overflow_behavior: str) -> List[int]:
        """
        Generate sequential seeds (increment/decrement) in batch.
        
        Args:
            mode (str): "increment" or "decrement"
            batch_count (int): Number of seeds to generate
            overflow_behavior (str): How to handle overflow
            
        Returns:
            List[int]: List of sequential seed values
        """
        seeds = []
        
        with self.__class__._lock:
            current_seed = self.__class__._last_seed
            
            for i in range(batch_count):
                if mode == 'increment':
                    new_seed = current_seed + 1
                else:  # decrement
                    new_seed = current_seed - 1
                
                # Handle overflow for each step
                final_seed = self._handle_overflow(new_seed, current_seed, mode, overflow_behavior)
                seeds.append(final_seed)
                current_seed = final_seed
            
            # Update the class state with the final seed
            self.__class__._last_seed = current_seed
        
        return seeds
    
    def _select_optimal_backend(self, use_torch_backend: str, batch_size: int = 1) -> str:
        """
        Select the optimal backend based on preference and batch size.
        
        Args:
            use_torch_backend (str): User preference - "auto", "random", or "torch"
            batch_size (int): Number of seeds to generate
            
        Returns:
            str: Selected backend - "random" or "torch"
        """
        if use_torch_backend == "random":
            return "random"
        elif use_torch_backend == "torch":
            return "torch"
        else:  # "auto"
            # Auto-select based on batch size and PyTorch availability
            if batch_size >= self.TORCH_CUDA_BATCH_THRESHOLD and torch.cuda.is_available():
                return "torch"
            elif batch_size >= self.TORCH_CPU_BATCH_THRESHOLD:  # Large batches benefit from torch even on CPU
                return "torch"
            else:
                return "random"
    
    def _handle_overflow(self, new_seed: int, current_seed: int, operation: str, overflow_behavior: str) -> int:
        """
        Handle overflow/underflow based on the specified behavior.
        
        Args:
            new_seed (int): The calculated new seed value
            current_seed (int): The current seed value
            operation (str): "increment" or "decrement"
            overflow_behavior (str): "wrap", "clamp", or "error"
            
        Returns:
            int: The final seed value after overflow handling
            
        Raises:
            ValueError: If overflow_behavior is "error" and overflow occurs
        """
        logger = self._get_logger()
        
        # Check for overflow conditions
        if operation == "increment" and new_seed > self.MAX_SEED_VALUE:
            if overflow_behavior == "wrap":
                result = self.MIN_SEED_VALUE
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Increment overflow: {current_seed} -> {result} (wrapped)")
                return result
            elif overflow_behavior == "clamp":
                result = self.MAX_SEED_VALUE
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Increment overflow: {current_seed} -> {result} (clamped)")
                return result
            elif overflow_behavior == "error":
                raise ValueError(f"Increment overflow: seed {current_seed} + 1 exceeds maximum {self.MAX_SEED_VALUE}")
                
        elif operation == "decrement" and new_seed < self.MIN_SEED_VALUE:
            if overflow_behavior == "wrap":
                result = self.MAX_SEED_VALUE
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Decrement underflow: {current_seed} -> {result} (wrapped)")
                return result
            elif overflow_behavior == "clamp":
                result = self.MIN_SEED_VALUE
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Decrement underflow: {current_seed} -> {result} (clamped)")
                return result
            elif overflow_behavior == "error":
                raise ValueError(f"Decrement underflow: seed {current_seed} - 1 is below minimum {self.MIN_SEED_VALUE}")
        
        # No overflow occurred
        return new_seed
    
    def _validate_and_clamp_seed(self, seed: int) -> int:
        """Validate and clamp seed to valid range."""
        if not isinstance(seed, int):
            raise ValueError(f"Generated seed must be integer, got {type(seed).__name__}")
        
        # Clamp to valid range
        clamped_seed = max(self.MIN_SEED_VALUE, min(seed, self.MAX_SEED_VALUE))
        
        if clamped_seed != seed:
            self._get_logger().warning(f"Seed {seed} clamped to {clamped_seed}")
        
        return clamped_seed

    def _apply_seed(self, seed_value: int, deterministic: bool = False) -> None:
        """
        Apply the seed value across multiple libraries for consistent randomization.
        
        Args:
            seed_value (int): The seed value to apply
            deterministic (bool): Whether to enable full deterministic mode
            
        Raises:
            RuntimeError: If seed application fails for any library
        """
        logger = self._get_logger()
        errors = []
        
        # Apply seeds to all available libraries
        errors.extend(self._apply_python_seed(seed_value, logger))
        errors.extend(self._apply_numpy_seed(seed_value, logger))
        errors.extend(self._apply_pytorch_seed(seed_value, logger))
        errors.extend(self._apply_cuda_seeds(seed_value, deterministic, logger))
        
        # Report any errors but don't fail completely
        if errors:
            logger.warning(f"Seed application completed with {len(errors)} errors: {'; '.join(errors)}")
        elif logger.isEnabledFor(logging.DEBUG):
            logger.debug("Successfully applied seed to all available libraries")
    
    def _apply_python_seed(self, seed_value: int, logger: logging.Logger) -> list:
        """Apply seed to Python's random module."""
        try:
            random.seed(seed_value)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Applied seed to Python random module")
            return []
        except Exception as e:
            error_msg = f"Failed to set Python random seed: {str(e)}"
            logger.error(error_msg)
            return [error_msg]
    
    def _apply_numpy_seed(self, seed_value: int, logger: logging.Logger) -> list:
        """Apply seed to NumPy's random module with 32-bit truncation."""
        try:
            # Handle potential overflow for NumPy (uses 32-bit seeds)
            numpy_seed = seed_value % self.NUMPY_MAX_SEED
            np.random.seed(numpy_seed)
            if numpy_seed != seed_value and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"NumPy seed truncated from {seed_value} to {numpy_seed}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Applied seed to NumPy random")
            return []
        except Exception as e:
            error_msg = f"Failed to set NumPy random seed: {str(e)}"
            logger.error(error_msg)
            return [error_msg]
    
    def _apply_pytorch_seed(self, seed_value: int, logger: logging.Logger) -> list:
        """Apply seed to PyTorch."""
        try:
            torch.manual_seed(seed_value)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Applied seed to PyTorch")
            return []
        except Exception as e:
            error_msg = f"Failed to set PyTorch seed: {str(e)}"
            logger.error(error_msg)
            return [error_msg]
    
    def _apply_cuda_seeds(self, seed_value: int, deterministic: bool, logger: logging.Logger) -> list:
        """Apply CUDA seeds and configure deterministic mode."""
        errors = []
        
        if not torch.cuda.is_available():
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("CUDA not available, skipping CUDA seed configuration")
            return errors
        
        # Apply CUDA seed
        try:
            torch.cuda.manual_seed_all(seed_value)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Applied seed to CUDA")
        except Exception as e:
            error_msg = f"Failed to set CUDA seed: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
        
        # Configure CUDNN deterministic mode
        try:
            torch.backends.cudnn.deterministic = deterministic
            torch.backends.cudnn.benchmark = not deterministic
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Set CUDNN deterministic={deterministic}, benchmark={not deterministic}")
        except Exception as e:
            error_msg = f"Failed to configure CUDNN: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
        
        return errors

    @classmethod
    def IS_CHANGED(cls, mode: str, seed: int, sync_libraries: bool, deterministic: bool, overflow_behavior: str = "wrap", use_torch_backend: str = "auto", batch_count: int = 1) -> Union[float, str]:
        """
        Force re-execution for modes that should produce a new result on each run.
        
        Optimized for performance - minimal logging overhead.
        
        Args:
            mode (str): The seed generation mode
            seed (int): The seed value (unused for dynamic modes)
            sync_libraries (bool): Whether libraries are synchronized
            deterministic (bool): Whether deterministic mode is enabled
            overflow_behavior (str): How to handle overflow (affects caching for increment/decrement)
            use_torch_backend (str): Backend preference for random generation
            batch_count (int): Number of seeds to generate
            
        Returns:
            Union[float, str]: Unique value to force re-execution for dynamic modes,
                             or constant for static modes
        """
        try:
            # Dynamic modes always need re-execution
            if mode in ["random", "increment", "decrement"]:
                timestamp = time.time()
                # Only log if debug is explicitly enabled to avoid performance overhead
                logger = cls._get_logger()
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"IS_CHANGED: {timestamp} for dynamic mode '{mode}'")
                return timestamp
            else:
                # For fixed mode, return a stable cache key including all parameters
                cache_key = f"fixed_{seed}_{sync_libraries}_{deterministic}_{overflow_behavior}_{use_torch_backend}_{batch_count}"
                logger = cls._get_logger()
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"IS_CHANGED: '{cache_key}' for static mode '{mode}'")
                return cache_key
        except Exception as e:
            # Minimal error handling - don't call logger to avoid recursion
            print(f"[AdvancedSeedGenerator] Error in IS_CHANGED: {e}")
            # Fallback to timestamp to ensure execution
            return time.time()
    
    @classmethod
    def reset_state(cls) -> None:
        """Reset the class-level state. Useful for testing or reinitialization."""
        with cls._lock:
            cls._last_seed = cls.DEFAULT_SEED
        if cls._logger and cls._logger.isEnabledFor(logging.INFO):
            cls._logger.info("Reset AdvancedSeedGenerator state")
    
    @classmethod
    def get_state_info(cls) -> dict:
        """Get current state information for debugging."""
        with cls._lock:
            current_seed = cls._last_seed
        return {
            "last_seed": current_seed,
            "min_seed": cls.MIN_SEED_VALUE,
            "max_seed": cls.MAX_SEED_VALUE,
            "default_seed": cls.DEFAULT_SEED,
            "numpy_max_seed": cls.NUMPY_MAX_SEED,
            "logger_level": cls._logger.level if cls._logger else "Not initialized",
            "thread_safe": True
        }

# ComfyUI Registration
NODE_CLASS_MAPPINGS = {
    "AdvancedSeedGenerator": AdvancedSeedGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdvancedSeedGenerator": "🎲 Advanced Seed Generator"
}

# Export for module-level access
__all__ = ["AdvancedSeedGenerator", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
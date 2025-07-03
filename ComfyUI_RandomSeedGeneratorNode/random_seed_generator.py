import random
import time
import logging
import os
import threading
import numpy as np
import torch
from typing import Tuple, Union, Optional, Final

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

    Overflow Behavior:
    When increment/decrement operations exceed valid bounds, wrap-around occurs:
    - Increment overflow: MAX_SEED (2^64-1) -> MIN_SEED (0)
    - Decrement underflow: MIN_SEED (0) -> MAX_SEED (2^64-1)
    
    This ensures continuous operation without errors while maintaining predictable behavior.

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
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "generate_seed"
    CATEGORY = "utils/random"

    def generate_seed(self, mode: str, seed: int, sync_libraries: bool = True, deterministic: bool = False) -> Tuple[int]:
        """
        Generate a seed value based on the selected mode and apply it if requested.

        Args:
            mode (str): The seed generation mode.
            seed (int): The user-defined seed (for 'fixed' mode).
            sync_libraries (bool): If True, synchronize the seed across Python, NumPy, and PyTorch.
            deterministic (bool): If True, enable full deterministic mode in PyTorch (may impact performance).

        Returns:
            A tuple containing the generated integer seed.
            
        Raises:
            ValueError: If mode is invalid or seed is out of bounds.
            RuntimeError: If seed generation or library synchronization fails.
        """
        logger = self._get_logger()
        
        try:
            # Validate inputs
            self._validate_inputs(mode, seed, sync_libraries, deterministic)
            
            logger.debug(f"Generating seed with mode='{mode}', seed={seed}, sync={sync_libraries}, deterministic={deterministic}")
            
            # Generate seed based on mode
            final_seed = self._generate_seed_by_mode(mode, seed)
            
            # Validate and clamp final seed
            final_seed = self._validate_and_clamp_seed(final_seed)
            
            # Update class-level state (thread-safe)
            with self.__class__._lock:
                self.__class__._last_seed = final_seed
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Updated _last_seed to {final_seed}")
            
            # Apply seed synchronization if requested
            if sync_libraries:
                self._apply_seed(final_seed, deterministic)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Applied seed {final_seed} to libraries")
            
            if logger.isEnabledFor(logging.INFO):
                logger.info(f"Successfully generated seed: {final_seed} (mode: {mode})")
            return (final_seed,)
            
        except Exception as e:
            logger.error(f"Failed to generate seed: {str(e)}")
            # Return fallback seed to prevent complete failure
            fallback_seed = self.DEFAULT_SEED
            logger.warning(f"Using fallback seed: {fallback_seed}")
            return (fallback_seed,)
    
    def _validate_inputs(self, mode: str, seed: int, sync_libraries: bool, deterministic: bool) -> None:
        """Validate all input parameters."""
        valid_modes = ["fixed", "increment", "decrement", "random"]
        
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
    
    def _generate_seed_by_mode(self, mode: str, seed: int) -> int:
        """
        Generate seed value based on the specified mode.
        
        Thread-safe generation with proper overflow handling.
        For increment/decrement modes, overflow behavior wraps around:
        - Increment overflow: MAX_SEED_VALUE -> MIN_SEED_VALUE 
        - Decrement underflow: MIN_SEED_VALUE -> MAX_SEED_VALUE
        """
        try:
            if mode == 'fixed':
                return seed
            elif mode == 'random':
                return random.randint(self.MIN_SEED_VALUE, self.MAX_SEED_VALUE)
            elif mode == 'increment':
                with self.__class__._lock:
                    current_seed = self.__class__._last_seed
                    new_seed = current_seed + 1
                    if new_seed > self.MAX_SEED_VALUE:
                        logger = self._get_logger()
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Increment overflow: {current_seed} -> {self.MIN_SEED_VALUE} (wrapped)")
                        return self.MIN_SEED_VALUE
                    return new_seed
            elif mode == 'decrement':
                with self.__class__._lock:
                    current_seed = self.__class__._last_seed
                    new_seed = current_seed - 1
                    if new_seed < self.MIN_SEED_VALUE:
                        logger = self._get_logger()
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Decrement underflow: {current_seed} -> {self.MAX_SEED_VALUE} (wrapped)")
                        return self.MAX_SEED_VALUE
                    return new_seed
            else:
                raise ValueError(f"Unsupported mode: {mode}")
        except Exception as e:
            raise RuntimeError(f"Failed to generate seed for mode '{mode}': {str(e)}")
    
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
    def IS_CHANGED(cls, mode: str, seed: int, sync_libraries: bool, deterministic: bool) -> Union[float, str]:
        """
        Force re-execution for modes that should produce a new result on each run.
        
        Optimized for performance - minimal logging overhead.
        
        Args:
            mode (str): The seed generation mode
            seed (int): The seed value (unused for dynamic modes)
            sync_libraries (bool): Whether libraries are synchronized
            deterministic (bool): Whether deterministic mode is enabled
            
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
                # For fixed mode, return a stable cache key
                cache_key = f"fixed_{seed}_{sync_libraries}_{deterministic}"
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
    "AdvancedSeedGenerator": "Advanced Seed Generator"
}

# Export for module-level access
__all__ = ["AdvancedSeedGenerator", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
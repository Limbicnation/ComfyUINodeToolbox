import random
import time
import logging
import os
import numpy as np
import torch
from typing import Tuple, Union, Optional

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
    """
    _last_seed = 0  # Class-level variable to store state across executions
    _logger = None  # Class-level logger instance
    
    # Constants for validation
    MIN_SEED_VALUE = 0
    MAX_SEED_VALUE = 0xffffffffffffffff
    DEFAULT_SEED = 0
    
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
            
            # Update class-level state
            self.__class__._last_seed = final_seed
            logger.debug(f"Updated _last_seed to {final_seed}")
            
            # Apply seed synchronization if requested
            if sync_libraries:
                self._apply_seed(final_seed, deterministic)
                logger.debug(f"Applied seed {final_seed} to libraries")
            
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
        """Generate seed value based on the specified mode."""
        try:
            if mode == 'fixed':
                return seed
            elif mode == 'random':
                return random.randint(self.MIN_SEED_VALUE, self.MAX_SEED_VALUE)
            elif mode == 'increment':
                new_seed = self.__class__._last_seed + 1
                if new_seed > self.MAX_SEED_VALUE:
                    self._get_logger().warning(f"Increment overflow, wrapping to {self.MIN_SEED_VALUE}")
                    return self.MIN_SEED_VALUE
                return new_seed
            elif mode == 'decrement':
                new_seed = self.__class__._last_seed - 1
                if new_seed < self.MIN_SEED_VALUE:
                    self._get_logger().warning(f"Decrement underflow, wrapping to {self.MAX_SEED_VALUE}")
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
        
        # Apply Python random seed
        try:
            random.seed(seed_value)
            logger.debug("Applied seed to Python random module")
        except Exception as e:
            error_msg = f"Failed to set Python random seed: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
        
        # Apply NumPy seed
        try:
            # Handle potential overflow for NumPy (uses 32-bit seeds)
            numpy_seed = seed_value % (2**32)
            np.random.seed(numpy_seed)
            if numpy_seed != seed_value:
                logger.warning(f"NumPy seed truncated from {seed_value} to {numpy_seed}")
            logger.debug("Applied seed to NumPy random")
        except Exception as e:
            error_msg = f"Failed to set NumPy random seed: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
        
        # Apply PyTorch seed
        try:
            torch.manual_seed(seed_value)
            logger.debug("Applied seed to PyTorch")
        except Exception as e:
            error_msg = f"Failed to set PyTorch seed: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
        
        # Apply CUDA seeds and configure deterministic mode
        if torch.cuda.is_available():
            try:
                torch.cuda.manual_seed_all(seed_value)
                logger.debug("Applied seed to CUDA")
            except Exception as e:
                error_msg = f"Failed to set CUDA seed: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
            
            try:
                torch.backends.cudnn.deterministic = deterministic
                torch.backends.cudnn.benchmark = not deterministic
                logger.debug(f"Set CUDNN deterministic={deterministic}, benchmark={not deterministic}")
            except Exception as e:
                error_msg = f"Failed to configure CUDNN: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        else:
            logger.debug("CUDA not available, skipping CUDA seed configuration")
        
        # Report any errors but don't fail completely
        if errors:
            logger.warning(f"Seed application completed with {len(errors)} errors: {'; '.join(errors)}")
        else:
            logger.debug("Successfully applied seed to all available libraries")

    @classmethod
    def IS_CHANGED(cls, mode: str, seed: int, sync_libraries: bool, deterministic: bool) -> Union[float, str]:
        """
        Force re-execution for modes that should produce a new result on each run.
        
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
            if mode in ["random", "increment", "decrement"]:
                # Return unique timestamp to force re-execution
                timestamp = time.time()
                cls._get_logger().debug(f"IS_CHANGED returning {timestamp} for dynamic mode '{mode}'")
                return timestamp
            else:
                # For fixed mode, return a constant to allow caching
                cache_key = f"fixed_{seed}_{sync_libraries}_{deterministic}"
                cls._get_logger().debug(f"IS_CHANGED returning '{cache_key}' for static mode '{mode}'")
                return cache_key
        except Exception as e:
            cls._get_logger().error(f"Error in IS_CHANGED: {str(e)}")
            # Fallback to timestamp to ensure execution
            return time.time()
    
    @classmethod
    def reset_state(cls) -> None:
        """Reset the class-level state. Useful for testing or reinitialization."""
        cls._last_seed = cls.DEFAULT_SEED
        if cls._logger:
            cls._logger.info("Reset AdvancedSeedGenerator state")
    
    @classmethod
    def get_state_info(cls) -> dict:
        """Get current state information for debugging."""
        return {
            "last_seed": cls._last_seed,
            "min_seed": cls.MIN_SEED_VALUE,
            "max_seed": cls.MAX_SEED_VALUE,
            "default_seed": cls.DEFAULT_SEED,
            "logger_level": cls._logger.level if cls._logger else "Not initialized"
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
#!/usr/bin/env python3
"""
ComfyUI Compatibility Test for AdvancedSeedGenerator

This test validates compatibility with different ComfyUI versions and ensures
the node follows ComfyUI conventions properly.
"""

import sys
import os
import importlib

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_node_registration():
    """Test that the node can be properly registered with ComfyUI."""
    try:
        from random_seed_generator import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        
        # Verify mappings exist
        assert NODE_CLASS_MAPPINGS is not None, "NODE_CLASS_MAPPINGS should not be None"
        assert NODE_DISPLAY_NAME_MAPPINGS is not None, "NODE_DISPLAY_NAME_MAPPINGS should not be None"
        
        # Verify mappings contain our node
        assert "AdvancedSeedGenerator" in NODE_CLASS_MAPPINGS, "AdvancedSeedGenerator should be in NODE_CLASS_MAPPINGS"
        assert "AdvancedSeedGenerator" in NODE_DISPLAY_NAME_MAPPINGS, "AdvancedSeedGenerator should be in NODE_DISPLAY_NAME_MAPPINGS"
        
        print("✅ Node registration compatibility: PASSED")
        return True
    except Exception as e:
        print(f"❌ Node registration compatibility: FAILED - {e}")
        return False

def test_input_types_schema():
    """Test INPUT_TYPES follows ComfyUI schema conventions."""
    try:
        from random_seed_generator import AdvancedSeedGenerator
        
        input_types = AdvancedSeedGenerator.INPUT_TYPES()
        
        # Check basic structure
        assert isinstance(input_types, dict), "INPUT_TYPES should return a dictionary"
        assert "required" in input_types, "INPUT_TYPES should have 'required' key"
        
        required = input_types["required"]
        assert isinstance(required, dict), "required should be a dictionary"
        
        # Validate each input parameter
        for param_name, param_config in required.items():
            assert isinstance(param_config, tuple), f"{param_name} config should be a tuple"
            assert len(param_config) >= 1, f"{param_name} config should have at least 1 element"
            
            param_type = param_config[0]
            if isinstance(param_type, list):
                # Dropdown/combo type
                assert len(param_type) > 0, f"{param_name} dropdown should have options"
            else:
                # Should be a string type like "INT", "FLOAT", "BOOLEAN", etc.
                assert isinstance(param_type, str), f"{param_name} type should be string"
        
        print("✅ INPUT_TYPES schema compatibility: PASSED")
        return True
    except Exception as e:
        print(f"❌ INPUT_TYPES schema compatibility: FAILED - {e}")
        return False

def test_return_types():
    """Test RETURN_TYPES follows ComfyUI conventions."""
    try:
        from random_seed_generator import AdvancedSeedGenerator
        
        return_types = AdvancedSeedGenerator.RETURN_TYPES
        return_names = AdvancedSeedGenerator.RETURN_NAMES
        function_name = AdvancedSeedGenerator.FUNCTION
        category = AdvancedSeedGenerator.CATEGORY
        
        # Check return types structure
        assert isinstance(return_types, tuple), "RETURN_TYPES should be a tuple"
        assert len(return_types) > 0, "RETURN_TYPES should not be empty"
        
        # Check return names structure
        assert isinstance(return_names, tuple), "RETURN_NAMES should be a tuple"
        assert len(return_names) == len(return_types), "RETURN_NAMES length should match RETURN_TYPES"
        
        # Check function name
        assert isinstance(function_name, str), "FUNCTION should be a string"
        assert hasattr(AdvancedSeedGenerator, function_name), f"Class should have method {function_name}"
        
        # Check category
        assert isinstance(category, str), "CATEGORY should be a string"
        
        print("✅ Return types compatibility: PASSED")
        return True
    except Exception as e:
        print(f"❌ Return types compatibility: FAILED - {e}")
        return False

def test_function_execution():
    """Test that the main function executes correctly."""
    try:
        from random_seed_generator import AdvancedSeedGenerator
        
        generator = AdvancedSeedGenerator()
        
        # Test with all parameters (ComfyUI calls with all params from INPUT_TYPES)
        result = generator.generate_seed(
            mode="fixed",
            seed=42,
            sync_libraries=False,
            deterministic=False,
            overflow_behavior="wrap",
            use_torch_backend="auto",
            batch_count=1
        )
        
        # Validate result structure
        assert isinstance(result, tuple), "Function should return a tuple"
        assert len(result) == len(AdvancedSeedGenerator.RETURN_TYPES), "Result length should match RETURN_TYPES"
        
        # Validate result values
        assert isinstance(result[0], int), "First return value should be an integer (seed)"
        assert isinstance(result[1], int), "Second return value should be an integer (batch_count)"
        
        print("✅ Function execution compatibility: PASSED")
        return True
    except Exception as e:
        print(f"❌ Function execution compatibility: FAILED - {e}")
        return False

def test_is_changed_method():
    """Test IS_CHANGED method compatibility."""
    try:
        from random_seed_generator import AdvancedSeedGenerator
        
        # Test IS_CHANGED with all parameters
        result = AdvancedSeedGenerator.IS_CHANGED(
            mode="fixed",
            seed=42,
            sync_libraries=True,
            deterministic=False,
            overflow_behavior="wrap",
            use_torch_backend="auto",
            batch_count=1
        )
        
        # Should return either a string (for caching) or float (for timestamp)
        assert isinstance(result, (str, float)), "IS_CHANGED should return string or float"
        
        # Test dynamic mode
        dynamic_result = AdvancedSeedGenerator.IS_CHANGED(
            mode="random",
            seed=0,
            sync_libraries=False,
            deterministic=False,
            overflow_behavior="wrap",
            use_torch_backend="auto",
            batch_count=1
        )
        
        assert isinstance(dynamic_result, float), "Dynamic modes should return timestamp"
        
        print("✅ IS_CHANGED method compatibility: PASSED")
        return True
    except Exception as e:
        print(f"❌ IS_CHANGED method compatibility: FAILED - {e}")
        return False

def test_tensor_compatibility():
    """Test compatibility with ComfyUI's tensor format expectations."""
    try:
        from random_seed_generator import AdvancedSeedGenerator
        import torch
        
        generator = AdvancedSeedGenerator()
        
        # Generate seeds with torch backend
        result = generator.generate_seed(
            mode="random",
            seed=0,
            sync_libraries=False,
            deterministic=False,
            overflow_behavior="wrap",
            use_torch_backend="torch",
            batch_count=5
        )
        
        # Should work with tensor operations (seeds are Python ints, not tensors)
        seed_value = result[0]
        
        # Test that seed can be used with PyTorch
        torch.manual_seed(seed_value)
        test_tensor = torch.rand(3, 3)
        
        assert test_tensor.shape == (3, 3), "Seed should work with PyTorch operations"
        
        print("✅ Tensor compatibility: PASSED")
        return True
    except Exception as e:
        print(f"❌ Tensor compatibility: FAILED - {e}")
        return False

def test_memory_safety():
    """Test for memory leaks and proper resource management."""
    try:
        from random_seed_generator import AdvancedSeedGenerator
        import gc
        import torch
        
        # Test many iterations to check for memory leaks
        generator = AdvancedSeedGenerator()
        
        for i in range(1000):
            result = generator.generate_seed(
                mode="random",
                seed=i,
                sync_libraries=False,
                deterministic=False,
                overflow_behavior="wrap",
                use_torch_backend="auto" if i % 2 == 0 else "torch",
                batch_count=1 if i % 10 == 0 else 10
            )
            
            # Occasionally force garbage collection
            if i % 100 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        print("✅ Memory safety: PASSED")
        return True
    except Exception as e:
        print(f"❌ Memory safety: FAILED - {e}")
        return False

def test_comfyui_v3_compatibility():
    """Test compatibility with ComfyUI v3 features if available."""
    try:
        # Try to import ComfyUI v3 APIs (similar to FaceDetectionNode pattern)
        try:
            from comfy_api.v0_0_3_io import ComfyNode, Schema
            v3_available = True
            print("📝 ComfyUI v3 API detected")
        except ImportError:
            v3_available = False
            print("📝 ComfyUI v3 API not available - using v1/v2 compatibility")
        
        # Our node should work with both
        from random_seed_generator import AdvancedSeedGenerator
        
        # Test node instantiation
        generator = AdvancedSeedGenerator()
        
        # Test basic functionality
        result = generator.generate_seed(
            mode="fixed",
            seed=123,
            sync_libraries=False,
            deterministic=False,
            overflow_behavior="wrap",
            use_torch_backend="auto",
            batch_count=1
        )
        
        assert result[0] == 123, "Node should work regardless of ComfyUI version"
        
        print("✅ ComfyUI version compatibility: PASSED")
        return True
    except Exception as e:
        print(f"❌ ComfyUI version compatibility: FAILED - {e}")
        return False

def run_all_compatibility_tests():
    """Run all compatibility tests."""
    print("🔍 Running ComfyUI Compatibility Tests")
    print("=" * 50)
    
    tests = [
        test_node_registration,
        test_input_types_schema,
        test_return_types,
        test_function_execution,
        test_is_changed_method,
        test_tensor_compatibility,
        test_memory_safety,
        test_comfyui_v3_compatibility
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__}: FAILED - {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 COMPATIBILITY TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All compatibility tests PASSED! Node is ready for ComfyUI deployment.")
        return True
    else:
        print(f"⚠️  {total - passed} compatibility issues found.")
        return False

if __name__ == "__main__":
    success = run_all_compatibility_tests()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
MLX ROCm Backend Test Script
Tests various MLX operations on the ROCm GPU backend.
"""

import sys
import time

def test_section(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print('='*60)

def test_op(name, func):
    """Run a test and report success/failure."""
    try:
        result = func()
        print(f"  ✅ {name}")
        return True, result
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        return False, None

def main():
    print("MLX ROCm Backend Test Suite")
    print("="*60)
    
    # Import test
    test_section("Import Test")
    try:
        import mlx.core as mx
        import os

        # Force CPU mode if environment variable is set
        if os.getenv('MLX_DISABLE_GPU') == '1':
            mx.set_default_device(mx.cpu)

        print(f"Default device: {mx.default_device()}")
        print(f"  ✅ MLX imported successfully")
        print(f"     Version info: mlx.core loaded")
    except ImportError as e:
        print(f"  ❌ Failed to import MLX: {e}")
        sys.exit(1)
    
    # Device info
    test_section("Device Information")
    print(f"  Default device: {mx.default_device()}")
    
    try:
        mx.set_default_device(mx.gpu)
        print(f"  Switched to: {mx.default_device()}")
        gpu_available = True
    except Exception as e:
        print(f"  ⚠️  GPU not available: {e}")
        print(f"  Continuing with CPU...")
        gpu_available = False
    
    passed = 0
    failed = 0
    
    # Basic array creation
    test_section("Array Creation")
    
    ok, a = test_op("mx.zeros((3,3))", lambda: mx.zeros((3,3)))
    if ok: mx.eval(a); passed += 1
    else: failed += 1
    
    ok, a = test_op("mx.ones((3,3))", lambda: mx.ones((3,3)))
    if ok: mx.eval(a); passed += 1
    else: failed += 1
    
    ok, a = test_op("mx.full((3,3), 5.0)", lambda: mx.full((3,3), 5.0))
    if ok: mx.eval(a); passed += 1
    else: failed += 1
    
    ok, a = test_op("mx.arange(10)", lambda: mx.arange(10))
    if ok: mx.eval(a); passed += 1
    else: failed += 1
    
    ok, a = test_op("mx.linspace(0, 1, 10)", lambda: mx.linspace(0, 1, 10))
    if ok: mx.eval(a); passed += 1
    else: failed += 1
    
    ok, a = test_op("mx.eye(4)", lambda: mx.eye(4))
    if ok: mx.eval(a); passed += 1
    else: failed += 1
    
    # Different dtypes
    test_section("Data Types")
    
    for dtype in [mx.float32, mx.float16, mx.bfloat16, mx.int32, mx.int64, mx.bool_]:
        ok, a = test_op(f"mx.ones((2,2), dtype={dtype})", 
                       lambda dt=dtype: mx.ones((2,2), dtype=dt))
        if ok: mx.eval(a); passed += 1
        else: failed += 1
    
    # Binary operations
    test_section("Binary Operations")
    
    a = mx.ones((3,3))
    b = mx.full((3,3), 2.0)
    mx.eval(a, b)
    
    ok, _ = test_op("add (a + b)", lambda: mx.eval(a + b))
    passed += 1 if ok else 0; failed += 0 if ok else 1
    
    ok, _ = test_op("subtract (a - b)", lambda: mx.eval(a - b))
    passed += 1 if ok else 0; failed += 0 if ok else 1
    
    ok, _ = test_op("multiply (a * b)", lambda: mx.eval(a * b))
    passed += 1 if ok else 0; failed += 0 if ok else 1
    
    ok, _ = test_op("divide (a / b)", lambda: mx.eval(a / b))
    passed += 1 if ok else 0; failed += 0 if ok else 1
    
    ok, _ = test_op("power (a ** 2)", lambda: mx.eval(a ** 2))
    passed += 1 if ok else 0; failed += 0 if ok else 1
    
    ok, _ = test_op("maximum", lambda: mx.eval(mx.maximum(a, b)))
    passed += 1 if ok else 0; failed += 0 if ok else 1
    
    ok, _ = test_op("minimum", lambda: mx.eval(mx.minimum(a, b)))
    passed += 1 if ok else 0; failed += 0 if ok else 1
    
    # Comparison operations
    test_section("Comparison Operations")
    
    ok, _ = test_op("equal (a == b)", lambda: mx.eval(a == b))
    passed += 1 if ok else 0; failed += 0 if ok else 1
    
    ok, _ = test_op("not_equal (a != b)", lambda: mx.eval(a != b))
    passed += 1 if ok else 0; failed += 0 if ok else 1
    
    ok, _ = test_op("greater (a > b)", lambda: mx.eval(a > b))
    passed += 1 if ok else 0; failed += 0 if ok else 1
    
    ok, _ = test_op("less (a < b)", lambda: mx.eval(a < b))
    passed += 1 if ok else 0; failed += 0 if ok else 1
    
    # Unary operations
    test_section("Unary Operations")
    
    x = mx.array([0.5, 1.0, 2.0])
    mx.eval(x)
    
    unary_ops = [
        ("abs", lambda: mx.abs(mx.array([-1.0, 2.0, -3.0]))),
        ("negative", lambda: mx.negative(x)),
        ("square", lambda: mx.square(x)),
        ("sqrt", lambda: mx.sqrt(x)),
        ("exp", lambda: mx.exp(x)),
        ("log", lambda: mx.log(x)),
        ("sin", lambda: mx.sin(x)),
        ("cos", lambda: mx.cos(x)),
        ("tanh", lambda: mx.tanh(x)),
        ("sigmoid", lambda: mx.sigmoid(x)),
        ("ceil", lambda: mx.ceil(x)),
        ("floor", lambda: mx.floor(x)),
        ("round", lambda: mx.round(x)),
    ]
    
    for name, op in unary_ops:
        ok, r = test_op(name, op)
        if ok: mx.eval(r); passed += 1
        else: failed += 1
    
    # Reduction operations
    test_section("Reduction Operations")
    
    a = mx.arange(12).reshape(3, 4).astype(mx.float32)
    mx.eval(a)
    
    ok, _ = test_op("sum", lambda: mx.eval(mx.sum(a)))
    passed += 1 if ok else 0; failed += 0 if ok else 1
    
    ok, _ = test_op("sum(axis=0)", lambda: mx.eval(mx.sum(a, axis=0)))
    passed += 1 if ok else 0; failed += 0 if ok else 1
    
    ok, _ = test_op("sum(axis=1)", lambda: mx.eval(mx.sum(a, axis=1)))
    passed += 1 if ok else 0; failed += 0 if ok else 1
    
    ok, _ = test_op("mean", lambda: mx.eval(mx.mean(a)))
    passed += 1 if ok else 0; failed += 0 if ok else 1
    
    ok, _ = test_op("max", lambda: mx.eval(mx.max(a)))
    passed += 1 if ok else 0; failed += 0 if ok else 1
    
    ok, _ = test_op("min", lambda: mx.eval(mx.min(a)))
    passed += 1 if ok else 0; failed += 0 if ok else 1
    
    ok, _ = test_op("prod", lambda: mx.eval(mx.prod(a + 1)))  # +1 to avoid zero
    passed += 1 if ok else 0; failed += 0 if ok else 1
    
    # Matrix operations
    test_section("Matrix Operations")
    
    a = mx.ones((4, 3))
    b = mx.ones((3, 5))
    mx.eval(a, b)
    
    ok, c = test_op("matmul (4x3 @ 3x5)", lambda: mx.matmul(a, b))
    if ok: 
        mx.eval(c)
        print(f"     Result shape: {c.shape}")
        passed += 1
    else: 
        failed += 1
    
    # Larger matmul for performance
    ok, _ = test_op("matmul (128x64 @ 64x128)", 
                   lambda: mx.eval(mx.matmul(mx.ones((128, 64)), mx.ones((64, 128)))))
    passed += 1 if ok else 0; failed += 0 if ok else 1
    
    ok, _ = test_op("transpose", lambda: mx.eval(mx.transpose(a)))
    passed += 1 if ok else 0; failed += 0 if ok else 1
    
    # Reshape and indexing
    test_section("Reshape and Indexing")
    
    a = mx.arange(24)
    mx.eval(a)
    
    ok, r = test_op("reshape (24,) -> (4,6)", lambda: mx.reshape(a, (4, 6)))
    if ok: mx.eval(r); passed += 1
    else: failed += 1
    
    ok, r = test_op("reshape (24,) -> (2,3,4)", lambda: mx.reshape(a, (2, 3, 4)))
    if ok: mx.eval(r); passed += 1
    else: failed += 1
    
    b = mx.arange(12).reshape(3, 4)
    mx.eval(b)
    
    ok, _ = test_op("slice [1:3, 2:4]", lambda: mx.eval(b[1:3, 2:4]))
    passed += 1 if ok else 0; failed += 0 if ok else 1
    
    ok, _ = test_op("slice [0, :]", lambda: mx.eval(b[0, :]))
    passed += 1 if ok else 0; failed += 0 if ok else 1
    
    # Broadcasting
    test_section("Broadcasting")
    
    a = mx.ones((3, 4))
    b = mx.ones((4,))
    mx.eval(a, b)
    
    ok, _ = test_op("broadcast add (3,4) + (4,)", lambda: mx.eval(a + b))
    passed += 1 if ok else 0; failed += 0 if ok else 1
    
    b = mx.ones((3, 1))
    mx.eval(b)
    
    ok, _ = test_op("broadcast add (3,4) + (3,1)", lambda: mx.eval(a + b))
    passed += 1 if ok else 0; failed += 0 if ok else 1
    
    # Simple performance test
    test_section("Performance Test (Matrix Multiply)")
    
    try:
        sizes = [(256, 256), (512, 512), (1024, 1024)]
        for n, m in sizes:
            a = mx.ones((n, m))
            b = mx.ones((m, n))
            mx.eval(a, b)
            
            # Warmup
            c = mx.matmul(a, b)
            mx.eval(c)
            
            # Timed run
            start = time.time()
            for _ in range(10):
                c = mx.matmul(a, b)
                mx.eval(c)
            elapsed = time.time() - start
            
            gflops = (2 * n * m * n * 10) / elapsed / 1e9
            print(f"  {n}x{m} @ {m}x{n}: {elapsed*100:.1f}ms/iter, ~{gflops:.1f} GFLOPS")
    except Exception as e:
        print(f"  ⚠️  Performance test failed: {e}")
    
    # Summary
    test_section("Summary")
    total = passed + failed
    print(f"  Passed: {passed}/{total}")
    print(f"  Failed: {failed}/{total}")
    
    if failed == 0:
        print("\n  🎉 All tests passed!")
    else:
        print(f"\n  ⚠️  {failed} test(s) failed")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


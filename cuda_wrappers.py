"""
Python wrappers for CUDA kernels using ctypes and raw CUDA runtime API
No PyTorch dependency
"""

import ctypes
import numpy as np
import os

try:
    cuda_paths = [
        '/usr/local/cuda/lib64/libcudart.so',
        '/usr/local/cuda/lib/libcudart.so',
        '/opt/cuda/lib64/libcudart.so',
        '/opt/cuda/lib/libcudart.so',
    ]

    cudart = None
    for path in cuda_paths:
        if os.path.exists(path):
            cudart = ctypes.CDLL(path)
            break

    if cudart is None:
        try:
            cudart = ctypes.CDLL('libcudart.so')
        except:
            try:
                cudart = ctypes.CDLL('libcudart.dylib')
            except:
                raise RuntimeError("Could not find CUDA runtime library. Please ensure CUDA is installed.")

    cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cudart.cudaMalloc.restype = ctypes.c_int

    cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    cudart.cudaMemcpy.restype = ctypes.c_int

    cudart.cudaFree.argtypes = [ctypes.c_void_p]
    cudart.cudaFree.restype = ctypes.c_int

    cudart.cudaDeviceSynchronize.argtypes = []
    cudart.cudaDeviceSynchronize.restype = ctypes.c_int

    cudart.cudaGetLastError.argtypes = []
    cudart.cudaGetLastError.restype = ctypes.c_int

    cudart.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    cudart.cudaMemset.restype = ctypes.c_int

    cudart.cudaDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int
    ]
    cudart.cudaDeviceGetAttribute.restype = ctypes.c_int

    # CUDA event API for kernel-only GPU timing (replaces host-side time.time()).
    cudart.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cudart.cudaEventCreate.restype = ctypes.c_int
    cudart.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    cudart.cudaEventRecord.restype = ctypes.c_int
    cudart.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
    cudart.cudaEventSynchronize.restype = ctypes.c_int
    cudart.cudaEventElapsedTime.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p
    ]
    cudart.cudaEventElapsedTime.restype = ctypes.c_int
    cudart.cudaEventDestroy.argtypes = [ctypes.c_void_p]
    cudart.cudaEventDestroy.restype = ctypes.c_int

    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2
    # cudaDevAttrMaxSharedMemoryPerBlockOptin
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97

except Exception as e:
    raise RuntimeError(f"Failed to load CUDA runtime: {e}")

try:
    lib = ctypes.CDLL('./kernels.so')
except OSError:
    raise RuntimeError("Failed to load kernels.so. Please compile the CUDA kernels first using 'make'.")

lib.launch_naive_kernel.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p
]
lib.launch_naive_kernel.restype = None

lib.launch_parallel_kernel.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p
]
lib.launch_parallel_kernel.restype = None

lib.launch_tiled_kernel.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p
]
lib.launch_tiled_kernel.restype = None

lib.launch_optimized_kernel.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p
]
lib.launch_optimized_kernel.restype = None

# WMMA (fp16 tensor-core) kernel: half* Q/K/V (passed as void*), float* output.
try:
    lib.launch_wmma_kernel.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.launch_wmma_kernel.restype = None
    HAS_WMMA = True
except AttributeError:
    HAS_WMMA = False  # kernels.so built without kernel_wmma.cu

# Split-K (key-parallel) kernel: Q/K/V/out + po/pm/pl scratch + num_splits.
try:
    lib.launch_splitk_kernel.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.launch_splitk_kernel.restype = None
    HAS_SPLITK = True
except AttributeError:
    HAS_SPLITK = False  # kernels.so built without kernel_splitk.cu

class DesignLimited(RuntimeError):
    """Raised when a kernel cannot run a config by design (not a bug).

    Used by the warp-parallel kernel: it materializes the full score row in
    shared memory, so N_input is capped by the device's per-block shared-memory
    limit. Beyond that we skip cleanly and the benchmark records the config as
    'design-limited' -- which is itself the Perceiver-scaling result.
    """


def _max_smem_optin(device=0):
    """Device max opt-in dynamic shared memory per block, in bytes."""
    val = ctypes.c_int(0)
    err = cudart.cudaDeviceGetAttribute(
        ctypes.byref(val), cudaDevAttrMaxSharedMemoryPerBlockOptin, device)
    if err != 0 or val.value <= 0:
        return 48 * 1024  # conservative default (static smem limit)
    return val.value


def warp_smem_bytes(N_input):
    """Shared memory the warp kernel needs for a given N_input."""
    return (4 * N_input + 4) * 4


def _allocate_gpu_memory(size_bytes):
    """Allocate GPU memory."""
    ptr = ctypes.c_void_p()
    err = cudart.cudaMalloc(ctypes.byref(ptr), size_bytes)
    if err != 0:
        raise RuntimeError(f"cudaMalloc failed with error {err}")
    return ptr

def _copy_to_gpu(host_data, gpu_ptr, size_bytes):
    """Copy data from host to GPU."""
    host_ptr = host_data.ctypes.data_as(ctypes.c_void_p)
    err = cudart.cudaMemcpy(gpu_ptr, host_ptr, size_bytes, cudaMemcpyHostToDevice)
    if err != 0:
        raise RuntimeError(f"cudaMemcpy H2D failed with error {err}")

def _copy_from_gpu(gpu_ptr, host_data, size_bytes):
    """Copy data from GPU to host."""
    host_ptr = host_data.ctypes.data_as(ctypes.c_void_p)
    err = cudart.cudaMemcpy(host_ptr, gpu_ptr, size_bytes, cudaMemcpyDeviceToHost)
    if err != 0:
        raise RuntimeError(f"cudaMemcpy D2H failed with error {err}")

def _free_gpu_memory(ptr):
    """Free GPU memory."""
    cudart.cudaFree(ptr)

def _run_kernel(kernel_func, Q, K, V, batch_size, N_latent, N_input, D):
    """Helper function to run a CUDA kernel with batch support."""

    Q = np.ascontiguousarray(Q, dtype=np.float32)
    K = np.ascontiguousarray(K, dtype=np.float32)
    V = np.ascontiguousarray(V, dtype=np.float32)

    Q_size = Q.size * 4
    K_size = K.size * 4
    V_size = V.size * 4
    output_size = batch_size * N_latent * D * 4

    Q_gpu = _allocate_gpu_memory(Q_size)
    K_gpu = _allocate_gpu_memory(K_size)
    V_gpu = _allocate_gpu_memory(V_size)
    output_gpu = _allocate_gpu_memory(output_size)

    try:
        err = cudart.cudaMemset(output_gpu, 0, output_size)
        if err != 0:
            raise RuntimeError(f"cudaMemset failed with error {err}")

        _copy_to_gpu(Q, Q_gpu, Q_size)
        _copy_to_gpu(K, K_gpu, K_size)
        _copy_to_gpu(V, V_gpu, V_size)

        kernel_func(
            ctypes.cast(Q_gpu, ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(K_gpu, ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(V_gpu, ctypes.POINTER(ctypes.c_float)),
            ctypes.cast(output_gpu, ctypes.POINTER(ctypes.c_float)),
            batch_size,
            N_latent,
            N_input,
            D,
            ctypes.c_void_p(0)
        )

        err = cudart.cudaGetLastError()
        if err != 0:
            raise RuntimeError(f"Kernel launch failed with CUDA error {err}. "
                            f"This usually indicates a kernel configuration problem, "
                            f"invalid memory access, or architecture incompatibility.")

        err = cudart.cudaDeviceSynchronize()
        if err != 0:
            err_detail = cudart.cudaGetLastError()
            raise RuntimeError(f"[ERROR] cudaDeviceSynchronize failed with error {err}, "
                            f"last CUDA error: {err_detail}. "
                            f"This usually indicates a kernel execution error, "
                            f"such as invalid memory access or illegal instruction.")

        output = np.zeros((batch_size, N_latent, D), dtype=np.float32)
        _copy_from_gpu(output_gpu, output, output_size)


        return output
    finally:
        _free_gpu_memory(Q_gpu)
        _free_gpu_memory(K_gpu)
        _free_gpu_memory(V_gpu)
        _free_gpu_memory(output_gpu)

def run_naive_kernel(Q, K, V, N_latent, N_input, D):
    """Run naive CUDA kernel (baseline - one thread per output element)."""
    if len(Q.shape) == 3:
        batch_size = Q.shape[0]
    else:
        batch_size = 1
        Q = np.expand_dims(Q, axis=0).astype(np.float32)
        K = np.expand_dims(K, axis=0).astype(np.float32)
        V = np.expand_dims(V, axis=0).astype(np.float32)

    return _run_kernel(lib.launch_naive_kernel, Q, K, V, batch_size, N_latent, N_input, D)

def run_warp_parallel_kernel(Q, K, V, N_latent, N_input, D):
    """Run warp-parallel CUDA kernel (warp-level cooperation).

    Raises DesignLimited when N_input exceeds what fits in the device's
    per-block shared memory -- the warp kernel materializes the full score row,
    so this is a hard design limit, not a failure.
    """
    needed = warp_smem_bytes(N_input)
    avail = _max_smem_optin()
    if needed > avail:
        raise DesignLimited(
            f"warp kernel needs {needed/1024:.1f} KB shared memory for "
            f"N_input={N_input} but device allows {avail/1024:.1f} KB/block "
            f"(materializes full score row; use tiled/vectorized instead)")
    if len(Q.shape) == 3:
        batch_size = Q.shape[0]
    else:
        batch_size = 1
        Q = np.expand_dims(Q, axis=0).astype(np.float32)
        K = np.expand_dims(K, axis=0).astype(np.float32)
        V = np.expand_dims(V, axis=0).astype(np.float32)

    return _run_kernel(lib.launch_parallel_kernel, Q, K, V, batch_size, N_latent, N_input, D)

def run_tiled_kernel(Q, K, V, N_latent, N_input, D):
    """Run tiled CUDA kernel (shared memory tiling)."""
    if len(Q.shape) == 3:
        batch_size = Q.shape[0]
    else:
        batch_size = 1
        Q = np.expand_dims(Q, axis=0).astype(np.float32)
        K = np.expand_dims(K, axis=0).astype(np.float32)
        V = np.expand_dims(V, axis=0).astype(np.float32)

    return _run_kernel(lib.launch_tiled_kernel, Q, K, V, batch_size, N_latent, N_input, D)

def run_vectorized_kernel(Q, K, V, N_latent, N_input, D):
    """Run vectorized CUDA kernel (online softmax + vectorization)."""
    if len(Q.shape) == 3:
        batch_size = Q.shape[0]
    else:
        batch_size = 1
        Q = np.expand_dims(Q, axis=0).astype(np.float32)
        K = np.expand_dims(K, axis=0).astype(np.float32)
        V = np.expand_dims(V, axis=0).astype(np.float32)

    return _run_kernel(lib.launch_optimized_kernel, Q, K, V, batch_size, N_latent, N_input, D)


LAUNCHERS = {
    'naive': lib.launch_naive_kernel,
    'warp': lib.launch_parallel_kernel,
    'tiled': lib.launch_tiled_kernel,
    'vectorized': lib.launch_optimized_kernel,
}


def _prep(arr, batch_size):
    """Normalize Q/K/V to a contiguous fp32 [batch_size, n, D] array."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.shape[0] == 1 and batch_size > 1:
        arr = np.repeat(arr, batch_size, axis=0)
    return np.ascontiguousarray(arr, dtype=np.float32)


def benchmark_kernel_events(impl, Q, K, V, batch_size, N_latent, N_input, D,
                            warmup=20, iters=100):
    """Kernel-only GPU timing via CUDA events.

    Allocates device buffers and copies Q/K/V to the GPU ONCE, then times only
    the kernel launches with the data resident -- excluding cudaMalloc, the
    H2D/D2H copies, cudaFree, and Python overhead that dominate the legacy
    time.time() path. Returns (output, mean_ms, std_ms).

    Raises DesignLimited for the warp kernel when N_input exceeds the device's
    per-block shared memory.
    """
    if impl == 'warp':
        needed = warp_smem_bytes(N_input)
        avail = _max_smem_optin()
        if needed > avail:
            raise DesignLimited(
                f"warp needs {needed/1024:.1f} KB shared memory for "
                f"N_input={N_input}, device allows {avail/1024:.1f} KB/block")
    launch_fn = LAUNCHERS[impl]

    Q = _prep(Q, batch_size)
    K = _prep(K, batch_size)
    V = _prep(V, batch_size)

    Q_size, K_size, V_size = Q.size * 4, K.size * 4, V.size * 4
    out_size = batch_size * N_latent * D * 4

    Q_gpu = _allocate_gpu_memory(Q_size)
    K_gpu = _allocate_gpu_memory(K_size)
    V_gpu = _allocate_gpu_memory(V_size)
    out_gpu = _allocate_gpu_memory(out_size)
    start_ev = ctypes.c_void_p()
    stop_ev = ctypes.c_void_p()
    cudart.cudaEventCreate(ctypes.byref(start_ev))
    cudart.cudaEventCreate(ctypes.byref(stop_ev))

    try:
        cudart.cudaMemset(out_gpu, 0, out_size)
        _copy_to_gpu(Q, Q_gpu, Q_size)
        _copy_to_gpu(K, K_gpu, K_size)
        _copy_to_gpu(V, V_gpu, V_size)

        qp = ctypes.cast(Q_gpu, ctypes.POINTER(ctypes.c_float))
        kp = ctypes.cast(K_gpu, ctypes.POINTER(ctypes.c_float))
        vp = ctypes.cast(V_gpu, ctypes.POINTER(ctypes.c_float))
        op = ctypes.cast(out_gpu, ctypes.POINTER(ctypes.c_float))
        null_stream = ctypes.c_void_p(0)

        def _launch():
            launch_fn(qp, kp, vp, op, batch_size, N_latent, N_input, D, null_stream)

        for _ in range(warmup):
            _launch()
        if cudart.cudaDeviceSynchronize() != 0:
            raise RuntimeError(
                f"kernel '{impl}' failed during warmup (cuda error "
                f"{cudart.cudaGetLastError()}); likely invalid config for "
                f"N_input={N_input}, N_latent={N_latent}")

        times = []
        ms = ctypes.c_float(0.0)
        for _ in range(iters):
            cudart.cudaEventRecord(start_ev, null_stream)
            _launch()
            cudart.cudaEventRecord(stop_ev, null_stream)
            cudart.cudaEventSynchronize(stop_ev)
            cudart.cudaEventElapsedTime(ctypes.byref(ms), start_ev, stop_ev)
            times.append(ms.value)

        output = np.zeros((batch_size, N_latent, D), dtype=np.float32)
        _copy_from_gpu(out_gpu, output, out_size)

        import statistics
        mean_ms = statistics.mean(times)
        std_ms = statistics.pstdev(times) if len(times) > 1 else 0.0
        return output, mean_ms, std_ms
    finally:
        cudart.cudaEventDestroy(start_ev)
        cudart.cudaEventDestroy(stop_ev)
        _free_gpu_memory(Q_gpu)
        _free_gpu_memory(K_gpu)
        _free_gpu_memory(V_gpu)
        _free_gpu_memory(out_gpu)


def choose_num_splits(N_latent, N_input, batch_size, sm_count=142):
    """Pick num_splits so the partial-kernel grid oversubscribes the GPU.

    Base blocks (no split) = ceil(Nl/4)*batch. Aim for ~4x sm_count blocks,
    capped so each split still holds >= ~16 keys.
    """
    import math
    base = math.ceil(N_latent / 4) * batch_size
    target = max(1, (4 * sm_count) // max(1, base))
    return max(1, min(N_input // 16, target)) or 1


def benchmark_splitk_events(Q, K, V, batch_size, N_latent, N_input, D,
                            warmup=20, iters=100, num_splits=None, sm_count=142):
    """CUDA-event timing for the split-K (key-parallel) kernel.

    Allocates Q/K/V/out AND the partial scratch (po/pm/pl) once, then times the
    partial+combine launches. Returns (output, mean_ms, std_ms, num_splits).
    """
    if not HAS_SPLITK:
        raise RuntimeError("kernels.so has no launch_splitk_kernel (rebuild "
                           "with kernel_splitk.cu).")
    if num_splits is None:
        num_splits = choose_num_splits(N_latent, N_input, batch_size, sm_count)

    Q = _prep(Q, batch_size); K = _prep(K, batch_size); V = _prep(V, batch_size)
    Q_size, K_size, V_size = Q.size * 4, K.size * 4, V.size * 4
    out_size = batch_size * N_latent * D * 4
    po_size = batch_size * N_latent * num_splits * D * 4
    pml_size = batch_size * N_latent * num_splits * 4

    Q_gpu = _allocate_gpu_memory(Q_size); K_gpu = _allocate_gpu_memory(K_size)
    V_gpu = _allocate_gpu_memory(V_size); out_gpu = _allocate_gpu_memory(out_size)
    po_gpu = _allocate_gpu_memory(po_size)
    pm_gpu = _allocate_gpu_memory(pml_size); pl_gpu = _allocate_gpu_memory(pml_size)
    start_ev = ctypes.c_void_p(); stop_ev = ctypes.c_void_p()
    cudart.cudaEventCreate(ctypes.byref(start_ev))
    cudart.cudaEventCreate(ctypes.byref(stop_ev))

    try:
        cudart.cudaMemset(out_gpu, 0, out_size)
        _copy_to_gpu(Q, Q_gpu, Q_size); _copy_to_gpu(K, K_gpu, K_size)
        _copy_to_gpu(V, V_gpu, V_size)
        fp = lambda p: ctypes.cast(p, ctypes.POINTER(ctypes.c_float))
        null_stream = ctypes.c_void_p(0)

        def _launch():
            lib.launch_splitk_kernel(
                fp(Q_gpu), fp(K_gpu), fp(V_gpu), fp(out_gpu),
                fp(po_gpu), fp(pm_gpu), fp(pl_gpu),
                batch_size, N_latent, N_input, D, num_splits, null_stream)

        for _ in range(warmup):
            _launch()
        if cudart.cudaDeviceSynchronize() != 0:
            raise RuntimeError(
                f"split-K failed during warmup (cuda err "
                f"{cudart.cudaGetLastError()}); Nl={N_latent} Ni={N_input} "
                f"splits={num_splits}")

        times = []; ms = ctypes.c_float(0.0)
        for _ in range(iters):
            cudart.cudaEventRecord(start_ev, null_stream)
            _launch()
            cudart.cudaEventRecord(stop_ev, null_stream)
            cudart.cudaEventSynchronize(stop_ev)
            cudart.cudaEventElapsedTime(ctypes.byref(ms), start_ev, stop_ev)
            times.append(ms.value)

        output = np.zeros((batch_size, N_latent, D), dtype=np.float32)
        _copy_from_gpu(out_gpu, output, out_size)
        import statistics
        return (output, statistics.mean(times),
                (statistics.pstdev(times) if len(times) > 1 else 0.0), num_splits)
    finally:
        cudart.cudaEventDestroy(start_ev); cudart.cudaEventDestroy(stop_ev)
        for p in (Q_gpu, K_gpu, V_gpu, out_gpu, po_gpu, pm_gpu, pl_gpu):
            _free_gpu_memory(p)


def benchmark_wmma_events(Q, K, V, batch_size, N_latent, N_input, D,
                          warmup=20, iters=100):
    """CUDA-event timing for the fp16 WMMA kernel.

    Takes fp32 Q/K/V (same references the other kernels use), casts to fp16,
    pre-scales Q by 1/sqrt(D) (the kernel expects pre-scaled Q so the score
    scaling folds into the load), runs the tensor-core kernel, and returns
    (fp32 output, mean_ms, std_ms). Requires Nl % 16 == 0 and Ni % 16 == 0.
    """
    if not HAS_WMMA:
        raise RuntimeError("kernels.so has no launch_wmma_kernel (rebuild with "
                           "kernel_wmma.cu and arch >= sm_70).")
    if N_latent % 16 or N_input % 16:
        raise DesignLimited(
            f"WMMA needs Nl%16==0 and Ni%16==0 (got Nl={N_latent}, Ni={N_input})")

    Q = _prep(Q, batch_size).astype(np.float32) / np.sqrt(D).astype(np.float32)
    K = _prep(K, batch_size)
    V = _prep(V, batch_size)
    Qh = np.ascontiguousarray(Q, dtype=np.float16)
    Kh = np.ascontiguousarray(K, dtype=np.float16)
    Vh = np.ascontiguousarray(V, dtype=np.float16)

    Q_size, K_size, V_size = Qh.size * 2, Kh.size * 2, Vh.size * 2  # fp16 = 2 B
    out_size = batch_size * N_latent * D * 4                         # fp32 out

    Q_gpu = _allocate_gpu_memory(Q_size)
    K_gpu = _allocate_gpu_memory(K_size)
    V_gpu = _allocate_gpu_memory(V_size)
    out_gpu = _allocate_gpu_memory(out_size)
    start_ev = ctypes.c_void_p(); stop_ev = ctypes.c_void_p()
    cudart.cudaEventCreate(ctypes.byref(start_ev))
    cudart.cudaEventCreate(ctypes.byref(stop_ev))

    try:
        cudart.cudaMemset(out_gpu, 0, out_size)
        _copy_to_gpu(Qh, Q_gpu, Q_size)
        _copy_to_gpu(Kh, K_gpu, K_size)
        _copy_to_gpu(Vh, V_gpu, V_size)
        op = ctypes.cast(out_gpu, ctypes.POINTER(ctypes.c_float))
        null_stream = ctypes.c_void_p(0)

        def _launch():
            lib.launch_wmma_kernel(Q_gpu, K_gpu, V_gpu, op,
                                   batch_size, N_latent, N_input, D, null_stream)

        for _ in range(warmup):
            _launch()
        if cudart.cudaDeviceSynchronize() != 0:
            raise RuntimeError(
                f"WMMA kernel failed during warmup (cuda err "
                f"{cudart.cudaGetLastError()}); Nl={N_latent} Ni={N_input}")

        times = []
        ms = ctypes.c_float(0.0)
        for _ in range(iters):
            cudart.cudaEventRecord(start_ev, null_stream)
            _launch()
            cudart.cudaEventRecord(stop_ev, null_stream)
            cudart.cudaEventSynchronize(stop_ev)
            cudart.cudaEventElapsedTime(ctypes.byref(ms), start_ev, stop_ev)
            times.append(ms.value)

        output = np.zeros((batch_size, N_latent, D), dtype=np.float32)
        _copy_from_gpu(out_gpu, output, out_size)
        import statistics
        return output, statistics.mean(times), (
            statistics.pstdev(times) if len(times) > 1 else 0.0)
    finally:
        cudart.cudaEventDestroy(start_ev)
        cudart.cudaEventDestroy(stop_ev)
        _free_gpu_memory(Q_gpu); _free_gpu_memory(K_gpu)
        _free_gpu_memory(V_gpu); _free_gpu_memory(out_gpu)

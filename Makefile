NVCC = nvcc


NVCC_FLAGS = -O3 -Xcompiler -fPIC -shared --compiler-options '-fPIC'


KERNEL_SOURCES = kernel_naive.cu kernel_warp_parallel.cu kernel_tiled.cu kernel_vectorized.cu kernel_wmma.cu kernel_splitk.cu kernel_wmma_splitk.cu

TARGET = kernels.so

all: $(TARGET)

$(TARGET): $(KERNEL_SOURCES)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(KERNEL_SOURCES)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean

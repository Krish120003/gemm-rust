import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import time


N = 2048

A = np.random.rand(N, N).astype(np.float32)
B = np.random.rand(N, N).astype(np.float32)

flop = N * N * 2 * N

avg_time = 0
runs = 10
for i in range(runs):
    start = time.monotonic()
    C = A @ B
    end = time.monotonic()
    avg_time += end - start

avg_time /= runs

print("GFLOPS: ", flop / avg_time / 1e9)

# write matricies A,B,C to file
np.savetxt("data/A.txt", A, fmt="%f")
np.savetxt("data/B.txt", B, fmt="%f")
np.savetxt("data/C.txt", C, fmt="%f")

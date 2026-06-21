import time

import numpy as np

from accelera.src.parallelizer.parallelizer import parallelizer


def adjust_pixel(pixel):
    return 255 * ((pixel / 255) ** (1 / 2.2))


# O(H*W) = O(n^2) if H=W=n --> Time complexity of the function is O(n^2)
def gamma_correction(image):
    for i in range(len(image)):
        for j in range(len(image[i])):
            image[i][j] = adjust_pixel(image[i][j])

    return image


data_size = 100
images = [
    np.random.uniform(0, 255, (512, 512)).astype(np.float64)
    for _ in range(data_size)
]

# Accelera parallelized version
acc_res = []
start_time = time.time()
acc_results = parallelizer.parallelize(gamma_correction)
for image in images:
    acc_res.append(acc_results(image))
end_time = time.time()
print(f"Execution time Accelera: {end_time - start_time} seconds")


# Non-Accelera version for comparison
res = []
start_time = time.time()
for image in images:
    res.append(gamma_correction(image))
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")


# Correctness check
if np.allclose(acc_res, res):
    print("Validation successful: Accelera results match non-Accelera results.")
else:
    print("Validation failed: Accelera results do not match non-Accelera results.")

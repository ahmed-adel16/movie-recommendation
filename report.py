import numpy as np
import time
import matplotlib.pyplot as plt

# Array sizes
sizes = [10_000_000, 20_000_000, 50_000_000, 75_000_000, 100_000_000]

times = []
c_values = []

def linear_search(arr, key):
    for i in range(len(arr)):
        if arr[i] == key:
            return i
    return -1

for n in sizes:
    total_time = 0
    
    for _ in range(3):  # repeat 3 times
        arr = np.random.randint(0, 100, size=n)

        start = time.time()
        linear_search(arr, -1)  # worst case
        end = time.time()

        total_time += (end - start)

    avg_time = total_time / 3
    times.append(avg_time)

    c_local = avg_time / n
    c_values.append(c_local)

    print(f"n={n}, T(n)={avg_time:.6f}, c={c_local:.10e}")

# Final estimated constant c
c_final = c_values[-1]
c_avg = sum(c_values) / len(c_values)

print("\nEstimated c (largest n):", c_final)
print("Average c:", c_avg)

# Plot
plt.plot(sizes, times, marker='o')
plt.xlabel("Input Size (n)")
plt.ylabel("Time T(n) in seconds")
plt.title("Linear Search Time Complexity")
plt.grid()
plt.show()
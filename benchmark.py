import timeit
from expectation.modules.martingales import TwoSidedNormalMixture

# Create instance
mixture = TwoSidedNormalMixture(v_opt=1.0, alpha_opt=0.05)

# Benchmark
iterations = 10_000_000



totaltime = timeit.timeit('mixture.log_superMG(0.5, 1.0)',
                          globals={'mixture': mixture}, number=iterations)
time_per_call_ns = (totaltime / iterations) * 1e9

print(f"Python: {time_per_call_ns:.2f} ns/call")
print(f"Total: {totaltime:.4f} seconds for {iterations} iterations")
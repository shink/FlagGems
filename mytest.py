import time
from functools import wraps

import flag_gems
import torch
import torch.nn.functional as F


def timer(batch_size=100):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            total_time = 0.0

            for _ in range(batch_size):
                start_time = time.perf_counter()
                fn(*args, **kwargs)
                end_time = time.perf_counter()
                total_time += end_time - start_time

            avg_time = total_time / batch_size
            print(
                f"{fn.__name__} runs {batch_size} times, average costs: {avg_time:.6f}s"
            )

        return wrapper

    return decorator


@timer(1000)
def pointwise_add(use_gems=False):
    M, N, K = 1024, 1024, 1024
    A = torch.randn((M, K), dtype=torch.float16, device=flag_gems.device)
    B = torch.randn((K, N), dtype=torch.float16, device=flag_gems.device)
    if use_gems:
        with flag_gems.use_gems():
            _ = torch.mm(A, B)
    else:
        _ = torch.mm(A, B)


@timer(1000)
def reduce_sum(use_gems=False):
    M, N = 1024, 1024
    A = torch.randn((M, N), dtype=torch.float16, device=flag_gems.device)
    if use_gems:
        with flag_gems.use_gems():
            _ = torch.sum(A)
    else:
        _ = torch.sum(A)


@timer(1000)
def fa_scaled_dot_product_attention(use_gems=False):
    query, key, value = (
        torch.randn(2, 4, 1024, 64, device=flag_gems.device),
        torch.randn(2, 4, 1024, 64, device=flag_gems.device),
        torch.randn(2, 4, 1024, 64, device=flag_gems.device),
    )
    if use_gems:
        with flag_gems.use_gems():
            _ = F.scaled_dot_product_attention(
                query,
                key,
                value,
            )
    else:
        _ = F.scaled_dot_product_attention(
            query,
            key,
            value,
        )


if __name__ == "__main__":
    print(f"Runs on: {flag_gems.device}")

    pointwise_add()
    pointwise_add(use_gems=True)

    reduce_sum()
    reduce_sum(use_gems=True)

    fa_scaled_dot_product_attention()
    fa_scaled_dot_product_attention(use_gems=True)

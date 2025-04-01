import torch
import triton


@triton.jit
def add_kernels(x, y, alpha):
    return x + y * alpha


def add(x, y, alpha):
    add_kernels(x, y, alpha)


def add_test():
    def fn():
        pass

    M, N = 1024, 1024
    x = torch.randn((M, N), dtype=torch.float16, device="npu")
    y = torch.randn((M, N), dtype=torch.float16, device="npu")
    alpha = torch.ones((M, N), dtype=torch.float16, device="npu")
    res = add(x, y, alpha)
    print(res)


if __name__ == "__main__":
    add_test()

import torch
import pytest
import flag_gems


RESOLUTION = {
    torch.float16: 1e-3,
    torch.float32: 1e-6,
    torch.bfloat16: 1e-2,
}


def allclose_with_dtype(a, b, dtype, equal_nan=False):
    atol = RESOLUTION[dtype]
    maxdiff = torch.max(torch.abs(a - b))
    assert torch.allclose(
        a, b, atol=atol, rtol=1e-3, equal_nan=equal_nan
    ), f"max diff: {maxdiff}"


def closer_to_golden(golden, ref, res, tolerance=1.05):
    diff_torch = torch.sum(torch.abs(golden - ref))
    diff_triton = torch.sum(torch.abs(golden - res))
    assert (
        diff_triton < diff_torch * tolerance
    ), f"Torch diff: {diff_torch}, Triton diff: {diff_triton}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_abs(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.abs(inp)
    with flag_gems.use_gems():
        res_out = torch.abs(inp)

    allclose_with_dtype(ref_out, res_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("alpha", [0, 1, 4, -9])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_add(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.add(inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.add(inp1, inp2, alpha=alpha)

    allclose_with_dtype(ref_out, res_out, dtype)


@pytest.mark.parametrize(
    "shape_a",
    [(16, 1024, 256)],
)
@pytest.mark.parametrize(
    "shape_b",
    [(1, 256), (1, 1, 256), (16, 1, 256), (1, 1024, 256), (1024, 256)],
)
@pytest.mark.parametrize("alpha", [0, 1, 4, -9])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_add_broadcast(shape_a, shape_b, alpha, dtype):
    inp1 = torch.randn(shape_a, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape_b, dtype=dtype, device="cuda")

    ref_out = torch.add(inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.add(inp1, inp2, alpha=alpha)

    allclose_with_dtype(ref_out, res_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize(
    "scalar",
    [111.111, -999.999],
)
@pytest.mark.parametrize("alpha", [0, 1, 4, -9])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_add_tensor_scalar(shape, scalar, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = scalar

    ref_out = torch.add(inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.add(inp1, inp2, alpha=alpha)

    allclose_with_dtype(ref_out, res_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize(
    "scalar",
    [111.111, -999.999],
)
@pytest.mark.parametrize("alpha", [0, 1, 4, -9])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_add_scalar_tensor(shape, scalar, alpha, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.add(inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.add(inp1, inp2, alpha=alpha)

    allclose_with_dtype(ref_out, res_out, dtype)


@pytest.mark.parametrize(
    "M, N, K",
    [
        (256, 256, 256),
        (1024, 1024, 1024),
        (1024, 128, 2048),
        (1024, 64, 1280),
        (640, 256, 512),
    ],
)
@pytest.mark.parametrize("alpha", [1.0, 0.5])
@pytest.mark.parametrize("beta", [1.0, 0.5])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_addmm(M, N, K, alpha, beta, dtype):
    mat1 = torch.randn((M, K), dtype=dtype, device="cuda")
    mat2 = torch.randn((K, N), dtype=dtype, device="cuda")
    bias = torch.randn((N,), dtype=dtype, device="cuda")

    golden_out = torch.addmm(
        bias.to(torch.float64),
        mat1.to(torch.float64),
        mat2.to(torch.float64),
        alpha=alpha,
        beta=beta,
    )

    ref_out = torch.addmm(bias, mat1, mat2, alpha=alpha, beta=beta)
    with flag_gems.use_gems():
        res_out = torch.addmm(bias, mat1, mat2, alpha=alpha, beta=beta)

    closer_to_golden(golden_out, ref_out, res_out)


@pytest.mark.parametrize(
    "batch, M, N, K",
    [
        (1, 1024, 1024, 1024),
        (3, 1024, 1024, 2048),
        (4, 1024, 64, 1280),
        (8, 640, 256, 512),
        (16, 1024, 128, 2048),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_bmm(batch, M, N, K, dtype):
    tensor_A = torch.randn((batch, M, K), dtype=dtype, device="cuda")
    tensor_B = torch.randn((batch, K, N), dtype=dtype, device="cuda")

    golden_out = torch.bmm(tensor_A.to(torch.float64), tensor_B.to(torch.float64))

    ref_out = torch.bmm(tensor_A, tensor_B)
    with flag_gems.use_gems():
        res_out = torch.bmm(tensor_A, tensor_B)

    closer_to_golden(golden_out, ref_out, res_out)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_cumsum(shape, dtype):
    dim = 1
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    golden_out = torch.cumsum(inp.to(torch.float64), dim=dim)

    ref_out = torch.cumsum(inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.cumsum(inp, dim=dim)

    closer_to_golden(golden_out, ref_out, res_out)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_div(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.div(inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    allclose_with_dtype(ref_out, res_out, dtype, equal_nan=True)


@pytest.mark.parametrize(
    "shape_a",
    [(16, 1024, 256)],
)
@pytest.mark.parametrize(
    "shape_b",
    [(1, 256), (1, 1, 256), (16, 1, 256), (1, 1024, 256), (1024, 256)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_div_broadcast(shape_a, shape_b, dtype):
    inp1 = torch.randn(shape_a, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape_b, dtype=dtype, device="cuda")

    ref_out = torch.div(inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    allclose_with_dtype(ref_out, res_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize(
    "scalar",
    [111.111, -999.999],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_div_tensor_scalar(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = scalar

    ref_out = torch.div(inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    allclose_with_dtype(ref_out, res_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize(
    "scalar",
    [200, 100],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_div_scalar_tensor(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randint(-5, 5, shape, dtype=dtype, device="cuda")

    ref_out = torch.div(inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.div(inp1, inp2)

    allclose_with_dtype(ref_out, res_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
@pytest.mark.parametrize("p", [0.3, 0.6, 0.9])
def test_accuracy_dropout(shape, dtype, p):
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)

    ref_out = torch.nn.functional.dropout(inp, p, True)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.dropout(inp, p, True)

    num_equal = torch.sum(torch.isclose(ref_out, res_out)).item()
    exp_equal = (p * p + (1 - p) * (1 - p)) * inp.numel()
    assert (
        abs(num_equal - exp_equal) / exp_equal <= 0.05
    ), f"num_equal: {num_equal}, exp_equal: {exp_equal}, num_total: {inp.numel()}"

    out_grad = torch.randn_like(inp)
    (ref_in_grad,) = torch.autograd.grad(ref_out, inp, out_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    num_equal = torch.sum(torch.isclose(ref_in_grad, res_in_grad)).item()
    exp_equal = (p * p + (1 - p) * (1 - p)) * inp.numel()
    assert (
        abs(num_equal - exp_equal) / exp_equal <= 0.05
    ), f"num_equal: {num_equal}, exp_equal: {exp_equal}, num_total: {inp.numel()}"


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_exp(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.exp(inp)
    with flag_gems.use_gems():
        res_out = torch.exp(inp)

    allclose_with_dtype(ref_out, res_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_gelu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.nn.functional.gelu(inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.gelu(inp)

    allclose_with_dtype(ref_out, res_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(4096, i * 32) for i in range(1, 20)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_layernorm(shape, dtype):
    layer_shape = shape[1:]
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    weight = torch.randn(layer_shape, dtype=dtype, device="cuda", requires_grad=True)
    bias = torch.randn(layer_shape, dtype=dtype, device="cuda", requires_grad=True)
    eps = 1e-5

    golden_inp = inp.to(torch.float64)
    golden_weight = weight.to(torch.float64)
    golden_bias = bias.to(torch.float64)

    golden_out = torch.layer_norm(
        golden_inp,
        list(layer_shape),
        weight=golden_weight,
        bias=golden_bias,
        eps=eps,
    )
    ref_out = torch.layer_norm(
        inp, list(layer_shape), weight=weight, bias=bias, eps=eps
    )
    (res_out, res_mean, res_rstd) = flag_gems.layer_norm(
        inp, list(layer_shape), weight=weight, bias=bias, eps=eps
    )

    ref_mean = torch.mean(inp, dim=1)
    ref_var = torch.var(inp, dim=1, correction=0)
    ref_rstd = torch.rsqrt(ref_var + eps)
    allclose_with_dtype(ref_mean, res_mean, dtype)
    allclose_with_dtype(ref_rstd, res_rstd, dtype)
    closer_to_golden(golden_out, ref_out, res_out)

    out_grad = torch.randn_like(inp)

    (golden_in_grad, golden_weight_grad, _) = torch.autograd.grad(
        golden_out, (golden_inp, golden_weight, golden_bias), out_grad.to(torch.float64)
    )
    (ref_in_grad, ref_weight_grad, ref_bias_grad) = torch.autograd.grad(
        ref_out, (inp, weight, bias), out_grad
    )
    (res_in_grad, res_weight_grad, res_bias_grad) = torch.autograd.grad(
        res_out, (inp, weight, bias), out_grad
    )
    closer_to_golden(golden_in_grad, ref_in_grad, res_in_grad, tolerance=2)
    closer_to_golden(golden_weight_grad, ref_weight_grad, res_weight_grad, tolerance=2)
    allclose_with_dtype(ref_bias_grad, res_bias_grad, dtype)


@pytest.mark.parametrize(
    "shape",
    [(4096, i * 32) for i in range(1, 20)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_mean(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.mean(inp)
    with flag_gems.use_gems():
        res_out = torch.mean(inp)

    allclose_with_dtype(ref_out, res_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [
        (256, 256, 256),
        (1024, 1024, 1024),
        (1024, 128, 2048),
        (1024, 64, 1280),
        (640, 256, 512),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_mm(shape, dtype):
    M, N, K = shape
    tensor_a = torch.randn((M, K), dtype=dtype, device="cuda")
    tensor_b = torch.randn((K, N), dtype=dtype, device="cuda")

    golden_out = torch.mm(tensor_a.to(torch.float64), tensor_b.to(torch.float64))
    ref_out = torch.mm(tensor_a, tensor_b)
    with flag_gems.use_gems():
        res_out = torch.mm(tensor_a, tensor_b)

    closer_to_golden(golden_out, ref_out, res_out)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_mul(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.mul(inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.mul(inp1, inp2)

    allclose_with_dtype(ref_out, res_out, dtype)


@pytest.mark.parametrize(
    "shape_a",
    [(16, 1024, 256)],
)
@pytest.mark.parametrize(
    "shape_b",
    [(1, 256), (1, 1, 256), (16, 1, 256), (1, 1024, 256), (1024, 256)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_mul_broadcast(shape_a, shape_b, dtype):
    inp1 = torch.randn(shape_a, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape_b, dtype=dtype, device="cuda")

    ref_out = torch.mul(inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.mul(inp1, inp2)

    allclose_with_dtype(ref_out, res_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize(
    "scalar",
    [111.111, -999.999],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_mul_tensor_scalar(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = scalar

    ref_out = torch.mul(inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.mul(inp1, inp2)

    allclose_with_dtype(ref_out, res_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize(
    "scalar",
    [111.111, -999.999],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_mul_scalar_tensor(shape, scalar, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.mul(inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.mul(inp1, inp2)

    allclose_with_dtype(ref_out, res_out, dtype)


@pytest.mark.parametrize(
    "inp",
    [0.9, 1.0, 100.9, -111.9],
)
@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_accuracy_pow_scalar_tensor(inp, shape, dtype):
    exponent = torch.randint(-5, 5, shape, dtype=dtype, device="cuda")
    ref_out = torch.pow(inp, exponent)
    with flag_gems.use_gems():
        res_out = torch.pow(inp, exponent)

    allclose_with_dtype(ref_out, res_out, dtype, equal_nan=True)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize(
    "exponent",
    [0.5, 1.5, 5.0, -1.0],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_pow_tensor_scalar(shape, exponent, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.pow(inp, exponent)
    with flag_gems.use_gems():
        res_out = torch.pow(inp, exponent)

    allclose_with_dtype(ref_out, res_out, dtype, equal_nan=True)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_pow_tensor_tensor(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    exponent = torch.randint(-10, 10, shape, dtype=dtype, device="cuda")

    ref_out = torch.pow(inp, exponent)
    with flag_gems.use_gems():
        res_out = torch.pow(inp, exponent)

    allclose_with_dtype(ref_out, res_out, dtype, equal_nan=True)


@pytest.mark.parametrize(
    "shape_a",
    [(16, 1024, 256)],
)
@pytest.mark.parametrize(
    "shape_b",
    [(1, 256), (1, 1, 256), (16, 1, 256), (1, 1024, 256), (1024, 256)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_pow_tensor_tensor_broadcast(shape_a, shape_b, dtype):
    inp = torch.randn(shape_a, dtype=dtype, device="cuda")
    exponent = torch.randint(-10, 10, shape_b, dtype=dtype, device="cuda")

    ref_out = torch.pow(inp, exponent)
    with flag_gems.use_gems():
        res_out = torch.pow(inp, exponent)

    allclose_with_dtype(ref_out, res_out, dtype, equal_nan=True)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_reciprocal(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.reciprocal(inp)
    with flag_gems.use_gems():
        res_out = torch.reciprocal(inp)

    allclose_with_dtype(ref_out, res_out, dtype, equal_nan=True)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_relu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)

    ref_out = torch.nn.functional.relu(inp)
    with flag_gems.use_gems():
        res_out = torch.relu(inp)

    allclose_with_dtype(ref_out, res_out, dtype)

    out_grad = torch.randn_like(inp)
    (ref_in_grad,) = torch.autograd.grad(ref_out, inp, out_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    allclose_with_dtype(ref_in_grad, res_in_grad, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_rsqrt(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.rsqrt(inp)
    with flag_gems.use_gems():
        res_out = torch.rsqrt(inp)

    allclose_with_dtype(ref_out, res_out, dtype, equal_nan=True)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("alpha", [0, 1, 4, -9])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_rsub(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.rsub(inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.rsub(inp1, inp2, alpha=alpha)

    allclose_with_dtype(ref_out, res_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_silu(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)

    ref_out = torch.nn.functional.silu(inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.silu(inp)

    allclose_with_dtype(ref_out, res_out, dtype)

    out_grad = torch.randn_like(inp)
    (ref_in_grad,) = torch.autograd.grad(ref_out, inp, out_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    allclose_with_dtype(ref_in_grad, res_in_grad, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("alpha", [0, 1, 4, -9])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_sub(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.sub(inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.sub(inp1, inp2, alpha=alpha)

    allclose_with_dtype(ref_out, res_out, dtype)


@pytest.mark.parametrize(
    "shape_a",
    [(16, 1024, 256)],
)
@pytest.mark.parametrize(
    "shape_b",
    [(1, 256), (1, 1, 256), (16, 1, 256), (1, 1024, 256), (1024, 256)],
)
@pytest.mark.parametrize("alpha", [0, 1, 4, -9])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_sub_broadcast(shape_a, shape_b, alpha, dtype):
    inp1 = torch.randn(shape_a, dtype=dtype, device="cuda")
    inp2 = torch.randn(shape_b, dtype=dtype, device="cuda")

    ref_out = torch.sub(inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.sub(inp1, inp2, alpha=alpha)

    allclose_with_dtype(ref_out, res_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize(
    "scalar",
    [111.111, -999.999],
)
@pytest.mark.parametrize("alpha", [0, 1, 4, -9])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_sub_tensor_scalar(shape, scalar, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device="cuda")
    inp2 = scalar

    ref_out = torch.sub(inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.sub(inp1, inp2, alpha=alpha)

    allclose_with_dtype(ref_out, res_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize(
    "scalar",
    [111.111, -999.999],
)
@pytest.mark.parametrize("alpha", [0, 1, 4, -9])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_sub_scalar_tensor(shape, scalar, alpha, dtype):
    inp1 = scalar
    inp2 = torch.randn(shape, dtype=dtype, device="cuda")

    ref_out = torch.sub(inp1, inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.sub(inp1, inp2, alpha=alpha)

    allclose_with_dtype(ref_out, res_out, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (16, 128, 64, 64), (20, 320, 15)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_softmax(shape, dtype):
    dim = 1
    inp = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)

    ref_out = torch.nn.functional.softmax(inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.softmax(inp, dim=dim)

    allclose_with_dtype(ref_out, res_out, dtype)

    out_grad = torch.randn_like(inp)
    (ref_in_grad,) = torch.autograd.grad(ref_out, inp, out_grad)
    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    allclose_with_dtype(ref_in_grad, res_in_grad, dtype)


@pytest.mark.parametrize(
    "shape",
    [(1024, 1024), (16, 1024, 256), (32, 128, 512, 512), (20, 320, 15)],
)
@pytest.mark.parametrize("diagonal", [-3, -1, 0, 1, 3])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_accuracy_triu(shape, diagonal, dtype):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    ref_out = torch.triu(inp, diagonal)
    with flag_gems.use_gems():
        res_out = torch.triu(inp, diagonal)

    allclose_with_dtype(ref_out, res_out, dtype)

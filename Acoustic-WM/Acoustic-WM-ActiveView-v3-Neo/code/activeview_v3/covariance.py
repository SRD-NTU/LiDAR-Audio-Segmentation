"""Lossless 16-real-coordinate representation of a 4x4 Hermitian matrix."""
import torch

PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]

def pack(c):
    return torch.cat([c.diagonal(dim1=-2, dim2=-1).real,
                      torch.stack([c[..., i, j].real for i, j in PAIRS], -1),
                      torch.stack([c[..., i, j].imag for i, j in PAIRS], -1)], -1)

def unpack(x):
    x = x.float()
    result = torch.diag_embed(x[..., :4]).to(torch.complex64)
    for n, (i, j) in enumerate(PAIRS):
        z = torch.complex(x[..., 4 + n], x[..., 10 + n])
        result[..., i, j] = z
        result[..., j, i] = z.conj()
    return result

def square_error(x, y):
    d = (x.float() - y.float()).square()
    return d[..., :4].sum(-1) + 2 * d[..., 4:].sum(-1)

def validate(c, tolerance=2e-3):
    assert torch.isfinite(c).all()
    assert (c - c.mH).abs().max() < tolerance
    assert (c.diagonal(dim1=-2, dim2=-1).real.sum(-1) - 1).abs().max() < tolerance
    assert torch.linalg.eigvalsh(c).min() > -tolerance

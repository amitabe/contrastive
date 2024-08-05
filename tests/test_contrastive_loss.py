import pytest

import torch
from torch.nn import functional as F
import contrastive  # Import the custom module


class SNNLoss(torch.nn.Module):
    def __init__(self, temperature=1, use_cosine=False, reduction="mean") -> None:
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.use_cosine = use_cosine
        self.get_distances = self.compute_cosine_distances if use_cosine else self.compute_l2_distances
        self.reduce = self.mean_reduce if reduction == "mean" else self.none_reduce

    @staticmethod
    def compute_l2_distances(z):
        return -(z.unsqueeze(1) - z.unsqueeze(0)).square().sum(dim=-1).sqrt()

    @staticmethod
    def compute_cosine_distances(z):
        normalized_z = F.normalize(z, dim=-1)
        distances = normalized_z @ normalized_z.T

        return distances

    @staticmethod
    def remove_diagnonal(x):
        # We remove the first element after flattening (on diagonal)
        # Since N^2 - 1 = (N - 1) * (N + 1) we reshape `x` that way
        # This reshape aligns the diagonal elements on the last column

        N = x.shape[0]
        new_shape = [N, N-1] + list(x.shape[2:])
        out = x.flatten(end_dim=1)[1:].unflatten(0, (N-1, N+1))[:, :-1].reshape(new_shape)
        return out

    # Defined at __init__ time
    def get_distances(self, x, y):
        pass

    # Defined at __init__ time
    def reduce(self, x):
        pass

    def mean_reduce(self, x):
        return x.mean()

    def none_reduce(self, x):
        return x

    def forward(self, x, y):
        z = torch.cat((x, y), dim=0)
        labels = torch.cat([torch.arange(x.shape[0], device=z.device)] * 2)

        return self.forward_w_labels(z, labels)

    def forward_w_labels(self, z, labels):
        sim = self.get_distances(z) / self.temperature

        # Masks
        positive_mask = labels.unsqueeze(1) == labels.unsqueeze(0)

        # Remove diagonal - we know it's always 0 (1 for cosine) and it makes it difficult to compute the max
        sim = self.remove_diagnonal(sim)
        positive_mask = self.remove_diagnonal(positive_mask)

        negative_mask = ~positive_mask

        # Numerical stability
        max_val = sim.max(dim=1, keepdim=True)[0].detach()
        sim = sim - max_val

        # Sum up the positive and negative samples using the masks
        sim_exp = sim.exp()

        numerator = (sim_exp * positive_mask).sum(dim=-1)
        denominator = (sim_exp * negative_mask).sum(dim=-1)

        log_arg = numerator / (denominator + numerator)

        # Numerical stability
        eps = torch.finfo(log_arg.dtype).tiny
        log_arg = log_arg + eps

        log_res = -log_arg.log()

        return self.reduce(log_res)


def python_snnl(tensor, labels, temperature, reduce="mean", use_cosine=False):
    snn = SNNLoss(temperature=temperature, use_cosine=use_cosine, reduction=reduce)

    return snn.forward_w_labels(z=tensor, labels=labels)


def test_snnl_basic():
    z = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    labels = torch.tensor([0, 1, 0], dtype=torch.int64)
    temperature = 0.5

    kwargs = {
        "tensor": z,
        "labels": labels,
        "temperature": temperature,
    }

    expected_loss = python_snnl(**kwargs)
    result = contrastive.snnl(**kwargs)

    assert torch.allclose(result, expected_loss, atol=1e-12), f"Expected {expected_loss}, but got {result}"


@pytest.mark.parametrize('execution_number', range(100))
def test_snnl_all_random(execution_number):
    import random

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for _ in range(execution_number):
        rand_type = random.choice([torch.float64, torch.float32, torch.float16])
        rand_int_type = random.choice([torch.int8, torch.int16, torch.int32, torch.int64])
        # reduce = random.choice(["mean", "none"])
        reduce = random.choice(["none",])
        N = random.randint(2, 100)


        z = torch.rand((N, N), dtype=rand_type).to(device)
        labels = torch.randint(N, (N,), dtype=rand_int_type).to(device)
        temperature = random.uniform(1e-20, 2)
        use_cosine = random.choice([True, False])

        kwargs = {
            "tensor": z,
            "labels": labels,
            "temperature": temperature,
            "reduce": reduce,
            "use_cosine": use_cosine,
        }

        expected_loss = python_snnl(**kwargs)
        result = contrastive.snnl(**kwargs)

        diff = (result-expected_loss).abs().max()
        median = (result-expected_loss).abs().median()
        avg = (result-expected_loss).abs().mean()
        assert torch.allclose(result, expected_loss, atol=1e-3), f"diff={diff}; avg={avg}; median={median}\n{kwargs}"
        assert result.isnan().sum() == 0, f"Nans were found: {result}"

if __name__ == "__main__":
    import pytest
    pytest.main()

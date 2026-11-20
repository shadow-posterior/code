"""Pack detached scalar reporting and validation into one host transfer."""
import torch


def scalar_metrics(metrics):
    if not metrics:
        return {}
    device = next((v.device for v in metrics.values() if isinstance(v, torch.Tensor)), torch.device('cpu'))
    values = torch.stack([torch.as_tensor(v, device=device, dtype=torch.float64).detach()
                          for v in metrics.values()])
    return dict(zip(metrics, values.cpu().tolist()))


class TensorChecks:
    """Keep synchronous, ordered exceptions without synchronizing per condition."""
    def __init__(self):
        self.conditions = []

    def require(self, condition, message, error=ValueError):
        self.conditions.append((condition.detach(), message, error))

    def check(self):
        if not self.conditions:
            return
        valid = torch.stack([c for c, _, _ in self.conditions]).cpu().tolist()
        for passed, (_, message, error) in zip(valid, self.conditions):
            if not passed:
                raise error(message)

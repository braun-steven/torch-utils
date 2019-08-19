# Torch Utilities

This repo contains some utility functions that I constantly reuse when writing experiment code using PyTorch. 

Feel free to contribute your helper functions which you could not live without!

## Utility Functions

```python
def time_delta_now(t_start: float) -> str: ...
def ensure_dir(path: str): ...
def count_params(model: torch.nn.Module) -> int: ...
def generate_run_base_dir(...): ...
def setup_logging(filename: str = "log.txt", level: str = "INFO"): ...
def set_seed(seed: int): ...
def set_cuda_device(cuda_device_id): ...
def make_multi_gpu(model: torch.nn.Module, cuda_device_id: List): ...
def load_args(base_dir: str, filename="args.txt") -> argparse.Namespace: ...
def save_args(args: argparse.Namespace, base_dir: str): ...
def clone_args(args: argparse.Namespace) -> argparse.Namespace: ...
def plot_samples(x: torch.Tensor, y: torch.Tensor): ...
```

## qdaq: Cuda Experiment Queue

`qdaq` makes running multiple PyTorch or Tensorflow experiments on more than one GPU easy.
The typical use case is e.g. a gridsearch over hyperparameters for the same experiment with a bunch of GPUs available. Usually it would be necessary to manually split the hyperparameter space and start the experiments on different GPUs. This has multiple drawbacks:

- It's tedious to split experiment setups and start them on the appropriate GPU device by hand.
- If 10 experiments are sequentially started on GPU0 and 10 on GPU1 it might happen that GPU0 finishes way earlier while GPU1 is still running with a queue of experiments. This results in unused GPU time on GPU0 (bad load balancing).

The main function is to provide a multiprocessing queue with available cuda devices. The idea is to define a `Job`, e.g. your experiment, create a bunch of jobs with different settings, specify a list of available cuda devices and let the internals handle the rest. In the background each job is put into a queue and grabs a cuda device as soon as it is available.

### Quick Start

The following is a quick example on how to use qdaq:

```python
import torch
from utils.qdaq import Job, start


# Create a job class that implements `run`
class Foo(Job):
    def __init__(self, q):
        # Save some job parameters
        self.q = q

    def run(self, cuda_device_id):
        # Get cuda device
        device = torch.device(f"cuda:{cuda_device_id}")

        # Send data to device
        x = torch.arange(1, 3).to(device)

        # Compute stuff
        y = x.pow(self.q)
        print(f"Cuda device {cuda_device_id}, Exponent {self.q}, Result {y}")


if __name__ == "__main__":
    exps = [Foo(i) for i in range(9)]
    start(exps, [1, 3])

# Output:
# Cuda device 3, Exponent 1, Result tensor([1, 2], device='cuda:3')
# Cuda device 1, Exponent 0, Result tensor([1, 1], device='cuda:1')
# Cuda device 3, Exponent 2, Result tensor([1, 4], device='cuda:3')
# Cuda device 1, Exponent 3, Result tensor([1, 8], device='cuda:1')
# Cuda device 1, Exponent 5, Result tensor([ 1, 32], device='cuda:1')
# Cuda device 3, Exponent 4, Result tensor([ 1, 16], device='cuda:3')
# Cuda device 1, Exponent 6, Result tensor([ 1, 64], device='cuda:1')
# Cuda device 3, Exponent 8, Result tensor([  1, 256], device='cuda:3')
# Cuda device 1, Exponent 7, Result tensor([  1, 128], device='cuda:1')
```

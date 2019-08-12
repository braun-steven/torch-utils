import multiprocessing as mp
import logging
from abc import ABC, abstractmethod
from typing import List, Union, Callable

# Get logger
logger = logging.getLogger(__name__)


class Job(ABC):
    """
    Abstract Job that describes what is to be done.
    The memberfunction `run` has to be implemented.
    """

    @abstractmethod
    def run(self, cuda_device_id):
        """
        Run the job on the given cuda device.

        Args:
            cuda_device_id (Union[int, List[int]]): Can be either a cuda device id or a list of 
                                                    cuda device ids.
        """
        pass


def _make_target(job) -> Callable:
    """
    Create target function that pulls an element from the queue and runs the
    job on that element.

    Args:
        job (Job): Job to run.

    Returns:
        Callable: Target to use for the multiprocessing API.
    """

    def _target(queue):
        # Lock cuda device
        cuda_device_id = queue.get()

        # Run job on this device
        job.run(cuda_device_id)

        # Free cuda device
        queue.put(cuda_device_id)

    return _target


def start(jobs: List[Job], cuda_device_ids: Union[List[int], List[List[int]]]):
    """
    Start the given jobs on the list of cuda devices.

    Args:
        jobs (List[Job]): [TODO:description]
        cuda_device_ids (Union[List[int], List[List[int]]]): [TODO:description]
    """
    assert len(jobs) > 0, "No jobs specified."
    assert len(cuda_device_ids) > 0, "No cuda devices specified."

    # Create queue
    queue = mp.Queue(maxsize=len(cuda_device_ids))

    # Put all devices into the queue
    for id in cuda_device_ids:
        queue.put(id)

    # Create processes
    processes = []
    for job in jobs:
        f = _make_target(job)
        p = mp.Process(target=f, args=(queue,))
        processes.append(p)

    # Start processes
    for p in processes:
        p.start()

    # Join processes
    for p in processes:
        p.join()


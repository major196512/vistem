import torch
import torch.multiprocessing as mp

from vistem.utils import setup_logger, find_caller
from vistem import dist

__all__ = ["launch"]


def launch(main_func, num_gpus_per_machine, num_machines=1, machine_rank=0, dist_ip=None, dist_port=None, args=()):
    world_size = num_machines * num_gpus_per_machine
    if world_size == 1:
        main_func(*args)

    else:
        if machine_rank == 0:
            dist_ip = "127.0.0.1"
            if dist_port is None : dist_port = _find_free_port()

        else:
            assert num_machines > 1, "At least 2 machines is needed"
            assert dist_ip is not None, "Set main machine ip address"
            assert dist_port is not None, "Set dist port number which is same with main machine port"

        dist_url = f"tcp://{dist_ip}:{dist_port}"
        logger = setup_logger()
        logger.info(f"pytorch distribute url : {dist_url}")

        mp.spawn(
            _distributed_worker,
            nprocs=num_gpus_per_machine,
            args=(main_func, world_size, num_gpus_per_machine, machine_rank, dist_url, args),
            daemon=False,
        )

def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()

    return port

def _distributed_worker(
    local_rank, main_func, world_size, num_gpus_per_machine, machine_rank, dist_url, args
):
    assert torch.cuda.is_available(), "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    try:
        dist.init_process_group(
            backend="NCCL", init_method=dist_url, world_size=world_size, rank=global_rank
        )
    except Exception as e:
        logger = setup_logger()
        logger.error("Process group URL: {}".format(dist_url))
        raise e

    dist.synchronize()

    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    # Setup the local process group (which contains ranks within the same machine)
    assert dist._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            dist._LOCAL_PROCESS_GROUP = pg

    main_func(*args)

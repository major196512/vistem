import torch
import datetime
from contextlib import contextmanager

from .default import Evaluators
from vistem.utils import setup_logger, Timer
from vistem import dist

def evaluator(model, data_loader, evaluators):
    num_devices = dist.get_world_size()
    _logger = setup_logger(__name__, all_rank=True)

    total = len(data_loader)  # inference data loader must have a fixed length
    _logger.info(f"Start inference on {total} images")

    if evaluators is None : evaluators = Evaluators([])
    evaluators.reset()

    timer = Timer(warmup = 5, pause=True)
    total_compute_time = 0
    total_time = 0
    
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            timer.resume()
            outputs = model(inputs)
            if torch.cuda.is_available() : torch.cuda.synchronize()
            timer.pause()
            evaluators.process(inputs, outputs)

            if timer.total_seconds() > 10:
                total_compute_time += timer.seconds()
                total_time += timer.total_seconds()
                timer.reset(pause=True)

                total_seconds_per_img = total_time / (idx + 1)
                seconds_per_img = total_compute_time / (idx + 1)
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                _logger.info(f"Inference done {idx + 1}/{total}. {seconds_per_img:.4f} s / img. ETA={eta}")

    total_compute_time += timer.seconds()
    total_time += timer.total_seconds()

    total_time_str = str(datetime.timedelta(seconds=total_time))
    _logger.info(
        f"Total inference time: {total_time_str} ({total_time / total:.6f} s / img per device, on {num_devices} devices)"
    )

    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    _logger.info(
        f"Total inference pure compute time: {total_compute_time_str} ({total_compute_time / total:.6f} s / img per device, on {num_devices} devices)"
    )

    results = evaluators.evaluate()
    if results is None : results = {}
    return results


@contextmanager
def inference_context(model):
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)

import torch
from torch.nn.parallel import DistributedDataParallel

import os
import datetime
from contextlib import contextmanager
from PIL import Image, ImageDraw

from vistem.loader import MetadataCatalog
from vistem.utils import setup_logger, Timer
from vistem import dist

from vistem.loader import build_test_loader
from vistem.modeling import build_model
from vistem.checkpointer import Checkpointer

class Visualizer:
    def __init__(self, cfg):
        self._logger = setup_logger(__name__, all_rank=True)
        
        if dist.is_main_process():
            self._logger.debug(f'Config File : \n{cfg}')
            if cfg.VISUALIZE_DIR and not os.path.isdir(cfg.VISUALIZE_DIR) : os.makedirs(cfg.VISUALIZE_DIR)
            self.visualize_dir = cfg.VISUALIZE_DIR
        dist.synchronize()
        
        self.test_loader = build_test_loader(cfg)

        self.model = build_model(cfg)
        self.model.eval()
        if dist.is_main_process():
            self._logger.debug(f"Model Structure\n{self.model}")
                
        if dist.get_world_size() > 1:
            self.model = DistributedDataParallel(self.model, device_ids=[dist.get_local_rank()], broadcast_buffers=False)

        self.checkpointer = Checkpointer(
            self.model,
            cfg.OUTPUT_DIR,
        )
        self.checkpointer.load(cfg.WEIGHTS)

        self.meta_data = MetadataCatalog.get(cfg.LOADER.TEST_DATASET)
        self.class_color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    def __call__(self):
        num_devices = dist.get_world_size()

        total = len(self.test_loader)  # inference data loader must have a fixed length
        self._logger.info(f"Start visualize on {total} images")

        timer = Timer(warmup = 5, pause=True)
        total_compute_time = 0
        total_time = 0

        with inference_context(self.model), torch.no_grad():
            for idx, inputs in enumerate(self.test_loader):
                timer.resume()
                outputs = self.model(inputs)
                if torch.cuda.is_available() : torch.cuda.synchronize()
                timer.pause()

                self.save_visualize(inputs, outputs)

                if timer.total_seconds() > 10:
                    total_compute_time += timer.seconds()
                    total_time += timer.total_seconds()
                    timer.reset(pause=True)

                    total_seconds_per_img = total_time / (idx + 1)
                    seconds_per_img = total_compute_time / (idx + 1)
                    eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                    self._logger.info(f"Visualize done {idx + 1}/{total}. {seconds_per_img:.4f} s / img. ETA={eta}")

        total_compute_time += timer.seconds()
        total_time += timer.total_seconds()

        total_time_str = str(datetime.timedelta(seconds=total_time))
        self._logger.info(
            f"Total Visualize time: {total_time_str} ({total_time / total:.6f} s / img per device, on {num_devices} devices)"
        )

        total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
        self._logger.info(
            f"Total visualize pure compute time: {total_compute_time_str} ({total_compute_time / total:.6f} s / img per device, on {num_devices} devices)"
        )

    def save_visualize(self, inputs, outputs):
        inputs = inputs[0]
        outputs = outputs[0]

        file_name = inputs['file_name']
        instances = outputs['instances']
        base_name = os.path.basename(file_name)
        split_name = base_name.split('.')[0]

        pred_boxes = instances.pred_boxes.tensor
        pred_cls = instances.pred_classes
        pred_scores = instances.pred_scores

        im = Image.open(file_name)
        draw = ImageDraw.Draw(im)

        class_color = list()
        save_info = ''
        for idx in range(len(instances)):
            boxes = pred_boxes[idx].cpu().numpy()
            classes = pred_cls[idx].cpu().numpy()
            scores = pred_scores[idx].cpu().numpy()

            if classes not in class_color : class_color.append(classes)
            color = self.class_color[class_color.index(classes) % 5]

            draw.rectangle(boxes, outline=tuple([int(c * scores) for c in color]), width=int(4 * scores)+1)
            save_info += f'{self.meta_data.category_names[classes]}({boxes[0]:.2f}, {boxes[1]:.2f}, {boxes[2]:.2f}, {boxes[3]:.2f}) : {scores}\n'

        im.save(os.path.join(self.visualize_dir, base_name))
        with open(os.path.join(self.visualize_dir, f'{split_name}.txt'), 'w') as f:
            for idx, color in enumerate(class_color):
                f.write(f'{self.meta_data.category_names[color]} : {self.class_color[idx % 5]}\n')
            f.write(save_info)

@contextmanager
def inference_context(model):
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
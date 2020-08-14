import torch
import torch.nn as nn

__all__ = ['BufferList', 'create_grid_offsets']

class BufferList(nn.Module):
    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


def create_grid_offsets(size, stride, device):
    grid_height, grid_width = size
    shifts_x = torch.arange(0, grid_width * stride, step=stride, dtype=torch.float32, device=device)
    shifts_y = torch.arange(
        0, grid_height * stride, step=stride, dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return shift_x, shift_y

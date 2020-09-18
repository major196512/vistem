import torch

__all__ = ['instance_mapper']

def annotation_mapper(dataset_dict : dict):
    classes = [dataset_dict.pop('annotations')]
    classes = torch.tensor(classes, dtype=torch.int64)

    dataset_dict["annotations"] = classes

    return dataset_dict
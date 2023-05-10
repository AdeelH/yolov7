# Adapted from:
# https://github.com/openvinotoolkit/openvino_notebooks/blob/develop/notebooks/226-yolov7-optimization/226-yolov7-optimization.ipynb

from os.path import join
import argparse
import yaml
import logging

import numpy as np
from openvino.runtime import Core, serialize
import nncf

from collections import namedtuple
from utils.datasets import create_dataloader

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def get_dataloader(data_root: str, img_size: int):
    # read dataset config
    data_config = join(data_root, 'coco.yaml')
    with open(data_config) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    # Dataloader
    data['val'] = join(data_root, data['val'])
    # imitation of commandline provided options for single class evaluation
    Option = namedtuple('Options', ['single_cls'])
    opt = Option(False)
    dataloader, _ = create_dataloader(
        data['val'], imgsz=img_size, batch_size=1, stride=32, opt=opt, pad=0.5)
    return dataloader


def quantize(model, dataloader):
    def transform_fn(data_item):
        img = data_item[0].numpy()
        x = img.astype(np.float32)  # uint8 to fp16/32
        x /= 255.0  # 0 - 255 to 0.0 - 1.0

        if x.ndim == 3:
            x = np.expand_dims(x, 0)

        return x

    quantization_dataset = nncf.Dataset(dataloader, transform_fn)
    quantized_model = nncf.quantize(
        model, quantization_dataset, preset=nncf.QuantizationPreset.MIXED)
    return quantized_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, help='Data root dir')
    parser.add_argument('--model-in', type=str, help='Input model path')
    parser.add_argument('--model-out', type=str, help='Output model path')
    parser.add_argument(
        '--img-size', type=int, default=640, help='Inference size (pixels)')
    opt = parser.parse_args()

    log.info('Reading model')
    core = Core()
    model = core.read_model(opt.model_in)

    log.info('Creating dataloader')
    dataloader = get_dataloader(opt.data_root, opt.img_size)

    log.info('Quantizing')
    quantized_model = quantize(model, dataloader)

    log.info('Saving quantized model')
    serialize(quantized_model, opt.model_out)

# Adapted from:
# https://github.com/openvinotoolkit/openvino_notebooks/blob/develop/notebooks/226-yolov7-optimization/226-yolov7-optimization.ipynb

from os.path import splitext
import argparse

from openvino.tools import mo
from openvino.runtime import serialize

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='Input ONNX model path')
    opt = parser.parse_args()

    in_path = opt.model_path
    name, _ = splitext(in_path)
    out_path = f'{name}.xml'

    model = mo.convert_model(in_path)
    serialize(model, out_path)

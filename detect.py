from typing import List, Optional, Tuple
from os import makedirs
from os.path import dirname
from jsonargparse import ArgumentParser, ActionConfigFile
from pathlib import Path
import json

import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader

from models.experimental import attempt_load
from utils.datasets import VideoDataset, VideoDatasetTorchVision
from utils.general import (check_img_size, non_max_suppression, scale_coords,
                           set_logging)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

CLASS_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]


def detect(video: str,
           model_path: str,
           out_json_path: bool,
           img_size: int,
           conf_thresh: float,
           iou_thresh: float,
           stride: int,
           device: str,
           agnostic_nms: bool = False,
           classes: Optional[List[int]] = None,
           out_video_path: Optional[str] = None,
           frame_predict_interval: Optional[int] = None,
           crop: Optional[Tuple[int, int, int, int]] = None,
           torchvision_video_reader: bool = False,
           batch_size: Optional[int] = None,
           num_workers: int = 0,
           debug: bool = False):
    # Initialize
    set_logging()
    device = select_device(device)

    # Load model
    img_size = check_img_size(img_size, s=stride)

    # colors for visualization
    colors = (
        [[0, 0, 255]] +
        [np.random.randint(0, 255, size=3).tolist() for _ in CLASS_NAMES[1:]])

    is_openvino = Path(model_path).suffix == '.xml'
    if is_openvino:
        from openvino.runtime import Core
        core = Core()
        # read converted model
        model = core.read_model(model_path)
        model = core.compile_model(model, 'CPU')
        output_blob = model.output(0)
    else:
        model = attempt_load(model_path, map_location=device)

    # Set Dataloader
    if torchvision_video_reader:
        ds_cls = VideoDatasetTorchVision
    else:
        ds_cls = VideoDataset
    dataset = ds_cls(
        video,
        img_size=img_size,
        stride=stride,
        auto=False,
        read_interval=frame_predict_interval,
        crop_xyxy=crop,
    )
    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers)

    # init vid_writer
    save_vid = out_video_path is not None
    if save_vid:
        fps = dataset.fps
        w = dataset.frame_w
        h = dataset.frame_h
        if crop is not None:
            x0, y0, x1, y1 = crop
            if x0 < 0:
                x0 += w
            if y0 < 0:
                y0 += h
            if x1 < 0:
                x1 += w
            if y1 < 0:
                y1 += h
            h, w = y1 - y0, x1 - x0
        vid_writer = cv2.VideoWriter(out_video_path,
                                     cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                     (w, h))

    nframes = dataset.nframes
    duration = dataset.duration
    fps = dataset.fps
    all_dets = {}
    frames_processed = 0
    for batch in dataloader:
        if debug and frames_processed >= 50:
            break
        img, raw_frames = batch
        N = len(img)
        raw_frames = raw_frames.numpy()
        batch_frame_inds = frames_processed + np.arange(N, dtype=int)
        frames_processed += N
        batch_frame_timestamps = duration * (batch_frame_inds / nframes)
        img = img.to(device=device, dtype=torch.float32)
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        if is_openvino:
            t1 = time_synchronized()
            with torch.inference_mode():
                pred = model(img)[output_blob]
                pred = torch.from_numpy(pred)
            t2 = time_synchronized()
        else:
            t1 = time_synchronized()
            with torch.inference_mode():
                pred = model(img)[0]
            t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(
            pred,
            conf_thresh,
            iou_thresh,
            classes=classes,
            agnostic=agnostic_nms)
        t3 = time_synchronized()

        # Process detections
        for i, (det, raw_frame, frame_idx, frame_timestamp) in enumerate(
                zip(pred, raw_frames, batch_frame_inds,
                    batch_frame_timestamps)):
            print(f'({frame_idx}/{nframes}): ', end='')
            if len(det):
                # Rescale boxes from img_size to raw_frame size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4],
                                          raw_frame.shape).round()

                all_dets[frame_idx] = dict(
                    frame_idx=int(frame_idx),
                    timestamp=float(frame_timestamp),
                    det=det)

                # Print results
                s = ''
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {CLASS_NAMES[int(c)]}{'s' * (n > 1)}, "
                print(s)

                # Write results
                if save_vid:
                    for *xyxy, conf, cls in reversed(det):
                        cls = int(cls)
                        label = f'{CLASS_NAMES[cls]} {conf:.2f}'
                        plot_one_box(
                            xyxy,
                            raw_frame,
                            label=label,
                            color=colors[int(cls)],
                            line_thickness=4)

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=4)

            # Print time (inference + NMS)
            print(f'Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, '
                  f'({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Save results
            if save_vid:
                vid_writer.write(raw_frame)

    frame_dets_to_json(dataset, all_dets, out_json_path)

    print('Done.')


def det_to_dict(det: list) -> dict:
    *xyxy, conf, class_id = det
    class_id = int(class_id)
    out = dict(
        xyxy=xyxy,
        conf=conf,
        class_id=class_id,
        class_name=CLASS_NAMES[class_id])
    return out


def frame_dets_to_list(frame_dets: torch.Tensor) -> list:
    frame_dets = frame_dets.tolist()
    out = [det_to_dict(det) for det in reversed(frame_dets)]
    return out


def frame_dets_to_json(dataset: VideoDataset, frame_dets: torch.Tensor,
                       out_path: str) -> None:
    video_metadata = dict(
        filename=Path(dataset.path).name,
        path=str(dataset.path),
        fps=dataset.fps,
        total_frames=dataset.nframes,
        duration=dataset.duration)
    for v in frame_dets.values():
        v['predictions'] = frame_dets_to_list(v['det'])
        del v['det']
    out = dict(
        video_metadata=video_metadata,
        frame_predictions=list(frame_dets.values()))
    makedirs(dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('video', type=str, help='video')
    parser.add_argument(
        'out_json_path',
        type=str,
        help='Path to JSON file containing predictions.')
    parser.add_argument(
        'model_path', type=str, help='Model path. Can be an OpenVINO XML.')
    parser.add_argument(
        '--conf-thresh',
        type=float,
        default=0.25,
        help='object confidence threshold')
    parser.add_argument(
        '--iou-thresh', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument(
        '--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument(
        '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--stride', type=int, default=32, help='model stride')
    parser.add_argument(
        '--out-video-path',
        type=str,
        help='Save video with detections drawn here.')
    parser.add_argument(
        '--classes',
        nargs='+',
        type=int,
        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument(
        '--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument(
        '--frame-predict-interval',
        type=int,
        default=1,
        help='Predict only on frames-indices that are multiples of this.')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='number of DataLoader workers')
    parser.add_argument(
        '--crop',
        type=int,
        nargs=4,
        help='Window to crop frames to in xyxy format.')
    parser.add_argument(
        '--torchvision-video-reader',
        action='store_true',
        help='Use the TorchVision video reader instead of OpenCV.')
    parser.add_argument('--debug', action='store_true', help='Debug mode.')
    parser.add_argument("--config", action=ActionConfigFile)
    opt = parser.parse_args()
    print(opt)

    args = opt.as_dict()
    del args['config']
    detect(**args)

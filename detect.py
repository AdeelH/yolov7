from os.path import join
import argparse
from pathlib import Path
import json

import cv2
import torch
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import (check_img_size, non_max_suppression, scale_coords,
                           xyxy2xywh, set_logging, increment_path)
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


def detect():
    source, weights, view_img, save_txt, imgsz = (opt.source, opt.weights,
                                                  opt.view_img, opt.save_txt,
                                                  opt.img_size)
    save_img = not opt.nosave

    # Directories
    save_dir = Path(
        increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(
        parents=True, exist_ok=True)  # make dir
    out_json_path = join(save_dir, 'predictions.json')

    # Initialize
    set_logging()
    device = select_device(opt.device)

    # Load model
    model = attempt_load(weights[:1], map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # colors for visualization
    colors = [np.random.randint(0, 255, size=3).tolist() for _ in CLASS_NAMES]

    if opt.openvino is not None:
        from openvino.runtime import Core
        core = Core()
        # read converted model
        model = core.read_model(opt.openvino)
        model = core.compile_model(model, 'CPU')
        output_blob = model.output(0)

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=False)

    pred_interval = opt.frame_predict_interval
    nframes = dataset.nframes
    duration = dataset.duration
    fps = dataset.fps
    all_dets = {}
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        if frame_idx % pred_interval != 0:
            continue
        vid_frac = frame_idx / nframes
        timestamp = duration * vid_frac
        print(f'({frame_idx}/{nframes}): ', end='')
        img = torch.from_numpy(img).to(device=device, dtype=torch.float32)
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        if opt.openvino is not None:
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
            opt.conf_thres,
            opt.iou_thres,
            classes=opt.classes,
            agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + (
                '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            # normalization gain whwh
            whwh_tensor = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4],
                                          im0.shape).round()

                all_dets[frame_idx] = dict(
                    frame_idx=frame_idx, timestamp=timestamp, det=det)

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {CLASS_NAMES[int(c)]}{'s' * (n > 1)}, "

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        # normalized xywh
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                                whwh_tensor).view(-1).tolist()
                        # label format
                        line = (cls, *xywh, conf) if opt.save_conf else (cls,
                                                                         *xywh)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{CLASS_NAMES[int(cls)]} {conf:.2f}'
                        plot_one_box(
                            xyxy,
                            im0,
                            label=label,
                            color=colors[int(cls)],
                            line_thickness=4)

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, '
                  f'({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(
                        f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release(
                            )  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                            (w, h))
                    vid_writer.write(im0)

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


def frame_dets_to_json(dataset: LoadImages, frame_dets: torch.Tensor,
                       out_path: str) -> None:
    video_metadata = dict(
        filename=Path(dataset.path).name,
        fps=dataset.fps,
        total_frames=dataset.nframes,
        duration=dataset.duration)
    for v in frame_dets.values():
        v['predictions'] = frame_dets_to_list(v['det'])
        del v['det']
    out = dict(
        video_metadata=video_metadata,
        frame_predictions=list(frame_dets.values()))
    with open(out_path, 'w') as f:
        json.dump(out, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--weights',
        nargs='+',
        type=str,
        default='yolov7.pt',
        help='model.pt path(s)')
    parser.add_argument(
        '--source', type=str, default='inference/images',
        help='source')  # file/folder, 0 for webcam
    parser.add_argument(
        '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument(
        '--conf-thres',
        type=float,
        default=0.25,
        help='object confidence threshold')
    parser.add_argument(
        '--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument(
        '--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument(
        '--view-img', action='store_true', help='display results')
    parser.add_argument(
        '--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument(
        '--save-conf',
        action='store_true',
        help='save confidences in --save-txt labels')
    parser.add_argument(
        '--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument(
        '--classes',
        nargs='+',
        type=int,
        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument(
        '--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument(
        '--augment', action='store_true', help='augmented inference')
    parser.add_argument(
        '--update', action='store_true', help='update all models')
    parser.add_argument(
        '--project',
        default='runs/detect',
        help='save results to project/name')
    parser.add_argument(
        '--name', default='exp', help='save results to project/name')
    parser.add_argument(
        '--exist-ok',
        action='store_true',
        help='existing project/name ok, do not increment')
    parser.add_argument(
        '--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--openvino', type=str, help='openVINO model path')
    parser.add_argument(
        '--frame-predict-interval',
        type=int,
        default=1,
        help='Predict only on frames-indices that are multiples of this.')
    opt = parser.parse_args()
    print(opt)

    with torch.inference_mode():
        detect()

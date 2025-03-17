import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np
from utils.utils import (
    time_synchronized, select_device, increment_path,
    scale_coords, xyxy2xywh, non_max_suppression, split_for_trace_model,
    driving_area_mask, lane_line_mask, plot_one_box, show_seg_result,
    AverageMeter, LoadImages
)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='data/weights/yolopv2.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/example.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--desired-fps', type=int, default=15, help='Desired FPS for processing video')
    return parser

def calculate_steering_angle(ll_seg_mask, frame_width, frame_height):
    # Define region of interest (ROI)
    roi_height = frame_height // 2
    roi = ll_seg_mask[roi_height:, :]

    # Find the nearest lane points
    lane_points = np.column_stack(np.where(roi > 0))

    if len(lane_points) == 0:
        return 0  # No lane detected

    # Calculate the center of the lane points
    lane_center = np.mean(lane_points[:, 1])

    # Calculate the deviation from the center of the frame
    frame_center = frame_width // 2
    deviation = lane_center - frame_center

    # Calculate the steering angle based on the deviation
    max_deviation = frame_width // 2
    steering_angle = (deviation / max_deviation) * 45  # Scale to -45 to 45 degrees

    # Apply a threshold to filter slight changes
    if abs(steering_angle) < 5:
        steering_angle = 0

    return steering_angle

def calculate_velocity(start_time, current_time, steering_angle):
    # Simulate velocity based on time and steering angle
    elapsed_time = current_time - start_time

    # Increase velocity to 20 km/h in the first 5 seconds
    if elapsed_time < 5:
        velocity = min(20, 4 * elapsed_time)  # Linear increase to 20 km/h in 5 seconds
    else:
        # Adjust velocity based on steering angle after 5 seconds
        # Reduce speed if steering angle is large (sharp turn)
        velocity = 20 - abs(steering_angle) * 0.2  # Reduce speed by 0.2 km/h per degree of steering angle
        velocity = max(10, velocity)  # Ensure minimum speed of 10 km/h

    return velocity

def detect():
    # setting and directories
    source, weights, save_txt, imgsz, desired_fps = opt.source, opt.weights, opt.save_txt, opt.img_size, opt.desired_fps
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    # Load model
    stride = 32
    model = torch.jit.load(weights, map_location='cpu')
    device = select_device('cpu')
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = model.to(device)

    if half:
        model.half()  # to FP16
    model.eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride, desired_fps=desired_fps)  # Pass desired_fps

    # Initialize frame counters
    total_frames = 0
    processed_frames = 0

    # Start time for velocity simulation
    start_time = time.time()

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    for path, img, im0s, vid_cap in dataset:
        total_frames += 1
        processed_frames += 1

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        [pred, anchor_grid], seg, ll = model(img)
        t2 = time_synchronized()

        # waste time: the incompatibility of  torch.jit.trace causes extra time consumption
        tw1 = time_synchronized()
        pred = split_for_trace_model(pred, anchor_grid)
        tw2 = time_synchronized()

        # Apply NMS
        t3 = time_synchronized()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t4 = time_synchronized()

        # change start
        da_seg_mask = driving_area_mask(seg)
        # change end

        ll_seg_mask = lane_line_mask(ll)

        # Calculate steering angle
        frame_height, frame_width = im0s.shape[:2]
        steering_angle = calculate_steering_angle(ll_seg_mask, frame_width, frame_height)

        # Calculate velocity based on time and steering angle
        current_time = time.time()
        velocity = calculate_velocity(start_time, current_time, steering_angle)

        # Process detections
        for i, det in enumerate(pred):  # detections per image

            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string


                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img :  # Add bbox to image
                        plot_one_box(xyxy, im0, line_thickness=3)

            # Print time (inference)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True)

            # Overlay steering angle and velocity on the frame
            cv2.putText(im0, f"Steering Angle: {steering_angle:.2f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(im0, f"Velocity: {velocity:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = opt.desired_fps  # Use desired_fps instead of original
                            w, h = im0.shape[1], im0.shape[0]
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

        inf_time.update(t2 - t1, img.size(0))
        nms_time.update(t4 - t3, img.size(0))
        waste_time.update(tw2 - tw1, img.size(0))
        print(f'Total Frames: {total_frames}, Processed Frames: {processed_frames}')
        print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg, nms_time.avg))
        print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == '__main__':
    opt = make_parser().parse_args()
    print(opt)

    with torch.no_grad():
        detect()
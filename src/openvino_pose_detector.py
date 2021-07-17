import cv2
import numpy as np
from math import floor
from pose_utils.pipelines import get_user_config, AsyncPipeline
from pose_utils import models
from openvino.inference_engine import IECore


SKELETON = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
)


def draw_poses(poses, frame, resize_ratios=(1, 1), min_threshold=0.2):
    for pose in poses:
        # Take only 2 first columns, containing plane coords
        points = (pose[:, :2] * resize_ratios).astype(np.int32)
        scores = pose[:, 2]
        # Draw joints.
        for i, j in SKELETON:
            if scores[i] > min_threshold and scores[j] > min_threshold:
                cv2.line(frame, tuple(points[i]), tuple(points[j]), color=(0, 200, 0), thickness=2)
        for i, p in enumerate(points):
            if scores[i] > min_threshold:
                cv2.circle(frame, tuple(p), 2, (0, 255, 255), 4)


def launch_detection_on_capture(capture, args):
    plugin_config = get_user_config(args["device"], '', None)
    ie = IECore()

    # prepare model params
    ret, frame = capture.read()
    if not ret:
        raise IOError('Can not read frame!')

    aspect_ratio = frame.shape[1] / frame.shape[0]
    if aspect_ratio >= 1:
        target_size = floor(frame.shape[0] * args["net_input_width"] / frame.shape[1])
    else:
        target_size = args["net_input_width"]

    model = models.HpeAssociativeEmbedding(ie, args["model_path"],
                                           aspect_ratio=aspect_ratio,
                                           target_size=target_size, prob_threshold=0.1)
    hpe_pipeline = AsyncPipeline(ie, model, plugin_config, device=args["device"], max_num_requests=1)

    net_input_size = (model.w, model.h)
    resize_ratios = (frame.shape[1] / net_input_size[0]), (frame.shape[0] / net_input_size[1])

    while True:
        ret, frame = capture.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, net_input_size, interpolation=cv2.INTER_AREA)

        hpe_pipeline.submit_data(resized_frame, 0, {'frame': resized_frame, 'start_time': 0})
        hpe_pipeline.await_any()

        results = hpe_pipeline.get_result(0)
        if results:
            (poses, scores), frame_meta = results
            if len(poses) > 0:
                draw_poses(poses, frame, resize_ratios)

        frame = cv2.flip(frame, 1)
        cv2.imshow("Just Dance", frame)

        if cv2.waitKey(1) == ord("q"):
            break


def launch_detection_on_webcam(args):
    capture = cv2.VideoCapture(args["cap_source"])
    if not capture.isOpened():
        raise IOError('Camera is not accessible')

    launch_detection_on_capture(capture, args)

    cv2.destroyAllWindows()
    capture.release()


if __name__ == "__main__":
    launch_detection_on_webcam({
        "cap_source": 0,
        "model_path": f"{__file__}/../../models/intel/human-pose-estimation-0007/FP16/human-pose-estimation-0007.xml",
        "device": "CPU",
        "net_input_width": 256
    })

import cv2
import numpy as np
from time import perf_counter
from math import floor
from pose_utils.pipelines import get_user_config, AsyncPipeline
from pose_utils import models
from openvino.inference_engine import IECore


def launch_detection_on_capture(capture, args):
    plugin_config = get_user_config(args["device"], '', None)
    ie = IECore()
    start_time = perf_counter()

    # prepare model params
    ret, frame = capture.read()

    aspect_ratio = frame.shape[1] / frame.shape[0]
    if aspect_ratio >= 1:
        target_size = floor(frame.shape[0] * args["net_input_width"] / frame.shape[1])
    else:
        target_size = args["net_input_width"]

    if not ret:
        raise IOError('Can not read frame!')

    model = models.HpeAssociativeEmbedding(ie, args["model_path"],
                                           aspect_ratio=aspect_ratio,
                                           target_size=target_size, prob_threshold=0.1)

    # cv2.resize() takes (width, height) as new size,
    # but img.shape has (height, width) format
    net_input_size = (model.w, model.h)
    frame = cv2.resize(frame, net_input_size, interpolation=cv2.INTER_AREA)

    show_frame_size = (1080, floor(frame.shape[0] * 1080 / frame.shape[1])) # width height

    hpe_pipeline = AsyncPipeline(ie, model, plugin_config, device=args["device"], max_num_requests=0)

    hpe_pipeline.submit_data(frame, 0, {'frame': frame, 'start_time': start_time})
    next_frame_id = 1
    next_frame_id_to_show = 0

    while True:
        if hpe_pipeline.callback_exceptions:
            raise hpe_pipeline.callback_exceptions[0]

        # Process all completed requests
        results = hpe_pipeline.get_result(next_frame_id_to_show)
        if results:
            (poses, scores), frame_meta = results
            frame = frame_meta['frame']
            if len(poses) > 0:
                for pose in poses:
                    points = pose[:, :2].astype(np.int32)
                    # Draw joints.
                    for p in points:
                        cv2.circle(frame, tuple(p), 1, (0, 255, 0), 2)

            start_time = frame_meta['start_time']
            next_frame_id_to_show += 1
            cv2.imshow("Show", cv2.resize(frame, show_frame_size, interpolation=cv2.INTER_AREA))
            continue

        if hpe_pipeline.is_ready():
            # Get new image/frame
            start_time = perf_counter()
            ret, frame = capture.read()
            if not ret:
                break
            frame = cv2.resize(frame, net_input_size, interpolation=cv2.INTER_AREA)
            if frame is None:
                break

            # Submit for inference
            hpe_pipeline.submit_data(frame, next_frame_id, {'frame': frame, 'start_time': start_time})
            next_frame_id += 1

        else:
            # Wait for empty request
            hpe_pipeline.await_any()

        if cv2.waitKey(30) == ord("q"):
            break


def launch_detection_on_webcam(args):
    capture = cv2.VideoCapture(args["cap_source"])
    if not capture.isOpened():
        raise IOError('Camera is not accessible')

    launch_detection_on_capture(capture, args)

    cv2.destroyAllWindows()
    capture.release()


if __name__ == "__main__":
    launch_detection_on_webcam({"cap_source": 0,
                                "model_path": "pose_utils/human-pose-estimation-0007.xml",
                                "device": "CPU",
                                "net_input_width": 512})
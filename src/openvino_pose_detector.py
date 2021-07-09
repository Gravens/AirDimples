import cv2
import numpy as np
from time import perf_counter
from pose_utils.pipelines import get_user_config, AsyncPipeline
from pose_utils import models
from openvino.inference_engine import IECore


def launch_detection_on_capture(capture, args):
    plugin_config = get_user_config(args["device"], '', None)
    ie = IECore()
    start_time = perf_counter()

    ret, frame = capture.read()
    model = models.HpeAssociativeEmbedding(ie, args["model_path"],
                                           aspect_ratio=frame.shape[1] / frame.shape[0],
                                           target_size=None, prob_threshold=0.1)
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
            cv2.imshow("rwer", frame)
            continue

        if hpe_pipeline.is_ready():
            # Get new image/frame
            start_time = perf_counter()
            ret, frame = capture.read()
            if frame is None:
                break

            # Submit for inference
            hpe_pipeline.submit_data(frame, next_frame_id, {'frame': frame, 'start_time': start_time})
            next_frame_id += 1

        else:
            # Wait for empty request
            hpe_pipeline.await_any()

        if cv2.waitKey(100) == ord("q"):
            break


def launch_detection_on_webcam(args):
    capture = cv2.VideoCapture(args["cap_source"])
    if not capture.isOpened():
        raise IOError('Camera is not accessible')

    launch_detection_on_capture(capture, args)

    capture.release()


if __name__ == "__main__":
    launch_detection_on_webcam({"cap_source": 0,
                                "model_path": "pose_utils/human-pose-estimation-0007.xml",
                                "device": "CPU"})
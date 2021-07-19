import cv2
from math import floor

import utils
from gameplay import SoloClassic
from models.intel_pose import IntelPoseModel
from pose_utils.pipelines import get_user_config, AsyncPipeline
from pose_utils import models
from openvino.inference_engine import IECore


def get_capture_shape(capture):
    ret, frame = capture.read()
    if not ret:
        raise IOError("Can't read initial frame")
    return frame.shape


def launch_detection_on_capture(capture, args):
    # Initialize Inference Engine
    ie = IECore()
    plugin_config = get_user_config(args["device"], '', None)
    model = IntelPoseModel()

    # Prepare model parameters
    cap_height, cap_width, _ = get_capture_shape(capture)
    aspect_ratio = cap_width / cap_height
    if aspect_ratio >= 1:
        target_size = floor(cap_height * args["net_input_width"] / cap_width)
    else:
        target_size = args["net_input_width"]

    model_embedding = models.HpeAssociativeEmbedding(
        ie, args["model_path"], aspect_ratio=aspect_ratio,
        target_size=target_size, prob_threshold=0.1
    )

    # Initialize pipeline
    hpe_pipeline = AsyncPipeline(ie, model_embedding, plugin_config, device=args["device"], max_num_requests=1)
    net_input_size = (model_embedding.w, model_embedding.h)

    game = SoloClassic(
        [cap_height, cap_width, _],
        circle_radius=50,
        life_time=1,
        max_items=10,
        body_part_indexes=model.body_part_indexes
    )

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            print("Received empty camera frame")
            break

        frame = cv2.flip(frame, 1)
        resized_frame = cv2.resize(frame, net_input_size, interpolation=cv2.INTER_AREA)

        hpe_pipeline.submit_data(resized_frame, 0, {'frame': resized_frame, 'start_time': 0})
        hpe_pipeline.await_any()

        results = hpe_pipeline.get_result(0)

        joints = model.get_joints_from_result(results)

        utils.draw_joints(frame, joints, model.SKELETON)

        game_status = game.process(frame, joints)

        cv2.imshow("Just Dance", frame)

        if cv2.waitKey(1) == ord("q") or not game_status:
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

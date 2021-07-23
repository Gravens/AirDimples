import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from typing.io import IO

from models.base_pose import PoseModel
from models.intel_pose import IntelPoseModel


@dataclass
class App:
    model: PoseModel
    root_path: Path = Path(__file__).parent.parent
    model_path: Path = root_path / 'models/intel/human-pose-estimation-0007/FP16/human-pose-estimation-0007.xml'
    neural_network_input_width: int = 256
    inference_device: str = 'CPU'
    max_frames_stored: int = 5
    max_joints_stored: int = 5
    window_name: str = 'Just Dance'
    quit_key: str = 'ctrl'
    flip_image: bool = True
    detection_threshold = 0.2
    log_level: int = logging.DEBUG
    log_stream: IO[str] = sys.stdout


@dataclass
class Graphics:
    hand_color: Tuple[int] = (122, 36, 27)
    foot_color: Tuple[int] = (15, 255, 235)
    joint_color: Tuple[int] = (0, 0, 255)
    joint_radius: int = 2
    joint_thickness: int = 2
    connection_color: Tuple[int] = (0, 255, 0)
    connection_thickness: int = 2
    label_default_color: Tuple[int] = (255, 51, 51)
    label_clicked_color: Tuple[int] = (51, 255, 51)
    button_default_color: Tuple[int] = (255, 51, 51)
    button_clicked_color: Tuple[int] = (51, 255, 51)
    countdown_label_color: Tuple[int] = (0, 255, 255)
    countdown_label_font_scale: int = 2
    countdown_label_thickness: int = 2


@dataclass
class InputBenchmarking:
    enabled: bool = True
    frame_count: int = 80
    default_fps: int = 24


@dataclass
class Gameplay:
    foot_circles_enabled: bool = True
    classic_max_circles_destroyed: int = 20
    classic_circle_life_time: int = 2
    intensive_interval: int = 3
    intensive_max_circles_on_screen: int = 5
    circle_radius: int = 44
    pacman_speed: int = 5
    pacman_max_progress: int = 300


@dataclass
class Config:
    app: App
    graphics: Graphics
    input_benchmarking: InputBenchmarking
    gameplay: Gameplay


config = Config(
    App(model=IntelPoseModel()),
    Graphics(),
    InputBenchmarking(),
    Gameplay()
)

"""MuJoCo robot simulator with automatic position control for Panda-Omron mobile manipulator."""

import time
import threading
import numpy as np
import glfw
import mujoco, mujoco.viewer
from scipy.spatial.transform import Rotation as R
from typing import Optional, List, Tuple, Dict, Any
from simulator_util import PathPlanner, GridMapUtils


class RobotConfig:
    """Robot simulation configuration constants."""

    # Mobile base joints: [x, y, theta]
    MOBILE_JOINT_NAMES = [
        "mobilebase0_joint_mobile_side",
        "mobilebase0_joint_mobile_forward",
        "mobilebase0_joint_mobile_yaw"
    ]

    MOBILE_ACTUATOR_NAMES = [
        "mobilebase0_actuator_mobile_side",
        "mobilebase0_actuator_mobile_forward",
        "mobilebase0_actuator_mobile_yaw"
    ]

    # Panda arm joints: [joint1 ~ joint7]
    ARM_JOINT_NAMES = [
        "robot0_joint1",
        "robot0_joint2",
        "robot0_joint3",
        "robot0_joint4",
        "robot0_joint5",
        "robot0_joint6",
        "robot0_joint7"
    ]

    ARM_ACTUATOR_NAMES = [
        "robot0_torq_j1",
        "robot0_torq_j2",
        "robot0_torq_j3",
        "robot0_torq_j4",
        "robot0_torq_j5",
        "robot0_torq_j6",
        "robot0_torq_j7"
    ]

    # End effector site name
    EE_SITE_NAME = "gripper0_right_grip_site"

    # Gripper actuator names (2-finger parallel gripper)
    GRIPPER_ACTUATOR_NAMES = [
        "gripper0_right_gripper_finger_joint1",
        "gripper0_right_gripper_finger_joint2"
    ]

    # Mobile PID controller gains
    MOBILE_KP = np.array([2.00, 2.00, 1.50])
    MOBILE_KI = np.array([0.30, 0.30, 0.01])
    MOBILE_I_LIMIT = np.array([0.60, 0.60, 0.02])
    MOBILE_KD = np.array([1.00, 1.00, 0.50])

    # Arm PID controller gains for position control (7 joints)
    ARM_KP = np.array([2.0, 2.0, 2.0, 2.0, 1.5, 1.0, 1.0])
    ARM_KI = np.array([0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05])
    ARM_I_LIMIT = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1])
    ARM_KD = np.array([0.05, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01])
    ARM_JOINT_LIMITS = np.array([[-2.9, 2.9]] * 7)

    # IK solver parameters
    IK_MAX_ITERATIONS = 100
    IK_POSITION_TOLERANCE = 0.001  # 1mm
    IK_ORIENTATION_TOLERANCE = 0.01  # ~0.57 degrees
    IK_DAMPING = 0.01  # Damped Least Squares damping factor
    IK_STEP_SIZE = 0.5  # Step size for joint updates

    # Camera settings
    CAM_LOOKAT = [-0.8, -0.8, 0.8]
    CAM_DISTANCE = 7.5
    CAM_AZIMUTH = 135
    CAM_ELEVATION = -25
    EYE_IN_HAND_CAMERA_NAME = "robot0_eye_in_hand"
    DEBUG_CAMERA_NAME = "robot0_debug_head_camera"
    CAMERA_DEFAULT_WIDTH = 320
    CAMERA_DEFAULT_HEIGHT = 240
    DEBUG_CAMERA_PAN_JOINT_NAME = "robot0_debug_camera_pan_joint"
    DEBUG_CAMERA_TILT_JOINT_NAME = "robot0_debug_camera_tilt_joint"
    DEBUG_CAMERA_PAN_ACTUATOR_NAME = "robot0_debug_camera_pan_actuator"
    DEBUG_CAMERA_TILT_ACTUATOR_NAME = "robot0_debug_camera_tilt_actuator"
    DEBUG_CAMERA_PAN_BODY_NAME = "robot0_debug_camera_pan_link"
    DEBUG_CAMERA_TILT_BODY_NAME = "robot0_debug_camera_tilt_link"
    DEBUG_CAMERA_TILT_OFFSET_LOCAL = np.array([0.08, 0.0, 0.0])
    DEBUG_CAMERA_DEFAULT_TARGET_JOINT = np.array([0.0, -0.15])
    DEBUG_CAMERA_MANUAL_STEP_DEG = 3.0
    DEBUG_CAMERA_CONVERGENCE_POS_THRESHOLD = 0.01
    DEBUG_CAMERA_CONVERGENCE_VEL_THRESHOLD = 0.05
    DEPTH_EPSILON = 1e-6
    POINT_CLOUD_DEFAULT_STRIDE = 4
    POINT_CLOUD_DEFAULT_MAX_DEPTH = 4.0
    POINT_CLOUD_DEFAULT_FRAME = "world"
    CAMERA_FRAME_CONVENTION = "opencv"
    VIEWER_KEY_TOGGLE = glfw.KEY_F8
    VIEWER_KEY_TOGGLE_WASD_DEBUG = glfw.KEY_F7
    VIEWER_KEY_THIRD_PERSON = glfw.KEY_F9
    VIEWER_KEY_ROBOT_EYE = glfw.KEY_F10
    VIEWER_KEY_ROBOT_EYE_DEBUG = glfw.KEY_F11
    VIEWER_KEY_TOGGLE_CONTROL_DEBUG = glfw.KEY_F12
    VIEWER_ROBOT_EYE_DEBUG_DISTANCE = 0.35

    MOBILE_INIT_POSITION = np.array([-1.0, 1.0, 0.0])
    ARM_INIT_POSITION = np.array([-0.0114, -1.0319,  0.0488, -2.2575,  0.0673,  1.5234, 0.6759])
    GRIPPER_INIT_WIDTH = 0.08

    # Mobile base physical dimensions
    MOBILE_BASE_RADIUS = 0.35  # Approximate radius of mobile base footprint in meters

    # Grid map parameters
    GRID_SIZE = 0.1  # Grid cell size in meters


class MujocoSimulator:
    """MuJoCo simulator with PD-controlled mobile base position tracking."""

    def __init__(self, xml_path: str = "../model/robocasa/site.xml") -> None:
        """Initialize simulator with MuJoCo model and control indices."""
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)
        self._mobile_target_position = RobotConfig.MOBILE_INIT_POSITION.copy()
        self._arm_target_joint = RobotConfig.ARM_INIT_POSITION.copy()
        self._gripper_target_width = RobotConfig.GRIPPER_INIT_WIDTH
        self.dt = self.model.opt.timestep # PID timestep
        self._mobile_error_integral = np.zeros(3,) # I of PID for mobile base
        self._arm_error_integral = np.zeros(7,) # I of PID for arm

        # Resolve joint/actuator names to indices
        self.mobile_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                                 for name in RobotConfig.MOBILE_JOINT_NAMES]
        self.mobile_actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                                    for name in RobotConfig.MOBILE_ACTUATOR_NAMES]
        
        # Resolve Panda arm joint IDs and set initial positions
        self.arm_joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                              for name in RobotConfig.ARM_JOINT_NAMES]
        self.arm_actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                                 for name in RobotConfig.ARM_ACTUATOR_NAMES]
        self.arm_dof_indices = []
        for joint_id in self.arm_joint_ids:
            dof_adr = self.model.jnt_dofadr[joint_id]
            dof_num = self._get_joint_dof_count(joint_id)
            self.arm_dof_indices.extend(range(dof_adr, dof_adr + dof_num))

        # Resolve end effector site ID
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, RobotConfig.EE_SITE_NAME)
        # Body used to measure the mobile base pose in world coordinates
        self.mobile_base_center_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "mobile_base_center"
        )

        # Resolve gripper actuator IDs
        self.gripper_actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                                     for name in RobotConfig.GRIPPER_ACTUATOR_NAMES]

        # Resolve debug camera pan/tilt rig IDs
        self.debug_camera_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, RobotConfig.DEBUG_CAMERA_PAN_JOINT_NAME),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, RobotConfig.DEBUG_CAMERA_TILT_JOINT_NAME),
        ]
        self.debug_camera_actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, RobotConfig.DEBUG_CAMERA_PAN_ACTUATOR_NAME),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, RobotConfig.DEBUG_CAMERA_TILT_ACTUATOR_NAME),
        ]
        self.debug_camera_qpos_indices = [
            self.model.jnt_qposadr[jid] for jid in self.debug_camera_joint_ids
        ]
        self.debug_camera_dof_indices = [
            self.model.jnt_dofadr[jid] for jid in self.debug_camera_joint_ids
        ]
        self.debug_camera_joint_limits = np.array(
            [self.model.jnt_range[jid] for jid in self.debug_camera_joint_ids],
            dtype=float
        )
        self.debug_camera_pan_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, RobotConfig.DEBUG_CAMERA_PAN_BODY_NAME
        )
        self.debug_camera_pan_parent_body_id = int(self.model.body_parentid[self.debug_camera_pan_body_id])
        self.debug_camera_tilt_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, RobotConfig.DEBUG_CAMERA_TILT_BODY_NAME
        )
        self.debug_camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, RobotConfig.DEBUG_CAMERA_NAME
        )
        for jid, name in zip(
            self.debug_camera_joint_ids,
            [RobotConfig.DEBUG_CAMERA_PAN_JOINT_NAME, RobotConfig.DEBUG_CAMERA_TILT_JOINT_NAME]
        ):
            self._require_valid_id(jid, name)
        for aid, name in zip(
            self.debug_camera_actuator_ids,
            [RobotConfig.DEBUG_CAMERA_PAN_ACTUATOR_NAME, RobotConfig.DEBUG_CAMERA_TILT_ACTUATOR_NAME]
        ):
            self._require_valid_id(aid, name)
        self._require_valid_id(self.debug_camera_pan_body_id, RobotConfig.DEBUG_CAMERA_PAN_BODY_NAME)
        self._require_valid_id(self.debug_camera_tilt_body_id, RobotConfig.DEBUG_CAMERA_TILT_BODY_NAME)
        self._require_valid_id(self.debug_camera_id, RobotConfig.DEBUG_CAMERA_NAME)
        self._require_valid_id(self.debug_camera_pan_parent_body_id, f"{RobotConfig.DEBUG_CAMERA_PAN_BODY_NAME}_parent")

        # Resolve object IDs
        self.object_ids = []
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name.startswith("object_"):
                self.object_ids.append(i)
        
        # Set initial mobile base positions (qpos) and velocities (ctrl=0 for velocity control)
        for i, (joint_id, actuator_id) in enumerate(zip(self.mobile_joint_ids, self.mobile_actuator_ids)):
            self.data.qpos[joint_id] = RobotConfig.MOBILE_INIT_POSITION[i]
            self.data.ctrl[actuator_id] = 0.0  # velocity control is 0.0

        # Set initial joint positions (qpos) and actuator targets (ctrl)
        for i, (joint_id, actuator_id) in enumerate(zip(self.arm_joint_ids, self.arm_actuator_ids)):
            self.data.qpos[joint_id] = RobotConfig.ARM_INIT_POSITION[i]
            self.data.ctrl[actuator_id] = RobotConfig.ARM_INIT_POSITION[i]

        # Initialize debug camera pan/tilt joint targets
        self._debug_camera_target_joint = np.clip(
            RobotConfig.DEBUG_CAMERA_DEFAULT_TARGET_JOINT.copy(),
            self.debug_camera_joint_limits[:, 0],
            self.debug_camera_joint_limits[:, 1],
        )
        for i, qpos_idx in enumerate(self.debug_camera_qpos_indices):
            self.data.qpos[qpos_idx] = self._debug_camera_target_joint[i]
            self.data.ctrl[self.debug_camera_actuator_ids[i]] = self._debug_camera_target_joint[i]
        
        # Initialize grid map
        self.grid_map = np.load("grid_map.npy")

        # Cache floor geometry information to avoid repeated queries
        self._floor_geom_id = None
        self._floor_size = None
        self._floor_pos = None

        # Compute forward kinematics
        mujoco.mj_forward(self.model, self.data)

        # Offscreen renderers for vision pipeline (cached by resolution)
        self._renderers: Dict[Tuple[int, int], mujoco.Renderer] = {}
        self._render_lock = threading.Lock()
        self._viewer_camera_mode = "third_person"
        self._eye_in_hand_camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, RobotConfig.EYE_IN_HAND_CAMERA_NAME
        )
        self._viewer_command_lock = threading.Lock()
        self._viewer_pending_camera_command: Optional[str] = None
        self._viewer_show_control_debug = True
        self._viewer_wasd_debug_enabled = False
        self._viewer_overlay_update_period = 0.15
        self._viewer_overlay_last_update_time = 0.0

    def _set_viewer_camera_mode(self, viewer: Any, mode: str) -> None:
        """Apply viewer camera mode."""
        if mode == "robot_eye":
            if self._eye_in_hand_camera_id < 0:
                print(
                    f"[VIEWER] '{RobotConfig.EYE_IN_HAND_CAMERA_NAME}' camera not found. "
                    "Staying in third-person mode."
                )
                self._viewer_camera_mode = "third_person"
                mode = "third_person"
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = self._eye_in_hand_camera_id
                self._viewer_camera_mode = "robot_eye"
                return
        elif mode == "robot_eye_debug":
            if self._eye_in_hand_camera_id < 0:
                print(
                    f"[VIEWER] '{RobotConfig.EYE_IN_HAND_CAMERA_NAME}' camera not found. "
                    "Staying in third-person mode."
                )
                self._viewer_camera_mode = "third_person"
                mode = "third_person"
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.distance = RobotConfig.VIEWER_ROBOT_EYE_DEBUG_DISTANCE
                self._viewer_camera_mode = "robot_eye_debug"
                self._update_robot_eye_debug_camera(viewer, force_distance=True)
                return

        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = RobotConfig.CAM_LOOKAT
        viewer.cam.distance = RobotConfig.CAM_DISTANCE
        viewer.cam.azimuth = RobotConfig.CAM_AZIMUTH
        viewer.cam.elevation = RobotConfig.CAM_ELEVATION
        self._viewer_camera_mode = "third_person"

    def _toggle_viewer_camera_mode(self, viewer: Any) -> None:
        """Toggle viewer camera between third-person and eye-in-hand views."""
        next_mode = "robot_eye" if self._viewer_camera_mode == "third_person" else "third_person"
        self._set_viewer_camera_mode(viewer, next_mode)
        if self._viewer_camera_mode == "robot_eye":
            print(
                "[VIEWER] Camera mode: robot_eye "
                f"({RobotConfig.EYE_IN_HAND_CAMERA_NAME}). Press 'F8' to switch back."
            )
        elif self._viewer_camera_mode == "robot_eye_debug":
            print(
                "[VIEWER] Camera mode: robot_eye_debug. "
                "Mouse drag rotates view, but camera follows eye-in-hand."
            )
        else:
            print("[VIEWER] Camera mode: third_person. Press 'F8' to switch to robot_eye.")

    def _format_control_debug_overlay(self) -> str:
        """Build compact control debug text."""
        mobile_pos = self.get_mobile_position()
        mobile_target = self.get_mobile_target_position()
        mobile_error = self.get_mobile_position_diff()
        mobile_vel = self.get_mobile_velocity()

        arm_error = self.get_arm_joint_diff()
        arm_vel = self.get_arm_joint_velocity()

        gripper_now = self.get_gripper_width()
        gripper_target = self._gripper_target_width
        cam_joint = self.get_debug_camera_joint_position()
        cam_target = self.get_debug_camera_target_joint()
        cam_error = self.get_debug_camera_joint_diff()

        return (
            f"base cur : [{mobile_pos[0]:+.2f}, {mobile_pos[1]:+.2f}, {mobile_pos[2]:+.2f}]\n"
            f"base tgt : [{mobile_target[0]:+.2f}, {mobile_target[1]:+.2f}, {mobile_target[2]:+.2f}]\n"
            f"base err : [{mobile_error[0]:+.2f}, {mobile_error[1]:+.2f}, {mobile_error[2]:+.2f}] "
            f"| |e|={np.linalg.norm(mobile_error):.3f}\n"
            f"base vel : |v|={np.linalg.norm(mobile_vel):.3f}\n"
            f"arm err  : |e|={np.linalg.norm(arm_error):.3f}\n"
            f"arm vel  : |v|={np.linalg.norm(arm_vel):.3f}\n"
            f"gripper  : now={gripper_now:.3f}, tgt={gripper_target:.3f}\n"
            f"cam pan/tilt cur: [{cam_joint[0]:+.2f}, {cam_joint[1]:+.2f}]\n"
            f"cam pan/tilt tgt: [{cam_target[0]:+.2f}, {cam_target[1]:+.2f}] "
            f"| |e|={np.linalg.norm(cam_error):.3f}"
        )

    def _set_viewer_overlay(self, viewer: Any) -> None:
        """Show viewer controls and optional control-debug overlay."""
        texts = [
            (
                None,
                mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                "Viewer",
                f"F7: WASD mode {'ON' if self._viewer_wasd_debug_enabled else 'OFF'}\n"
                "F8: Toggle third <-> eye\n"
                "F9: Third person\n"
                "F10: Robot eye (fixed)\n"
                "F11: Robot eye debug (free look)\n"
                f"F12: Control debug {'ON' if self._viewer_show_control_debug else 'OFF'}\n"
                "W/A/S/D: cam pan/tilt (when WASD mode ON)\n"
                f"Mode: {self._viewer_camera_mode}",
            )
        ]
        if self._viewer_show_control_debug:
            texts.append(
                (
                    None,
                    mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                    "Control Debug",
                    self._format_control_debug_overlay(),
                )
            )
        viewer.set_texts(texts)

    def _update_robot_eye_debug_camera(self, viewer: Any, force_distance: bool = False) -> None:
        """Keep free camera centered on eye-in-hand for mouse-look debugging."""
        if self._eye_in_hand_camera_id < 0:
            return
        eye_pos = self.data.cam_xpos[self._eye_in_hand_camera_id]
        viewer.cam.lookat[:] = eye_pos
        if force_distance:
            viewer.cam.distance = RobotConfig.VIEWER_ROBOT_EYE_DEBUG_DISTANCE
        else:
            viewer.cam.distance = max(viewer.cam.distance, 0.02)

    def _queue_viewer_camera_command(self, command: str) -> None:
        """Queue a viewer camera command to be applied in the simulation thread."""
        with self._viewer_command_lock:
            self._viewer_pending_camera_command = command

    def _pop_viewer_camera_command(self) -> Optional[str]:
        """Pop queued camera command if available."""
        with self._viewer_command_lock:
            command = self._viewer_pending_camera_command
            self._viewer_pending_camera_command = None
        return command

    def set_viewer_camera_mode(self, mode: str) -> None:
        """Request viewer camera mode change from any thread."""
        mode_alias = {
            "third": "third_person",
            "third_person": "third_person",
            "robot": "robot_eye",
            "robot_eye": "robot_eye",
            "robot_eye_debug": "robot_eye_debug",
            "debug": "robot_eye_debug",
            "inspect": "robot_eye_debug",
            "eye": "robot_eye",
            "toggle": "toggle",
        }
        normalized = mode_alias.get(mode.strip().lower())
        if normalized is None:
            raise ValueError("mode must be one of: third_person, robot_eye, robot_eye_debug, toggle")
        self._queue_viewer_camera_command(normalized)

    def toggle_viewer_camera_mode(self) -> None:
        """Request camera toggle from any thread."""
        self._queue_viewer_camera_command("toggle")

    def toggle_viewer_control_debug(self) -> None:
        """Toggle control debug overlay from any thread."""
        self._queue_viewer_camera_command("toggle_control_debug")

    def toggle_viewer_wasd_debug_mode(self) -> None:
        """Toggle WASD debug-camera input mode from any thread."""
        self._queue_viewer_camera_command("toggle_wasd_debug")

    def _apply_viewer_camera_command(self, viewer: Any, command: str) -> None:
        """Apply queued command on viewer state."""
        if command == "toggle":
            self._toggle_viewer_camera_mode(viewer)
            return
        if command == "toggle_control_debug":
            self._viewer_show_control_debug = not self._viewer_show_control_debug
            status = "ON" if self._viewer_show_control_debug else "OFF"
            print(f"[VIEWER] Control debug overlay: {status}.")
            return
        if command == "toggle_wasd_debug":
            self._viewer_wasd_debug_enabled = not self._viewer_wasd_debug_enabled
            status = "ON" if self._viewer_wasd_debug_enabled else "OFF"
            print(f"[VIEWER] WASD debug-camera mode: {status}.")
            return
        if command == "debug_cam_pan_left":
            self._nudge_debug_camera_target(delta_pan=+np.deg2rad(RobotConfig.DEBUG_CAMERA_MANUAL_STEP_DEG))
            return
        if command == "debug_cam_pan_right":
            self._nudge_debug_camera_target(delta_pan=-np.deg2rad(RobotConfig.DEBUG_CAMERA_MANUAL_STEP_DEG))
            return
        if command == "debug_cam_tilt_up":
            self._nudge_debug_camera_target(delta_tilt=+np.deg2rad(RobotConfig.DEBUG_CAMERA_MANUAL_STEP_DEG))
            return
        if command == "debug_cam_tilt_down":
            self._nudge_debug_camera_target(delta_tilt=-np.deg2rad(RobotConfig.DEBUG_CAMERA_MANUAL_STEP_DEG))
            return
        self._set_viewer_camera_mode(viewer, command)
        if self._viewer_camera_mode == "robot_eye":
            print(
                "[VIEWER] Camera mode: robot_eye "
                f"({RobotConfig.EYE_IN_HAND_CAMERA_NAME})."
            )
        elif self._viewer_camera_mode == "robot_eye_debug":
            print("[VIEWER] Camera mode: robot_eye_debug (free look following eye-in-hand).")
        else:
            print("[VIEWER] Camera mode: third_person.")

    def _get_joint_dof_count(self, joint_id: int) -> int:
        joint_type = self.model.jnt_type[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            return 6
        if joint_type == mujoco.mjtJoint.mjJNT_BALL:
            return 3
        if joint_type in (mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE):
            return 1
        raise ValueError(f"Unsupported joint type for joint_id {joint_id}")

    @staticmethod
    def _wrap_to_pi(value: float) -> float:
        """Wrap angle to [-pi, pi]."""
        return float(np.arctan2(np.sin(value), np.cos(value)))

    @staticmethod
    def _build_homogeneous(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """Build 4x4 homogeneous transform matrix."""
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation
        return matrix

    @staticmethod
    def _camera_mujoco_to_opencv_rotation() -> np.ndarray:
        """Rotation from MuJoCo camera frame to OpenCV-like camera frame."""
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=float,
        )

    def _require_valid_id(self, id_value: int, name: str) -> None:
        """Raise a descriptive error if a MuJoCo name lookup failed."""
        if id_value < 0:
            raise ValueError(f"Required model element '{name}' was not found.")

    def _resolve_camera_id(self, camera_name: Optional[str]) -> int:
        """Resolve camera name to camera id."""
        if not camera_name:
            raise ValueError("camera_name is required for this operation")
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            raise ValueError(f"Unknown camera name: {camera_name}")
        return cam_id

    # ============================================================
    # Debug Camera Rig Methods
    # ============================================================

    def get_debug_camera_joint_position(self) -> np.ndarray:
        """Get current debug camera joint angles [pan, tilt] in radians."""
        return np.array([self.data.qpos[idx] for idx in self.debug_camera_qpos_indices], dtype=float)

    def get_debug_camera_target_joint(self) -> np.ndarray:
        """Get current debug camera target angles [pan, tilt] in radians."""
        return self._debug_camera_target_joint.copy()

    def set_debug_camera_target_joint(self, target_joint: np.ndarray) -> None:
        """Set debug camera target angles [pan, tilt] in radians."""
        target = np.array(target_joint, dtype=float).reshape(-1)
        if target.shape[0] != 2:
            raise ValueError("target_joint must have shape [2] for [pan, tilt]")
        clipped = np.clip(target, self.debug_camera_joint_limits[:, 0], self.debug_camera_joint_limits[:, 1])
        self._debug_camera_target_joint[:] = clipped

    def get_debug_camera_joint_diff(self) -> np.ndarray:
        """Get debug camera joint errors [pan, tilt] in radians."""
        diff = self._debug_camera_target_joint - self.get_debug_camera_joint_position()
        diff[0] = self._wrap_to_pi(diff[0])  # pan wraps around naturally
        return diff

    def get_debug_camera_joint_velocity(self) -> np.ndarray:
        """Get debug camera joint velocities [pan, tilt] in rad/s."""
        return np.array([self.data.qvel[idx] for idx in self.debug_camera_dof_indices], dtype=float)

    def _compute_debug_camera_control(self) -> np.ndarray:
        """Compute debug camera control commands [pan, tilt]."""
        return self._debug_camera_target_joint.copy()

    def _nudge_debug_camera_target(self, delta_pan: float = 0.0, delta_tilt: float = 0.0) -> None:
        """Increment debug camera target by small deltas."""
        target = self.get_debug_camera_target_joint()
        target[0] += delta_pan
        target[1] += delta_tilt
        self.set_debug_camera_target_joint(target)

    def look_at_point(self, target_xyz: np.ndarray) -> np.ndarray:
        """Set debug camera target so the rig looks at a world-space point."""
        target = np.array(target_xyz, dtype=float).reshape(-1)
        if target.shape[0] != 3:
            raise ValueError("target_xyz must have shape [3] for [x, y, z]")

        self._require_valid_id(self.debug_camera_pan_body_id, RobotConfig.DEBUG_CAMERA_PAN_BODY_NAME)

        # Use pan-parent frame to compute desired yaw (pan).
        pan_origin_world = self.data.xpos[self.debug_camera_pan_body_id].copy()
        pan_parent_rot_world = self.data.xmat[self.debug_camera_pan_parent_body_id].reshape(3, 3).copy()
        vec_world = target - pan_origin_world
        vec_parent = pan_parent_rot_world.T @ vec_world
        pan = np.arctan2(vec_parent[1], vec_parent[0])

        # Tilt is computed in pan-rotated frame from tilt joint location.
        pan_rot_local = R.from_euler("z", pan).as_matrix()
        vec_after_pan = pan_rot_local.T @ vec_parent
        vec_from_tilt = vec_after_pan - RobotConfig.DEBUG_CAMERA_TILT_OFFSET_LOCAL
        tilt = np.arctan2(-vec_from_tilt[2], vec_from_tilt[0])

        target_joint = np.array([pan, tilt], dtype=float)
        self.set_debug_camera_target_joint(target_joint)
        return self.get_debug_camera_target_joint()
    
    # ============================================================
    # Vision / Camera Methods
    # ============================================================

    def list_cameras(self) -> List[str]:
        """List named cameras available in the MuJoCo model."""
        cameras = []
        for i in range(self.model.ncam):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            if name:
                cameras.append(name)
        return cameras

    def _get_renderer(self, width: int, height: int) -> mujoco.Renderer:
        """Get cached offscreen renderer for resolution, creating if needed."""
        key = (width, height)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, width=width, height=height)
        return self._renderers[key]

    @staticmethod
    def _build_free_camera() -> mujoco.MjvCamera:
        """Build a free camera that matches viewer defaults."""
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = RobotConfig.CAM_LOOKAT
        cam.distance = RobotConfig.CAM_DISTANCE
        cam.azimuth = RobotConfig.CAM_AZIMUTH
        cam.elevation = RobotConfig.CAM_ELEVATION
        return cam

    def get_camera_intrinsics(
        self,
        width: int = RobotConfig.CAMERA_DEFAULT_WIDTH,
        height: int = RobotConfig.CAMERA_DEFAULT_HEIGHT,
        camera_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return pinhole intrinsics for a named camera (or configured free camera)."""
        if camera_name:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if cam_id < 0:
                raise ValueError(f"Unknown camera name: {camera_name}")
            fovy = float(self.model.cam_fovy[cam_id])
        else:
            # Use MuJoCo global visual FOV for free-camera baseline intrinsics
            fovy = float(self.model.vis.global_.fovy)

        fovy_rad = np.deg2rad(fovy)
        fy = 0.5 * height / np.tan(fovy_rad / 2.0)
        fx = fy
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0

        return {
            "width": width,
            "height": height,
            "fov_y_deg": fovy,
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
        }

    def get_camera_extrinsics(self, camera_name: str) -> Dict[str, Any]:
        """Return camera extrinsics for a named camera."""
        cam_id = self._resolve_camera_id(camera_name)
        cam_pos = self.data.cam_xpos[cam_id].copy()
        rot_world_from_mujoco_cam = self.data.cam_xmat[cam_id].reshape(3, 3).copy()
        rot_mujoco_cam_to_opencv = self._camera_mujoco_to_opencv_rotation()
        rot_world_from_opencv_cam = rot_world_from_mujoco_cam @ rot_mujoco_cam_to_opencv

        t_world_from_opencv = self._build_homogeneous(rot_world_from_opencv_cam, cam_pos)
        t_opencv_from_world = np.linalg.inv(t_world_from_opencv)

        return {
            "camera_name": camera_name,
            "frame_convention": RobotConfig.CAMERA_FRAME_CONVENTION,
            "world_from_camera": t_world_from_opencv.tolist(),
            "camera_from_world": t_opencv_from_world.tolist(),
            "position_world": cam_pos.astype(float).tolist(),
            "rotation_world_from_camera": rot_world_from_opencv_cam.astype(float).tolist(),
        }

    def capture_camera_frame(
        self,
        width: int = RobotConfig.CAMERA_DEFAULT_WIDTH,
        height: int = RobotConfig.CAMERA_DEFAULT_HEIGHT,
        camera_name: Optional[str] = None,
        include_depth: bool = True,
    ) -> Dict[str, np.ndarray]:
        """Capture RGB (and optional depth) frame from a named or free camera."""
        camera = camera_name if camera_name else self._build_free_camera()

        with self._render_lock:
            renderer = self._get_renderer(width, height)
            renderer.update_scene(self.data, camera=camera)
            rgb = renderer.render().copy()

            depth = None
            if include_depth:
                renderer.enable_depth_rendering()
                renderer.update_scene(self.data, camera=camera)
                depth = renderer.render().copy()
                renderer.disable_depth_rendering()

        return {"rgb": rgb, "depth": depth}

    def get_camera_point_cloud(
        self,
        camera_name: str,
        width: int = RobotConfig.CAMERA_DEFAULT_WIDTH,
        height: int = RobotConfig.CAMERA_DEFAULT_HEIGHT,
        max_depth: float = RobotConfig.POINT_CLOUD_DEFAULT_MAX_DEPTH,
        stride: int = RobotConfig.POINT_CLOUD_DEFAULT_STRIDE,
        frame: str = RobotConfig.POINT_CLOUD_DEFAULT_FRAME,
    ) -> Dict[str, Any]:
        """Generate RGBD point cloud from a named camera."""
        if stride <= 0:
            raise ValueError("stride must be a positive integer")
        if frame not in ("world", "camera"):
            raise ValueError("frame must be 'world' or 'camera'")

        frame_data = self.capture_camera_frame(
            width=width,
            height=height,
            camera_name=camera_name,
            include_depth=True,
        )
        depth = frame_data["depth"]
        rgb = frame_data["rgb"]
        if depth is None:
            raise ValueError("Depth rendering is not available for point cloud generation")

        intrinsics = self.get_camera_intrinsics(width=width, height=height, camera_name=camera_name)
        fx = intrinsics["fx"]
        fy = intrinsics["fy"]
        cx = intrinsics["cx"]
        cy = intrinsics["cy"]

        h, w = depth.shape
        xs = np.arange(0, w, stride, dtype=np.int32)
        ys = np.arange(0, h, stride, dtype=np.int32)
        uu, vv = np.meshgrid(xs, ys)

        sampled_depth = depth[vv, uu]
        valid_mask = np.isfinite(sampled_depth) & (sampled_depth > RobotConfig.DEPTH_EPSILON)
        if max_depth is not None and max_depth > 0:
            valid_mask &= sampled_depth <= max_depth

        if not np.any(valid_mask):
            return {
                "camera_name": camera_name,
                "frame": frame,
                "points": [],
                "colors": [],
                "num_points": 0,
                "intrinsics": intrinsics,
                "extrinsics": self.get_camera_extrinsics(camera_name),
            }

        z = sampled_depth[valid_mask].astype(np.float32)
        u = uu[valid_mask].astype(np.float32)
        v = vv[valid_mask].astype(np.float32)

        x = ((u - cx) / fx) * z
        y = ((v - cy) / fy) * z
        points_camera = np.stack([x, y, z], axis=1).astype(np.float32)

        sampled_rgb = rgb[vv, uu]
        colors = sampled_rgb[valid_mask].astype(np.float32) / 255.0

        extrinsics = self.get_camera_extrinsics(camera_name)
        if frame == "camera":
            points_out = points_camera
        else:
            world_from_camera = np.array(extrinsics["world_from_camera"], dtype=np.float64)
            rot = world_from_camera[:3, :3]
            trans = world_from_camera[:3, 3]
            points_out = (rot @ points_camera.T).T + trans

        return {
            "camera_name": camera_name,
            "frame": frame,
            "points": points_out.astype(float).tolist(),
            "colors": colors.astype(float).tolist(),
            "num_points": int(points_out.shape[0]),
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
        }

    # ============================================================
    # Mobile Base Control Methods
    # ============================================================

    def get_mobile_position(self) -> np.ndarray:
        """Get current mobile base pose [x, y, theta] in world frame."""
        base_pos = self.data.site_xpos[self.mobile_base_center_id]
        base_rot = self.data.site_xmat[self.mobile_base_center_id].reshape(3, 3)
        base_theta = np.arctan2(base_rot[1, 0], base_rot[0, 0])
        return np.array([base_pos[0], base_pos[1], base_theta])

    def get_mobile_target_position(self) -> np.ndarray:
        """Get current mobile base target pose [x, y, theta] in world frame."""
        return self._mobile_target_position

    def set_mobile_target_position(self, mobile_target_position: np.ndarray) -> None:
        """Set mobile base target pose [x, y, theta] in world frame."""
        self._mobile_target_position = np.array(mobile_target_position)
        self._mobile_error_integral[:] = 0

    def get_mobile_position_diff(self) -> np.ndarray:
        """Get mobile base position error [delta_x, delta_y, delta_theta] between target and current position."""
        diff = self._mobile_target_position - self.get_mobile_position()
        diff[2] = np.arctan2(np.sin(diff[2]), np.cos(diff[2]))
        return diff

    def get_mobile_velocity(self) -> np.ndarray:
        """Get current mobile base velocity [vx, vy, omega] from joint velocities."""
        return np.array([
            self.data.qvel[self.mobile_joint_ids[0]],
            self.data.qvel[self.mobile_joint_ids[1]],
            self.data.qvel[self.mobile_joint_ids[2]]
        ])

    def _compute_mobile_control(self) -> np.ndarray:
        """Compute PD control commands [vx, vy, omega] for mobile base to reach target."""
        current_pos = self.get_mobile_position()
        current_vel = self.get_mobile_velocity()

        pos_error = self.get_mobile_position_diff()
        self._mobile_error_integral += pos_error * self.dt
        self._mobile_error_integral = np.clip(
            self._mobile_error_integral,
            -RobotConfig.MOBILE_I_LIMIT,
            RobotConfig.MOBILE_I_LIMIT
        )

        p_term = RobotConfig.MOBILE_KP * pos_error
        i_term = RobotConfig.MOBILE_KI * self._mobile_error_integral
        d_term = RobotConfig.MOBILE_KD * current_vel

        pid_cmd = p_term + i_term - d_term
        return pid_cmd
    
    # ============================================================
    # Mobile Planning Methods
    # ============================================================

    def plan_mobile_path(self, target_pos: np.ndarray, simplify: bool = True) -> Optional[List[np.ndarray]]:
        """Plan path for mobile base to reach target position using A* algorithm."""
        # Ensure target_pos is array-like with 2 elements
        target_pos = np.array(target_pos[:2]) if len(target_pos) > 2 else np.array(target_pos)

        grid_size = RobotConfig.GRID_SIZE

        # Inflate obstacles by robot radius for collision-free planning
        inflated_map = PathPlanner.inflate_obstacles(
            self.grid_map,
            RobotConfig.MOBILE_BASE_RADIUS,
            grid_size
        )

        # Get current position
        current_joint = self.get_mobile_position()

        # Convert to grid coordinates
        start_grid = self._world_to_grid(current_joint[:2], grid_size)
        goal_grid = self._world_to_grid(target_pos[:2], grid_size)

        # Find the closest free cell to target in the INFLATED map
        # This ensures sufficient clearance at the final goal position
        if inflated_map[goal_grid[0], goal_grid[1]] == 1:
            # Target is in obstacle (safety inflated), find nearest free cell along axis
            adjusted_goal = PathPlanner.find_nearest_axial_free_cell(goal_grid, inflated_map)
            if adjusted_goal is None:
                return None
        else:
            # Target is already in safe free space
            adjusted_goal = goal_grid

        # Run A* search on inflated map to the adjusted goal
        # This ensures safe path planning while reaching as close as possible
        path_grid, closest_point = PathPlanner.astar_search(start_grid, adjusted_goal, inflated_map)
        
        if path_grid is None:
            return None

        # Simplify path if requested
        if simplify and len(path_grid) > 2:
            # First pass: Line-of-sight simplification
            path_grid = PathPlanner.simplify_path_line_of_sight(path_grid, inflated_map)
            
            # Second pass: Angle-based filtering
            path_grid = PathPlanner.simplify_path_angle_filter(path_grid)
            
            # Third pass: B-spline smoothing
            path_grid = PathPlanner.smooth_path_bspline(path_grid)
        
        # Convert grid path to world coordinates
        path_world = []
        for i, grid_pos in enumerate(path_grid):
            world_xy = self._grid_to_world(grid_pos, grid_size)
            
            # Calculate orientation (theta)
            if i < len(path_grid) - 1:
                # Point towards next waypoint
                next_xy = self._grid_to_world(path_grid[i + 1], grid_size)
                theta = np.arctan2(next_xy[1] - world_xy[1], next_xy[0] - world_xy[0])
            elif i > 0:
                # Last waypoint: use direction from previous waypoint (natural arrival)
                prev_xy = self._grid_to_world(path_grid[i - 1], grid_size)
                theta = np.arctan2(world_xy[1] - prev_xy[1], world_xy[0] - prev_xy[0])
            else:
                # Single waypoint: point towards original target
                theta = np.arctan2(target_pos[1] - world_xy[1], target_pos[0] - world_xy[0])
            
            path_world.append(np.array([world_xy[0], world_xy[1], theta]))

        # Add final rotation waypoint to face the original target
        if len(path_world) > 0:
            last_pos = path_world[-1][:2]  # [x, y] of last waypoint
            target_theta = np.arctan2(target_pos[1] - last_pos[1], target_pos[0] - last_pos[0])

            # Add rotation waypoint (same position, different orientation)
            path_world.append(np.array([last_pos[0], last_pos[1], target_theta]))

        return path_world
    
    def follow_mobile_path(self, path_world: List[np.ndarray], timeout_per_waypoint: float = 30.0, verbose: bool = False) -> bool:
        """Follow a path by sequentially moving to each waypoint."""
        if verbose:
            print(f"Following path with {len(path_world)} waypoints")
        
        for i, waypoint in enumerate(path_world):
            if verbose:
                print(f"Moving to waypoint {i+1}/{len(path_world)}: [{waypoint[0]:.2f}, {waypoint[1]:.2f}, {waypoint[2]:.2f}]")
            
            # Check if this is the last waypoint
            is_last_waypoint = (i == len(path_world) - 1)

            # Get current position
            curr_pos = self.get_mobile_position()

            # Calculate angle difference
            angle_diff = waypoint[2] - curr_pos[2]
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

            if abs(angle_diff) > np.deg2rad(45):
                rotate_target = curr_pos.copy()
                rotate_target[2] = waypoint[2]
                self.set_mobile_target_position(rotate_target)

                # Wait for rotation to complete
                start_time = time.time()
                while time.time() - start_time < 5.0:
                    if np.abs(self.get_mobile_position_diff()[2]) < np.deg2rad(5):
                        break
                    time.sleep(0.02)

            self.set_mobile_target_position(waypoint)
                      
            # Wait for convergence
            start_time = time.time()
            converged = False
            
            while time.time() - start_time < timeout_per_waypoint:
                # Check position and velocity convergence
                pos_diff = self.get_mobile_position_diff()
                pos_diff[-1] /= 2  # Theta weighted at 50%
                pos_error = np.linalg.norm(pos_diff)
                vel_error = np.linalg.norm(self.get_mobile_velocity())
                
                if is_last_waypoint:
                    # Last waypoint: Strict stop required
                    if pos_error < 0.05 and vel_error < 0.05:
                        converged = True
                        if verbose:
                            print(f"  Reached destination in {time.time() - start_time:.2f}s")
                        break
                else:
                    # Intermediate waypoints: Pass through without stopping (no velocity check)
                    if pos_error < 0.15:
                        converged = True
                        break
                
                time.sleep(0.02)
            
            if not converged:
                if verbose:
                    print(f"  Timeout at waypoint {i+1} (pos_error={pos_error:.4f}, vel_error={vel_error:.4f})")
                return False
        
        if verbose:
            print("Path following completed successfully")
        return True

    # ============================================================
    # Arm Joint Control Methods
    # ============================================================

    def get_arm_target_joint(self) -> np.ndarray:
        """Get current arm target joint positions [j1~j7] in radians."""
        return self._arm_target_joint

    def set_arm_target_joint(self, arm_target_joint: np.ndarray) -> None:
        """Set arm target joint positions [j1~j7] in radians."""
        self._arm_target_joint = np.array(arm_target_joint)
        self._arm_error_integral[:] = 0

    def get_arm_joint_position(self) -> np.ndarray:
        """Get current arm joint positions [j1~j7] from joint states."""
        return np.array([self.data.qpos[jid] for jid in self.arm_joint_ids])

    def get_arm_joint_diff(self) -> np.ndarray:
        """Get arm position error [delta_j1~delta_j7] between target and current position."""
        return self._arm_target_joint - self.get_arm_joint_position()

    def get_arm_joint_velocity(self) -> np.ndarray:
        """Get current arm joint velocities [v1~v7] from joint velocities."""
        return np.array([self.data.qvel[jid] for jid in self.arm_joint_ids])

    def _compute_arm_control(self) -> np.ndarray:
        """Compute PID position control commands [j1~j7] for arm to reach target."""
        current_pos = self.get_arm_joint_position()
        current_vel = self.get_arm_joint_velocity()

        pos_error = self._arm_target_joint - current_pos
        
        # Update integral term with anti-windup
        self._arm_error_integral += pos_error * self.dt
        self._arm_error_integral = np.clip(
            self._arm_error_integral,
            -RobotConfig.ARM_I_LIMIT,
            RobotConfig.ARM_I_LIMIT
        )

        p_term = RobotConfig.ARM_KP * pos_error
        i_term = RobotConfig.ARM_KI * self._arm_error_integral
        d_term = RobotConfig.ARM_KD * current_vel

        return current_pos + p_term + i_term - d_term

    # ============================================================
    # End Effector Control Methods
    # ============================================================
    
    @staticmethod
    def _rotation_matrix_to_euler_xyz(rot: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to XYZ Euler angles [roll, pitch, yaw]."""
        return R.from_matrix(rot.reshape(3, 3)).as_euler("xyz")

    def get_ee_position(self, data: Optional[mujoco.MjData] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return current end effector position and orientation in world frame."""
        if data is None:
            data = self.data

        ee_pos = data.site_xpos[self.ee_site_id].copy()
        ee_rot = data.site_xmat[self.ee_site_id]
        ee_ori = self._rotation_matrix_to_euler_xyz(ee_rot)
        return ee_pos, ee_ori

    def _compute_ee_jacobian(self, data: Optional[mujoco.MjData] = None) -> np.ndarray:
        """Compute 6x7 Jacobian for the end effector site (arm joints only)."""
        if data is None:
            data = self.data

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, data, jacp, jacr, self.ee_site_id)

        jacp_arm = jacp[:, self.arm_dof_indices]
        jacr_arm = jacr[:, self.arm_dof_indices]
        return np.vstack([jacp_arm, jacr_arm])

    def _solve_ik_position(self, target_pos: np.ndarray, max_iterations: Optional[int] = None) -> Tuple[bool, np.ndarray]:
        """Solve IK for a target position (orientation is kept constant)."""
        if max_iterations is None:
            max_iterations = RobotConfig.IK_MAX_ITERATIONS

        q = self.get_arm_joint_position().copy()

        ik_data = mujoco.MjData(self.model)
        ik_data.qpos[:] = self.data.qpos[:]

        for _ in range(max_iterations):
            for i, joint_id in enumerate(self.arm_joint_ids):
                ik_data.qpos[joint_id] = q[i]
            mujoco.mj_forward(self.model, ik_data)

            current_pos = ik_data.site_xpos[self.ee_site_id].copy()
            pos_error = target_pos - current_pos

            if np.linalg.norm(pos_error) < RobotConfig.IK_POSITION_TOLERANCE:
                return True, q

            jacobian = self._compute_ee_jacobian(ik_data)[:3, :]
            jjt = jacobian @ jacobian.T
            damping = (RobotConfig.IK_DAMPING ** 2) * np.eye(jacobian.shape[0])
            inv_term = np.linalg.inv(jjt + damping)
            dq = jacobian.T @ (inv_term @ pos_error)
            q += RobotConfig.IK_STEP_SIZE * dq
            q = np.clip(q, RobotConfig.ARM_JOINT_LIMITS[:, 0], RobotConfig.ARM_JOINT_LIMITS[:, 1])

        return False, q
    
    def set_ee_target_position(self, target_pos: np.ndarray) -> Tuple[bool, np.ndarray]:
        """Set end effector target position in world frame."""
        success, joint_angles = self._solve_ik_position(target_pos)
        if success:
            self.set_arm_target_joint(joint_angles)
        return success, joint_angles

    # ============================================================
    # Gripper Control Methods
    # ============================================================

    def get_gripper_width(self) -> float:
        """Get current gripper width in meters."""
        return 2.0 * self.data.ctrl[self.gripper_actuator_ids[0]]
    
    def set_target_gripper_width(self, width: float) -> None:
        """Set target gripper width in meters (0.0 = closed, 0.08 = fully open)."""
        self._gripper_target_width = np.clip(width, 0.0, 0.08)
    
    def get_gripper_width_diff(self) -> float:
        """Get gripper width error between target and current position."""
        return self._gripper_target_width - self.get_gripper_width()
    
    def get_gripper_width_velocity(self) -> float:
        """Get gripper width velocity."""
        return self.data.ctrl[self.gripper_actuator_ids[0]]
    
    def _compute_gripper_control(self) -> np.ndarray:
        """Compute gripper control commands."""
        # Target width is symmetric: finger1 = +width/2, finger2 = -width/2
        target_finger1 = self._gripper_target_width / 2.0
        target_finger2 = -self._gripper_target_width / 2.0
        
        return np.array([target_finger1, target_finger2])
    
    # ============================================================
    # Pick & Place Methods
    # ============================================================

    def _wait_for_arm_convergence(self, timeout: float = 10.0) -> bool:
        """Wait for arm to converge to target position."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            pos_error = np.linalg.norm(self.get_arm_joint_diff())
            vel_error = np.linalg.norm(self.get_arm_joint_velocity())
            if pos_error < 0.1 and vel_error < 0.1:
                return True
            time.sleep(0.02)
        return False

    def pick_object(
        self, 
        object_pos: np.ndarray, 
        approach_height: float = 0.1, 
        lift_height: float = 0.2,
        return_to_home: bool = True,
        timeout: float = 10.0,
        verbose: bool = False
    ) -> bool:
        """Pick up an object at the specified position."""
        if verbose:
            print(f"Starting pick sequence at position [{object_pos[0]:.3f}, {object_pos[1]:.3f}, {object_pos[2]:.3f}]")
        
        # Step 1: Open gripper
        if verbose:
            print("  Step 1: Opening gripper...")
        self.set_target_gripper_width(0.08)
        time.sleep(1.0)
        
        # Step 2: Move to approach position (above object)
        approach_pos = np.array([object_pos[0], object_pos[1], object_pos[2] + approach_height])
        if verbose:
            print(f"  Step 2: Moving to approach position (height: {approach_height:.3f}m above object)...")
        success, _ = self.set_ee_target_position(approach_pos)
        if not success:
            if verbose:
                print("  Failed to reach approach position")
            return False
        
        if not self._wait_for_arm_convergence(timeout):
            if verbose:
                print("  Timeout waiting for approach position")
            return False
        
        # Step 3: Lower to grasp position
        grasp_pos = np.array([object_pos[0], object_pos[1], object_pos[2]])
        if verbose:
            print(f"  Step 3: Lowering to grasp position...")
        success, _ = self.set_ee_target_position(grasp_pos)
        if not success:
            if verbose:
                print("  Failed to reach grasp position")
            return False
        
        if not self._wait_for_arm_convergence(timeout):
            if verbose:
                print("  Timeout waiting for grasp position")
            return False
        
        # Step 4: Close gripper to grasp
        if verbose:
            print("  Step 4: Closing gripper to grasp...")
        self.set_target_gripper_width(0.02)
        time.sleep(1.5)  # Wait for gripper to close and stabilize
        
        # Step 5: Lift object
        lift_pos = np.array([object_pos[0], object_pos[1], object_pos[2] + lift_height])
        if verbose:
            print(f"  Step 5: Lifting object (height: {lift_height:.3f}m above original position)...")
        success, _ = self.set_ee_target_position(lift_pos)
        if not success:
            if verbose:
                print("  Failed to lift object")
            return False
        
        if not self._wait_for_arm_convergence(timeout):
            if verbose:
                print("  Timeout waiting for lift position")
            return False
        
        # Step 6: Return to home position (optional)
        if return_to_home:
            if verbose:
                print("  Step 6: Returning arm to home position...")
            self.set_arm_target_joint(RobotConfig.ARM_INIT_POSITION)
            
            if not self._wait_for_arm_convergence(timeout):
                if verbose:
                    print("  Timeout waiting for home position")
                return False
        
        if verbose:
            print("  Pick sequence completed successfully!")
        return True
    
    def place_object(
        self,
        place_pos: np.ndarray,
        approach_height: float = 0.2,
        retract_height: float = 0.3,
        return_to_home: bool = True,
        timeout: float = 10.0,
        verbose: bool = False
    ) -> bool:
        """Place an object at the specified position."""
        if verbose:
            print(f"Starting place sequence at position [{place_pos[0]:.3f}, {place_pos[1]:.3f}, {place_pos[2]:.3f}]")
        
        # Step 1: Move to approach position (above placement location)
        approach_pos = np.array([place_pos[0], place_pos[1], place_pos[2] + approach_height])
        if verbose:
            print(f"  Step 1: Moving to approach position (height: {approach_height:.3f}m above target)...")
        success, _ = self.set_ee_target_position(approach_pos)
        if not success:
            if verbose:
                print("  Failed to reach approach position")
            return False
        
        if not self._wait_for_arm_convergence(timeout):
            if verbose:
                print("  Timeout waiting for approach position")
            return False
        
        # Step 2: Open gripper to release
        if verbose:
            print("  Step 2: Opening gripper to release object...")
        self.set_target_gripper_width(0.08)
        time.sleep(1.5)  # Wait for gripper to open and object to settle
        
        # Step 3: Retract upward
        retract_pos = np.array([place_pos[0], place_pos[1], place_pos[2] + retract_height])
        if verbose:
            print(f"  Step 3: Retracting (height: {retract_height:.3f}m above placement)...")
        success, _ = self.set_ee_target_position(retract_pos)
        if not success:
            if verbose:
                print("  Failed to retract")
            return False
        
        if not self._wait_for_arm_convergence(timeout):
            if verbose:
                print("  Timeout waiting for retract position")
            return False
        
        # Step 4: Return to home position (optional)
        if return_to_home:
            if verbose:
                print("  Step 4: Returning arm to home position...")
            self.set_arm_target_joint(RobotConfig.ARM_INIT_POSITION)
            
            if not self._wait_for_arm_convergence(timeout):
                if verbose:
                    print("  Timeout waiting for home position")
                return False
        
        if verbose:
            print("  Place sequence completed successfully!")
        return True
    
    # ============================================================
    # Object Interaction Methods
    # ============================================================

    def get_object_positions(self) -> dict:
        """Get list of object dictionaries with id, name, position and orientation in world frame."""
        objects = {}
        for i in self.object_ids:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name.startswith("object_"):
                objects[name] = {
                    'id': i,
                    'pos': self.data.xpos[i], 
                    'ori': self._rotation_matrix_to_euler_xyz(self.data.xmat[i])
                }
        return objects

    # ============================================================
    # Grid Map Methods
    # ============================================================

    def get_grid_map(self) -> np.ndarray:
        """Get grid map of the environment.

        Returns:
            np.ndarray: Binary occupancy grid (0=free, 1=occupied)
        """
        return self.grid_map

    def _get_floor_info(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get cached floor geometry information.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (floor_size, floor_pos) where floor_pos may change
        """
        if self._floor_geom_id is None:
            self._floor_geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor_room_g0"
            )
            self._floor_size = self.model.geom_size[self._floor_geom_id]
        # Floor position may change, so always get from data
        self._floor_pos = self.data.geom_xpos[self._floor_geom_id]
        return self._floor_size, self._floor_pos

    def _world_to_grid(self, world_pos: Tuple[float, float], grid_size: float = RobotConfig.GRID_SIZE) -> Tuple[int, int]:
        """Convert world position [x, y] to grid indices [i, j].

        Args:
            world_pos: World position (x, y) in meters
            grid_size: Grid cell size in meters (default: RobotConfig.GRID_SIZE)

        Returns:
            Tuple[int, int]: Grid indices (i, j)
        """
        _, floor_pos = self._get_floor_info()
        return GridMapUtils.world_to_grid(world_pos, floor_pos, self.grid_map.shape, grid_size)

    def _grid_to_world(self, grid_pos: Tuple[int, int], grid_size: float = RobotConfig.GRID_SIZE) -> np.ndarray:
        """Convert grid indices [i, j] to world position [x, y].

        Args:
            grid_pos: Grid indices (i, j)
            grid_size: Grid cell size in meters (default: RobotConfig.GRID_SIZE)

        Returns:
            np.ndarray: World position [x, y] in meters
        """
        _, floor_pos = self._get_floor_info()
        return GridMapUtils.grid_to_world(grid_pos, floor_pos, self.grid_map.shape, grid_size)

    # ============================================================
    # Simulation Loop
    # ============================================================

    def run(self) -> None:
        """Run simulation with 3D viewer and PD control loop (blocking)."""
        def key_callback(keycode: int) -> None:
            if keycode == RobotConfig.VIEWER_KEY_TOGGLE_WASD_DEBUG:
                self._queue_viewer_camera_command("toggle_wasd_debug")
            elif keycode == RobotConfig.VIEWER_KEY_TOGGLE:
                self._queue_viewer_camera_command("toggle")
            elif keycode == RobotConfig.VIEWER_KEY_THIRD_PERSON:
                self._queue_viewer_camera_command("third_person")
            elif keycode == RobotConfig.VIEWER_KEY_ROBOT_EYE:
                self._queue_viewer_camera_command("robot_eye")
            elif keycode == RobotConfig.VIEWER_KEY_ROBOT_EYE_DEBUG:
                self._queue_viewer_camera_command("robot_eye_debug")
            elif keycode == RobotConfig.VIEWER_KEY_TOGGLE_CONTROL_DEBUG:
                self._queue_viewer_camera_command("toggle_control_debug")
            elif self._viewer_wasd_debug_enabled:
                if keycode == glfw.KEY_W:
                    self._queue_viewer_camera_command("debug_cam_tilt_up")
                elif keycode == glfw.KEY_S:
                    self._queue_viewer_camera_command("debug_cam_tilt_down")
                elif keycode == glfw.KEY_A:
                    self._queue_viewer_camera_command("debug_cam_pan_left")
                elif keycode == glfw.KEY_D:
                    self._queue_viewer_camera_command("debug_cam_pan_right")

        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_callback) as v:
            # Camera setup (default: third-person)
            with v.lock():
                self._set_viewer_camera_mode(v, "third_person")
                self._set_viewer_overlay(v)

                # Hide debug visuals
                v.opt.geomgroup[0] = 0
                v.opt.sitegroup[0] = v.opt.sitegroup[1] = v.opt.sitegroup[2] = 0
                v.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
                v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
                v.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
                v.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
                v.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = False
                v.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
                v.opt.label = mujoco.mjtLabel.mjLABEL_NONE

            print(
                "[VIEWER] Camera hotkeys: "
                "'F7' WASD mode, 'F8' toggle, 'F9' third_person, 'F10' robot_eye, "
                "'F11' robot_eye_debug, 'F12' control debug on/off."
            )

            # Main loop
            while v.is_running():
                viewer_command = self._pop_viewer_camera_command()
                if viewer_command is not None:
                    with v.lock():
                        self._apply_viewer_camera_command(v, viewer_command)
                        self._set_viewer_overlay(v)
                        self._viewer_overlay_last_update_time = time.time()

                # mobile base control
                mobile_control = self._compute_mobile_control()
                self.data.ctrl[self.mobile_actuator_ids[0]] = mobile_control[0]
                self.data.ctrl[self.mobile_actuator_ids[1]] = mobile_control[1]
                self.data.ctrl[self.mobile_actuator_ids[2]] = mobile_control[2]

                # arm control
                arm_control = self._compute_arm_control()
                for i, actuator_id in enumerate(self.arm_actuator_ids):
                    self.data.ctrl[actuator_id] = arm_control[i]

                # gripper control
                gripper_control = self._compute_gripper_control()
                for i, actuator_id in enumerate(self.gripper_actuator_ids):
                    self.data.ctrl[actuator_id] = gripper_control[i]

                # debug camera rig control
                debug_cam_control = self._compute_debug_camera_control()
                for i, actuator_id in enumerate(self.debug_camera_actuator_ids):
                    self.data.ctrl[actuator_id] = debug_cam_control[i]

                mujoco.mj_step(self.model, self.data)

                if self._viewer_camera_mode == "robot_eye_debug":
                    with v.lock():
                        self._update_robot_eye_debug_camera(v)

                now = time.time()
                if now - self._viewer_overlay_last_update_time >= self._viewer_overlay_update_period:
                    with v.lock():
                        self._set_viewer_overlay(v)
                    self._viewer_overlay_last_update_time = now

                v.sync()

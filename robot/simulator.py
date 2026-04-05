"""MuJoCo robot simulator with automatic position control for Panda-Omron mobile manipulator."""

import time
import threading
from collections import deque
import numpy as np
import glfw
import mujoco, mujoco.viewer
from scipy.spatial.transform import Rotation as R
from typing import Optional, List, Tuple, Dict, Any
from simulator_util import PathPlanner, GridMapUtils

try:
    import tkinter as tk
    from tkinter import ttk
except Exception:  # pragma: no cover - optional runtime dependency
    tk = None
    ttk = None


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
    ARM_HOLD_POSITION_EPS = 8e-4
    ARM_DEBUG_STABILIZE_POS_EPS = 1.2e-3
    ARM_DEBUG_STABILIZE_VEL_EPS = 0.03

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
    DEBUG_CAMERA_YAW_JOINT_NAME = "robot0_debug_camera_yaw_joint"
    DEBUG_CAMERA_PITCH_JOINT_NAME = "robot0_debug_camera_pitch_joint"
    DEBUG_CAMERA_ROLL_JOINT_NAME = "robot0_debug_camera_roll_joint"
    DEBUG_CAMERA_YAW_ACTUATOR_NAME = "robot0_debug_camera_yaw_actuator"
    DEBUG_CAMERA_PITCH_ACTUATOR_NAME = "robot0_debug_camera_pitch_actuator"
    DEBUG_CAMERA_ROLL_ACTUATOR_NAME = "robot0_debug_camera_roll_actuator"
    DEBUG_CAMERA_YAW_BODY_NAME = "robot0_debug_camera_yaw_link"
    DEBUG_CAMERA_PITCH_BODY_NAME = "robot0_debug_camera_pitch_link"
    DEBUG_CAMERA_ROLL_BODY_NAME = "robot0_debug_camera_roll_link"
    DEBUG_CAMERA_PITCH_OFFSET_LOCAL = np.array([0.04, 0.0, 0.0])
    DEBUG_CAMERA_DEFAULT_TARGET_JOINT = np.array([-0.55, -0.12, 0.0])
    DEBUG_CAMERA_DEFAULT_LOOK_AT_WORLD = np.array([0.20, -0.15, 0.85])
    DEBUG_CAMERA_HOME_YAW_CORRECTION_RAD = np.pi
    DEBUG_CAMERA_DEFAULT_FOVY_DEG = 65.0
    DEBUG_CAMERA_MIN_FOVY_DEG = 30.0
    DEBUG_CAMERA_MAX_FOVY_DEG = 95.0
    DEBUG_CAMERA_ZOOM_STEP_DEG = 2.0
    DEBUG_CAMERA_ROLL_STEP_DEG = 3.0
    DEBUG_CAMERA_MANUAL_STEP_DEG = 3.0
    DEBUG_CAMERA_CONVERGENCE_POS_THRESHOLD = 0.01
    DEBUG_CAMERA_CONVERGENCE_VEL_THRESHOLD = 0.05
    DEBUG_CAMERA_ENABLE_KINEMATIC_STABILIZATION = True
    DEBUG_CAMERA_STABILIZATION_RECOMPUTE_FORWARD = True
    DEPTH_EPSILON = 1e-6
    POINT_CLOUD_DEFAULT_STRIDE = 4
    POINT_CLOUD_DEFAULT_MAX_DEPTH = 4.0
    POINT_CLOUD_DEFAULT_FRAME = "world"
    CAMERA_FRAME_CONVENTION = "opencv"
    VIEWER_MODE_THIRD_PERSON = "third_person"
    VIEWER_MODE_HAND_CAMERA_FIXED = "hand_camera_fixed"
    VIEWER_MODE_HAND_CAMERA_INSPECT = "hand_camera_inspect"
    VIEWER_MODE_ATTACHED_DEBUG_CAMERA_VIEW = "attached_debug_camera_view"
    VIEWER_MODE_ATTACHED_DEBUG_CAMERA_CONTROL = "attached_debug_camera_control"
    # Backward-compatible mode aliases used by external sandbox calls.
    VIEWER_MODE_ROBOT_EYE_FIXED = VIEWER_MODE_HAND_CAMERA_FIXED
    VIEWER_MODE_ROBOT_EYE_INSPECT = VIEWER_MODE_HAND_CAMERA_INSPECT
    VIEWER_MODE_DEBUG_CAMERA_MANUAL = VIEWER_MODE_ATTACHED_DEBUG_CAMERA_VIEW

    VIEWER_KEY_ATTACHED_DEBUG_CAMERA_VIEW_TOGGLE = glfw.KEY_F7
    VIEWER_KEY_TOGGLE = glfw.KEY_F8
    VIEWER_KEY_THIRD_PERSON = glfw.KEY_F9
    VIEWER_KEY_HAND_CAMERA_FIXED = glfw.KEY_F10
    VIEWER_KEY_HAND_CAMERA_INSPECT = glfw.KEY_F11
    VIEWER_KEY_TOGGLE_DEBUG_CAMERA_PANEL = glfw.KEY_F12
    VIEWER_KEY_TOGGLE_HELP = glfw.KEY_H
    VIEWER_KEY_DEBUG_CAMERA_PAN_LEFT = glfw.KEY_LEFT
    VIEWER_KEY_DEBUG_CAMERA_PAN_RIGHT = glfw.KEY_RIGHT
    VIEWER_KEY_DEBUG_CAMERA_TILT_UP = glfw.KEY_UP
    VIEWER_KEY_DEBUG_CAMERA_TILT_DOWN = glfw.KEY_DOWN
    VIEWER_KEY_DEBUG_CAMERA_ZOOM_IN = glfw.KEY_EQUAL
    VIEWER_KEY_DEBUG_CAMERA_ZOOM_OUT = glfw.KEY_MINUS
    VIEWER_KEY_DEBUG_CAMERA_ROLL_LEFT = glfw.KEY_COMMA
    VIEWER_KEY_DEBUG_CAMERA_ROLL_RIGHT = glfw.KEY_PERIOD
    VIEWER_KEY_DEBUG_CAMERA_RESET_ORIENTATION = glfw.KEY_R
    VIEWER_KEY_DEBUG_CAMERA_UPRIGHT_HOME = glfw.KEY_U
    VIEWER_KEY_DEBUG_CAMERA_RESET_ZOOM = glfw.KEY_0
    VIEWER_KEY_DEBUG_CAMERA_FLIP_DIRECTION = glfw.KEY_Y
    VIEWER_ROBOT_EYE_DEBUG_DISTANCE = 0.35
    VIEWER_FN_KEY_DEBOUNCE_SEC = 0.20
    VIEWER_ARROW_KEY_REPEAT_SEC = 0.03

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

        # Resolve attached debug camera 3-DOF rig IDs (yaw/pitch/roll)
        self.debug_camera_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, RobotConfig.DEBUG_CAMERA_YAW_JOINT_NAME),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, RobotConfig.DEBUG_CAMERA_PITCH_JOINT_NAME),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, RobotConfig.DEBUG_CAMERA_ROLL_JOINT_NAME),
        ]
        self.debug_camera_actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, RobotConfig.DEBUG_CAMERA_YAW_ACTUATOR_NAME),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, RobotConfig.DEBUG_CAMERA_PITCH_ACTUATOR_NAME),
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, RobotConfig.DEBUG_CAMERA_ROLL_ACTUATOR_NAME),
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
        self.debug_camera_yaw_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, RobotConfig.DEBUG_CAMERA_YAW_BODY_NAME
        )
        self.debug_camera_yaw_parent_body_id = int(self.model.body_parentid[self.debug_camera_yaw_body_id])
        self.debug_camera_pitch_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, RobotConfig.DEBUG_CAMERA_PITCH_BODY_NAME
        )
        self.debug_camera_roll_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, RobotConfig.DEBUG_CAMERA_ROLL_BODY_NAME
        )
        self.debug_camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, RobotConfig.DEBUG_CAMERA_NAME
        )
        for jid, name in zip(
            self.debug_camera_joint_ids,
            [
                RobotConfig.DEBUG_CAMERA_YAW_JOINT_NAME,
                RobotConfig.DEBUG_CAMERA_PITCH_JOINT_NAME,
                RobotConfig.DEBUG_CAMERA_ROLL_JOINT_NAME,
            ],
        ):
            self._require_valid_id(jid, name)
        for aid, name in zip(
            self.debug_camera_actuator_ids,
            [
                RobotConfig.DEBUG_CAMERA_YAW_ACTUATOR_NAME,
                RobotConfig.DEBUG_CAMERA_PITCH_ACTUATOR_NAME,
                RobotConfig.DEBUG_CAMERA_ROLL_ACTUATOR_NAME,
            ],
        ):
            self._require_valid_id(aid, name)
        self._require_valid_id(self.debug_camera_yaw_body_id, RobotConfig.DEBUG_CAMERA_YAW_BODY_NAME)
        self._require_valid_id(self.debug_camera_pitch_body_id, RobotConfig.DEBUG_CAMERA_PITCH_BODY_NAME)
        self._require_valid_id(self.debug_camera_roll_body_id, RobotConfig.DEBUG_CAMERA_ROLL_BODY_NAME)
        self._require_valid_id(self.debug_camera_id, RobotConfig.DEBUG_CAMERA_NAME)
        self._require_valid_id(self.debug_camera_yaw_parent_body_id, f"{RobotConfig.DEBUG_CAMERA_YAW_BODY_NAME}_parent")

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

        # Initialize attached debug camera orientation targets [yaw, pitch, roll]
        self._debug_camera_target_joint = np.clip(
            RobotConfig.DEBUG_CAMERA_DEFAULT_TARGET_JOINT.copy(),
            self.debug_camera_joint_limits[:, 0],
            self.debug_camera_joint_limits[:, 1],
        )
        for i, qpos_idx in enumerate(self.debug_camera_qpos_indices):
            self.data.qpos[qpos_idx] = self._debug_camera_target_joint[i]
            self.data.ctrl[self.debug_camera_actuator_ids[i]] = self._debug_camera_target_joint[i]
        self._debug_camera_home_look_target_world = RobotConfig.DEBUG_CAMERA_DEFAULT_LOOK_AT_WORLD.copy()
        self._debug_camera_home_fovy_deg = float(
            np.clip(
                RobotConfig.DEBUG_CAMERA_DEFAULT_FOVY_DEG,
                RobotConfig.DEBUG_CAMERA_MIN_FOVY_DEG,
                RobotConfig.DEBUG_CAMERA_MAX_FOVY_DEG,
            )
        )
        self._debug_camera_fovy_deg = self._debug_camera_home_fovy_deg
        self.model.cam_fovy[self.debug_camera_id] = self._debug_camera_fovy_deg
        
        # Initialize grid map
        self.grid_map = np.load("grid_map.npy")

        # Cache floor geometry information to avoid repeated queries
        self._floor_geom_id = None
        self._floor_size = None
        self._floor_pos = None

        # Compute forward kinematics
        mujoco.mj_forward(self.model, self.data)
        try:
            self.look_at_point(RobotConfig.DEBUG_CAMERA_DEFAULT_LOOK_AT_WORLD)
            corrected_home = self.get_debug_camera_target_joint()
            corrected_home[0] = self._wrap_to_pi(
                corrected_home[0] + float(RobotConfig.DEBUG_CAMERA_HOME_YAW_CORRECTION_RAD)
            )
            corrected_home[2] = 0.0
            self.set_debug_camera_target_joint(corrected_home)
            self._apply_debug_camera_stabilization(recompute_kinematics=True)
        except ValueError:
            # Keep the default joint target if look-at target cannot be resolved.
            pass
        self._debug_camera_home_target_joint = self._debug_camera_target_joint.copy()

        # Offscreen renderers for vision pipeline (cached by resolution)
        self._renderers: Dict[Tuple[int, int], mujoco.Renderer] = {}
        self._render_lock = threading.Lock()
        self._viewer_camera_mode = RobotConfig.VIEWER_MODE_THIRD_PERSON
        self._viewer_mode_before_attached_debug = RobotConfig.VIEWER_MODE_THIRD_PERSON
        self._eye_in_hand_camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, RobotConfig.EYE_IN_HAND_CAMERA_NAME
        )
        self._viewer_command_lock = threading.Lock()
        self._viewer_pending_camera_commands = deque()
        self._debug_panel_available = tk is not None and ttk is not None
        self._viewer_show_compact_status = not self._debug_panel_available
        self._viewer_show_extended_help = False
        self._viewer_overlay_update_period = 0.15
        self._viewer_overlay_last_update_time = 0.0
        self._viewer_last_key_time: Dict[int, float] = {}
        self._debug_camera_manual_step_deg = float(RobotConfig.DEBUG_CAMERA_MANUAL_STEP_DEG)
        self._debug_camera_step_left_deg = float(RobotConfig.DEBUG_CAMERA_MANUAL_STEP_DEG)
        self._debug_camera_step_right_deg = float(RobotConfig.DEBUG_CAMERA_MANUAL_STEP_DEG)
        self._debug_camera_step_up_deg = float(RobotConfig.DEBUG_CAMERA_MANUAL_STEP_DEG)
        self._debug_camera_step_down_deg = float(RobotConfig.DEBUG_CAMERA_MANUAL_STEP_DEG)
        self._debug_camera_roll_step_deg = float(RobotConfig.DEBUG_CAMERA_ROLL_STEP_DEG)
        self._debug_camera_zoom_step_deg = float(RobotConfig.DEBUG_CAMERA_ZOOM_STEP_DEG)
        self._debug_panel_window_visible = self._debug_panel_available
        self._debug_panel_stop_event = threading.Event()
        self._debug_panel_thread: Optional[threading.Thread] = None
        self._debug_live_preview_pending = False

    @staticmethod
    def _is_attached_debug_camera_mode(mode: str) -> bool:
        return mode in (
            RobotConfig.VIEWER_MODE_ATTACHED_DEBUG_CAMERA_VIEW,
            RobotConfig.VIEWER_MODE_ATTACHED_DEBUG_CAMERA_CONTROL,
        )

    def _normalize_non_debug_restore_mode(self, mode: str) -> str:
        """Normalize fallback mode used when leaving attached debug camera view."""
        if mode in (
            RobotConfig.VIEWER_MODE_THIRD_PERSON,
            RobotConfig.VIEWER_MODE_HAND_CAMERA_FIXED,
            RobotConfig.VIEWER_MODE_HAND_CAMERA_INSPECT,
        ):
            return mode
        return RobotConfig.VIEWER_MODE_THIRD_PERSON

    def _set_viewer_camera_mode(self, viewer: Any, mode: str) -> bool:
        """Apply viewer camera mode. Returns True if requested mode became active."""
        if mode == RobotConfig.VIEWER_MODE_HAND_CAMERA_FIXED:
            if self._eye_in_hand_camera_id < 0:
                print(
                    f"[VIEWER] '{RobotConfig.EYE_IN_HAND_CAMERA_NAME}' camera not found. "
                    "Staying in third-person view."
                )
                self._set_viewer_camera_mode(viewer, RobotConfig.VIEWER_MODE_THIRD_PERSON)
                return False
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = self._eye_in_hand_camera_id
            self._viewer_camera_mode = RobotConfig.VIEWER_MODE_HAND_CAMERA_FIXED
            return True

        if mode == RobotConfig.VIEWER_MODE_HAND_CAMERA_INSPECT:
            if self._eye_in_hand_camera_id < 0:
                print(
                    f"[VIEWER] '{RobotConfig.EYE_IN_HAND_CAMERA_NAME}' camera not found. "
                    "Staying in third-person view."
                )
                self._set_viewer_camera_mode(viewer, RobotConfig.VIEWER_MODE_THIRD_PERSON)
                return False
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.distance = RobotConfig.VIEWER_ROBOT_EYE_DEBUG_DISTANCE
            self._viewer_camera_mode = RobotConfig.VIEWER_MODE_HAND_CAMERA_INSPECT
            self._update_robot_eye_debug_camera(viewer, force_distance=True)
            return True

        if self._is_attached_debug_camera_mode(mode):
            if self.debug_camera_id < 0:
                print(
                    f"[VIEWER] '{RobotConfig.DEBUG_CAMERA_NAME}' camera not found. "
                    "Staying in third-person view."
                )
                self._set_viewer_camera_mode(viewer, RobotConfig.VIEWER_MODE_THIRD_PERSON)
                return False
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = self.debug_camera_id
            # Keep user workflow simple: rotate/zoom while staying in debug camera view.
            self._viewer_camera_mode = RobotConfig.VIEWER_MODE_ATTACHED_DEBUG_CAMERA_VIEW
            return True

        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = RobotConfig.CAM_LOOKAT
        viewer.cam.distance = RobotConfig.CAM_DISTANCE
        viewer.cam.azimuth = RobotConfig.CAM_AZIMUTH
        viewer.cam.elevation = RobotConfig.CAM_ELEVATION
        self._viewer_camera_mode = RobotConfig.VIEWER_MODE_THIRD_PERSON
        return True

    def _toggle_viewer_camera_mode(self, viewer: Any) -> None:
        """Quick toggle between third-person and fixed hand camera."""
        next_mode = (
            RobotConfig.VIEWER_MODE_HAND_CAMERA_FIXED
            if self._viewer_camera_mode == RobotConfig.VIEWER_MODE_THIRD_PERSON
            else RobotConfig.VIEWER_MODE_THIRD_PERSON
        )
        self._set_viewer_camera_mode(viewer, next_mode)

    def _viewer_mode_human_label(self) -> str:
        """Human-readable label for the current viewer mode."""
        labels = {
            RobotConfig.VIEWER_MODE_THIRD_PERSON: "third-person",
            RobotConfig.VIEWER_MODE_HAND_CAMERA_FIXED: "hand camera (fixed)",
            RobotConfig.VIEWER_MODE_HAND_CAMERA_INSPECT: "hand camera (free inspect)",
            RobotConfig.VIEWER_MODE_ATTACHED_DEBUG_CAMERA_VIEW: "attached debug camera view",
            RobotConfig.VIEWER_MODE_ATTACHED_DEBUG_CAMERA_CONTROL: "attached debug camera view",
        }
        return labels.get(self._viewer_camera_mode, self._viewer_camera_mode)

    def _build_compact_status_text(self) -> str:
        """Compact fallback text block when separate panel is unavailable."""
        attached_view_on = self._is_attached_debug_camera_mode(self._viewer_camera_mode)
        lines = [
            f"View mode: {self._viewer_mode_human_label()}",
            f"Attached debug camera view: {'ON' if attached_view_on else 'OFF'}",
            f"Live preview: {'ATTACHED DEBUG CAMERA' if attached_view_on else 'NOT ACTIVE'}",
        ]
        if attached_view_on:
            joint = self.get_debug_camera_joint_position()
            left_right_deg = np.rad2deg(joint[0])
            up_down_deg = -np.rad2deg(joint[1])
            roll_deg = np.rad2deg(joint[2])
            lines.append(f"Camera left/right: {left_right_deg:+.1f} deg")
            lines.append(f"Camera up/down: {up_down_deg:+.1f} deg")
            lines.append(f"Camera roll: {roll_deg:+.1f} deg")
            lines.append(f"Zoom (FOV): {self.get_debug_camera_zoom_fovy():.1f} deg")
        lines.append("F7 view  Arrows rotate  ,/. roll  +/- zoom  U upright")
        return "\n".join(lines)

    @staticmethod
    def _build_extended_help_text() -> str:
        """Extended help text (hidden by default)."""
        return (
            "Attached debug camera controls\n"
            "F7  : Enter/exit attached debug camera view\n"
            "F8  : Quick toggle third-person <-> hand camera (fixed)\n"
            "F9  : Third-person view\n"
            "F10 : Hand camera (fixed)\n"
            "F11 : Hand camera (free inspect)\n"
            "F12 : Show/hide debug camera control window\n"
            "H   : Show/hide this help\n"
            "\n"
            "Control window actions auto-enable attached debug camera live preview.\n"
            "Left/Right/Up/Down step sizes can be adjusted independently in the panel.\n"
            "\n"
            "While attached debug camera view is active\n"
            "Left/Right : Camera left/right\n"
            "Up/Down    : Camera up/down\n"
            ", / .      : Camera roll left/right\n"
            "+ / -      : Zoom in/out\n"
            "R          : Reset camera orientation\n"
            "0          : Reset zoom\n"
            "U          : Upright home view\n"
            "Y          : Flip direction 180 degrees\n"
            "\n"
            "Note: Attached debug camera controls do not change\n"
            "mobile base, arm, or gripper APIs."
        )

    def _set_viewer_overlay(self, viewer: Any) -> None:
        """Show optional fallback overlay and help."""
        texts: List[Tuple[Optional[int], Optional[int], Optional[str], Optional[str]]] = []

        # MuJoCo Python viewer does not expose native custom category insertion
        # for built-in right-side UI sections (Joint/Control/Equality). Keep
        # overlay minimal; the dedicated debug control window is the primary UI.
        if self._viewer_show_compact_status:
            texts.append(
                (
                    None,
                    mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                    "Attached Debug Camera",
                    self._build_compact_status_text(),
                )
            )

        if self._viewer_show_extended_help:
            texts.append(
                (
                    None,
                    mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
                    "Camera Help",
                    self._build_extended_help_text(),
                )
            )

        if texts:
            viewer.set_texts(texts)
        else:
            viewer.clear_texts()

    def _accept_viewer_key(self, keycode: int, cooldown_sec: float) -> bool:
        """Debounce viewer key events to prevent rapid double-toggles."""
        now = time.time()
        last_time = self._viewer_last_key_time.get(keycode, 0.0)
        if now - last_time < cooldown_sec:
            return False
        self._viewer_last_key_time[keycode] = now
        return True

    def set_debug_camera_manual_step_deg(self, step_deg: float) -> float:
        """Set manual camera movement step in degrees (backward-compatible, applies to all directions)."""
        step = float(step_deg)
        step = float(np.clip(step, 0.2, 180.0))
        self._debug_camera_manual_step_deg = step
        self._debug_camera_step_left_deg = step
        self._debug_camera_step_right_deg = step
        self._debug_camera_step_up_deg = step
        self._debug_camera_step_down_deg = step
        self._debug_camera_roll_step_deg = step
        return self._debug_camera_manual_step_deg

    def set_debug_camera_direction_steps_deg(
        self,
        *,
        left: Optional[float] = None,
        right: Optional[float] = None,
        up: Optional[float] = None,
        down: Optional[float] = None,
        roll: Optional[float] = None,
    ) -> Dict[str, float]:
        """Set independent angle step sizes for left/right/up/down/roll camera controls."""
        if left is not None:
            self._debug_camera_step_left_deg = float(np.clip(float(left), 0.2, 180.0))
        if right is not None:
            self._debug_camera_step_right_deg = float(np.clip(float(right), 0.2, 180.0))
        if up is not None:
            self._debug_camera_step_up_deg = float(np.clip(float(up), 0.2, 180.0))
        if down is not None:
            self._debug_camera_step_down_deg = float(np.clip(float(down), 0.2, 180.0))
        if roll is not None:
            self._debug_camera_roll_step_deg = float(np.clip(float(roll), 0.2, 180.0))
        self._debug_camera_manual_step_deg = float(
            np.mean(
                [
                    self._debug_camera_step_left_deg,
                    self._debug_camera_step_right_deg,
                    self._debug_camera_step_up_deg,
                    self._debug_camera_step_down_deg,
                ]
            )
        )
        return {
            "left": self._debug_camera_step_left_deg,
            "right": self._debug_camera_step_right_deg,
            "up": self._debug_camera_step_up_deg,
            "down": self._debug_camera_step_down_deg,
            "roll": self._debug_camera_roll_step_deg,
        }

    def set_debug_camera_zoom_step_deg(self, step_deg: float) -> float:
        """Set manual zoom step in degrees of FOV."""
        step = float(step_deg)
        step = float(np.clip(step, 0.2, 20.0))
        self._debug_camera_zoom_step_deg = step
        return self._debug_camera_zoom_step_deg

    def reset_debug_camera_orientation(self) -> np.ndarray:
        """Reset attached debug camera direction to configured home orientation."""
        self.set_debug_camera_target_joint(self._debug_camera_home_target_joint.copy())
        return self.get_debug_camera_target_joint()

    def look_at_debug_camera_home_target(self) -> np.ndarray:
        """Rotate attached debug camera to look at configured home world target."""
        self.look_at_point(self._debug_camera_home_look_target_world.copy())
        corrected = self.get_debug_camera_target_joint()
        corrected[0] = self._wrap_to_pi(
            corrected[0] + float(RobotConfig.DEBUG_CAMERA_HOME_YAW_CORRECTION_RAD)
        )
        corrected[2] = 0.0
        self.set_debug_camera_target_joint(corrected)
        return self.get_debug_camera_target_joint()

    def reset_debug_camera(self) -> Dict[str, Any]:
        """Reset attached debug camera orientation and zoom to home settings."""
        orientation = self.reset_debug_camera_orientation()
        fovy = self.reset_debug_camera_zoom()
        return {
            "orientation": orientation.astype(float).tolist(),
            "fov_y_deg": float(fovy),
        }

    def upright_reset_debug_camera(self) -> Dict[str, Any]:
        """Restore stable, upright home view for attached debug camera."""
        self.look_at_debug_camera_home_target()
        target = self.get_debug_camera_target_joint()
        target[2] = 0.0
        self.set_debug_camera_target_joint(target)
        fovy = self.reset_debug_camera_zoom()
        return {
            "orientation": self.get_debug_camera_target_joint().astype(float).tolist(),
            "fov_y_deg": float(fovy),
        }

    def flip_debug_camera_direction_180(self) -> np.ndarray:
        """Rotate attached debug camera horizontal direction by 180 degrees."""
        target = self.get_debug_camera_target_joint()
        target[0] = self._wrap_to_pi(target[0] + np.pi)
        self.set_debug_camera_target_joint(target)
        return self.get_debug_camera_target_joint()

    def _start_debug_camera_control_panel(self) -> None:
        """Start separate attached debug camera control window if tkinter is available."""
        if not self._debug_panel_available:
            return
        if self._debug_panel_thread and self._debug_panel_thread.is_alive():
            return
        self._debug_panel_stop_event.clear()
        self._debug_panel_thread = threading.Thread(
            target=self._run_debug_camera_control_panel,
            daemon=True,
            name="DebugCameraControlPanel",
        )
        self._debug_panel_thread.start()

    def _stop_debug_camera_control_panel(self) -> None:
        """Request attached debug camera control panel shutdown."""
        self._debug_panel_stop_event.set()
        panel_thread = self._debug_panel_thread
        if panel_thread and panel_thread.is_alive():
            panel_thread.join(timeout=2.0)
        self._debug_panel_thread = None

    def _run_debug_camera_control_panel(self) -> None:
        """Tkinter-based separate attached debug camera control window."""
        if not self._debug_panel_available:
            return

        try:
            root = tk.Tk()
        except Exception as e:
            self._debug_panel_available = False
            self._viewer_show_compact_status = True
            print(f"[VIEWER] Debug camera control window unavailable: {e}")
            return

        root.title("Debug Camera Control")
        root.geometry("440x730+20+120")
        root.resizable(False, False)

        main = ttk.Frame(root, padding=10)
        main.pack(fill="both", expand=True)

        view_mode_var = tk.StringVar(value="View mode: -")
        view_on_var = tk.StringVar(
            value="Attached debug camera view: OFF\nLive preview: NOT ACTIVE"
        )
        lr_var = tk.StringVar(value="Camera left/right: +0.0 deg")
        ud_var = tk.StringVar(value="Camera up/down: +0.0 deg")
        roll_var = tk.StringVar(value="Camera roll: +0.0 deg")
        zoom_var = tk.StringVar(value="Zoom (FOV): 65.0 deg")
        left_step_var = tk.StringVar(value=f"{self._debug_camera_step_left_deg:.1f}")
        right_step_var = tk.StringVar(value=f"{self._debug_camera_step_right_deg:.1f}")
        up_step_var = tk.StringVar(value=f"{self._debug_camera_step_up_deg:.1f}")
        down_step_var = tk.StringVar(value=f"{self._debug_camera_step_down_deg:.1f}")
        roll_step_var = tk.StringVar(value=f"{self._debug_camera_roll_step_deg:.1f}")
        zoom_step_var = tk.StringVar(value=f"{self._debug_camera_zoom_step_deg:.1f}")
        help_visible = tk.BooleanVar(value=False)
        slider_sync = {"updating": False}

        joint_limits_deg = np.rad2deg(self.debug_camera_joint_limits)
        yaw_slider_var = tk.DoubleVar(value=0.0)
        pitch_slider_var = tk.DoubleVar(value=0.0)
        roll_slider_var = tk.DoubleVar(value=0.0)
        zoom_slider_var = tk.DoubleVar(value=self.get_debug_camera_zoom_fovy())

        ttk.Label(main, text="Attached Debug Camera", font=("Segoe UI", 10, "bold")).grid(
            row=0, column=0, columnspan=4, sticky="w"
        )
        ttk.Label(main, textvariable=view_mode_var).grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Label(main, textvariable=view_on_var).grid(row=2, column=0, columnspan=4, sticky="w")
        ttk.Label(main, textvariable=lr_var).grid(row=3, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Label(main, textvariable=ud_var).grid(row=4, column=0, columnspan=4, sticky="w")
        ttk.Label(main, textvariable=roll_var).grid(row=5, column=0, columnspan=4, sticky="w")
        ttk.Label(main, textvariable=zoom_var).grid(row=6, column=0, columnspan=4, sticky="w")

        ttk.Label(main, text="Left step (deg)").grid(row=7, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(main, width=7, textvariable=left_step_var).grid(row=7, column=1, sticky="w", pady=(8, 0))
        ttk.Label(main, text="Right step (deg)").grid(row=7, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(main, width=7, textvariable=right_step_var).grid(row=7, column=3, sticky="w", pady=(8, 0))
        ttk.Label(main, text="Up step (deg)").grid(row=8, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(main, width=7, textvariable=up_step_var).grid(row=8, column=1, sticky="w", pady=(4, 0))
        ttk.Label(main, text="Down step (deg)").grid(row=8, column=2, sticky="w", pady=(4, 0))
        ttk.Entry(main, width=7, textvariable=down_step_var).grid(row=8, column=3, sticky="w", pady=(4, 0))
        ttk.Label(main, text="Roll step (deg)").grid(row=9, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(main, width=7, textvariable=roll_step_var).grid(row=9, column=1, sticky="w", pady=(4, 0))
        ttk.Label(main, text="Zoom step (deg)").grid(row=9, column=2, sticky="w", pady=(4, 0))
        ttk.Entry(main, width=7, textvariable=zoom_step_var).grid(row=9, column=3, sticky="w", pady=(4, 0))

        def _read_angle_steps() -> Dict[str, float]:
            def _parse_or_default(raw: str, default: float) -> float:
                try:
                    return float(raw)
                except ValueError:
                    return default

            steps = {
                "left": float(np.clip(_parse_or_default(left_step_var.get(), self._debug_camera_step_left_deg), 0.2, 180.0)),
                "right": float(np.clip(_parse_or_default(right_step_var.get(), self._debug_camera_step_right_deg), 0.2, 180.0)),
                "up": float(np.clip(_parse_or_default(up_step_var.get(), self._debug_camera_step_up_deg), 0.2, 180.0)),
                "down": float(np.clip(_parse_or_default(down_step_var.get(), self._debug_camera_step_down_deg), 0.2, 180.0)),
                "roll": float(np.clip(_parse_or_default(roll_step_var.get(), self._debug_camera_roll_step_deg), 0.2, 180.0)),
            }
            left_step_var.set(f"{steps['left']:.1f}")
            right_step_var.set(f"{steps['right']:.1f}")
            up_step_var.set(f"{steps['up']:.1f}")
            down_step_var.set(f"{steps['down']:.1f}")
            roll_step_var.set(f"{steps['roll']:.1f}")
            self._queue_viewer_camera_command(
                {
                    "type": "panel_set_steps",
                    "left": steps["left"],
                    "right": steps["right"],
                    "up": steps["up"],
                    "down": steps["down"],
                    "roll": steps["roll"],
                },
                coalesce_key="panel_steps",
            )
            return steps

        def _read_zoom_step() -> float:
            try:
                step = float(zoom_step_var.get())
            except ValueError:
                step = self._debug_camera_zoom_step_deg
            step = float(np.clip(step, 0.2, 20.0))
            zoom_step_var.set(f"{step:.1f}")
            self._queue_viewer_camera_command(
                {"type": "panel_set_steps", "zoom_step": step},
                coalesce_key="panel_steps",
            )
            return step

        def _ensure_debug_view_mode() -> None:
            if not self._is_attached_debug_camera_mode(self._viewer_camera_mode):
                self._debug_live_preview_pending = True
                self._queue_viewer_camera_command(RobotConfig.VIEWER_MODE_ATTACHED_DEBUG_CAMERA_VIEW)

        def _camera_left() -> None:
            _read_angle_steps()
            _ensure_debug_view_mode()
            self._queue_viewer_camera_command("debug_cam_yaw_left")

        def _camera_right() -> None:
            _read_angle_steps()
            _ensure_debug_view_mode()
            self._queue_viewer_camera_command("debug_cam_yaw_right")

        def _camera_up() -> None:
            _read_angle_steps()
            _ensure_debug_view_mode()
            self._queue_viewer_camera_command("debug_cam_pitch_up")

        def _camera_down() -> None:
            _read_angle_steps()
            _ensure_debug_view_mode()
            self._queue_viewer_camera_command("debug_cam_pitch_down")

        def _roll_left() -> None:
            _read_angle_steps()
            _ensure_debug_view_mode()
            self._queue_viewer_camera_command("debug_cam_roll_left")

        def _roll_right() -> None:
            _read_angle_steps()
            _ensure_debug_view_mode()
            self._queue_viewer_camera_command("debug_cam_roll_right")

        def _zoom_in() -> None:
            _read_zoom_step()
            _ensure_debug_view_mode()
            self._queue_viewer_camera_command("debug_cam_zoom_in")

        def _zoom_out() -> None:
            _read_zoom_step()
            _ensure_debug_view_mode()
            self._queue_viewer_camera_command("debug_cam_zoom_out")

        def _reset_camera_orientation() -> None:
            _ensure_debug_view_mode()
            self._queue_viewer_camera_command("debug_cam_reset_home")

        def _reset_camera_zoom() -> None:
            _ensure_debug_view_mode()
            self._queue_viewer_camera_command("debug_cam_reset_zoom")

        def _look_at_home_target() -> None:
            _ensure_debug_view_mode()
            self._queue_viewer_camera_command("debug_cam_look_at_home_target")

        def _upright_home_view() -> None:
            _ensure_debug_view_mode()
            self._queue_viewer_camera_command("debug_cam_upright_home")

        def _flip_direction_180() -> None:
            _ensure_debug_view_mode()
            self._queue_viewer_camera_command("debug_cam_flip_direction_180")

        def _set_from_sliders(*_args: Any) -> None:
            if slider_sync["updating"]:
                return
            _ensure_debug_view_mode()
            target = np.array(
                [
                    np.deg2rad(yaw_slider_var.get()),
                    np.deg2rad(-pitch_slider_var.get()),
                    np.deg2rad(roll_slider_var.get()),
                ],
                dtype=float,
            )
            self._queue_viewer_camera_command(
                {
                    "type": "panel_set_target_zoom",
                    "target": target.tolist(),
                    "fovy": float(zoom_slider_var.get()),
                    "ensure_view": True,
                },
                coalesce_key="panel_target_zoom",
            )

        yaw_slider_var.trace_add("write", _set_from_sliders)
        pitch_slider_var.trace_add("write", _set_from_sliders)
        roll_slider_var.trace_add("write", _set_from_sliders)
        zoom_slider_var.trace_add("write", _set_from_sliders)

        ttk.Button(main, text="Enter/Exit Attached Camera View (F7)", command=lambda: self._queue_viewer_camera_command("toggle_attached_debug_camera_view")).grid(
            row=10, column=0, columnspan=4, sticky="ew", pady=(10, 4)
        )
        ttk.Button(main, text="Third-person", command=lambda: self._queue_viewer_camera_command(RobotConfig.VIEWER_MODE_THIRD_PERSON)).grid(
            row=11, column=0, sticky="ew", pady=2
        )
        ttk.Button(main, text="Hand Camera", command=lambda: self._queue_viewer_camera_command(RobotConfig.VIEWER_MODE_HAND_CAMERA_FIXED)).grid(
            row=11, column=1, sticky="ew", pady=2
        )
        ttk.Button(main, text="Inspect Mode", command=lambda: self._queue_viewer_camera_command(RobotConfig.VIEWER_MODE_HAND_CAMERA_INSPECT)).grid(
            row=11, column=2, columnspan=2, sticky="ew", pady=2
        )

        ttk.Button(main, text="Camera Left", command=_camera_left).grid(row=12, column=0, sticky="ew", pady=2)
        ttk.Button(main, text="Camera Right", command=_camera_right).grid(row=12, column=1, sticky="ew", pady=2)
        ttk.Button(main, text="Camera Up", command=_camera_up).grid(row=12, column=2, sticky="ew", pady=2)
        ttk.Button(main, text="Camera Down", command=_camera_down).grid(row=12, column=3, sticky="ew", pady=2)
        ttk.Button(main, text="Roll Left", command=_roll_left).grid(row=13, column=0, sticky="ew", pady=2)
        ttk.Button(main, text="Roll Right", command=_roll_right).grid(row=13, column=1, sticky="ew", pady=2)
        ttk.Button(main, text="Zoom In", command=_zoom_in).grid(row=13, column=2, sticky="ew", pady=2)
        ttk.Button(main, text="Zoom Out", command=_zoom_out).grid(row=13, column=3, sticky="ew", pady=2)

        ttk.Label(main, text="Left/Right angle").grid(row=14, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Scale(main, from_=joint_limits_deg[0, 0], to=joint_limits_deg[0, 1], variable=yaw_slider_var).grid(
            row=15, column=0, columnspan=4, sticky="ew"
        )
        ttk.Label(main, text="Up/Down angle").grid(row=16, column=0, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Scale(main, from_=-joint_limits_deg[1, 1], to=-joint_limits_deg[1, 0], variable=pitch_slider_var).grid(
            row=17, column=0, columnspan=4, sticky="ew"
        )
        ttk.Label(main, text="Roll angle").grid(row=18, column=0, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Scale(main, from_=joint_limits_deg[2, 0], to=joint_limits_deg[2, 1], variable=roll_slider_var).grid(
            row=19, column=0, columnspan=4, sticky="ew"
        )
        ttk.Label(main, text="Zoom (FOV)").grid(row=20, column=0, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Scale(
            main,
            from_=RobotConfig.DEBUG_CAMERA_MAX_FOVY_DEG,
            to=RobotConfig.DEBUG_CAMERA_MIN_FOVY_DEG,
            variable=zoom_slider_var,
        ).grid(row=21, column=0, columnspan=4, sticky="ew")

        ttk.Button(main, text="Reset Camera Orientation", command=_reset_camera_orientation).grid(
            row=22, column=0, columnspan=2, sticky="ew", pady=(8, 2)
        )
        ttk.Button(main, text="Reset Zoom", command=_reset_camera_zoom).grid(
            row=22, column=2, columnspan=2, sticky="ew", pady=(8, 2)
        )
        ttk.Button(main, text="Look At Home Target", command=_look_at_home_target).grid(
            row=23, column=0, columnspan=2, sticky="ew", pady=2
        )
        ttk.Button(main, text="Upright Home View", command=_upright_home_view).grid(
            row=23, column=2, columnspan=2, sticky="ew", pady=2
        )
        ttk.Button(main, text="Flip Direction 180 deg", command=_flip_direction_180).grid(
            row=24, column=0, columnspan=4, sticky="ew", pady=2
        )

        help_label = ttk.Label(
            main,
            text=(
                "Shortcuts in attached debug camera view:\n"
                "- Arrows: camera left/right/up/down\n"
                "- Comma/Period: roll left/right\n"
                "- Plus/Minus: zoom in/out\n"
                "- R: reset camera, 0: reset zoom, U: upright home view, Y: flip direction 180\n"
                "Panel controls auto-enable attached debug camera live preview.\n"
                "Left/Right/Up/Down use independent step options (up to 180 deg)."
            ),
            wraplength=390,
            justify="left",
        )

        def _toggle_help() -> None:
            help_visible.set(not help_visible.get())
            if help_visible.get():
                help_label.grid(row=27, column=0, columnspan=4, sticky="w", pady=(8, 0))
            else:
                help_label.grid_forget()

        ttk.Button(main, text="Show/Hide Help", command=_toggle_help).grid(
            row=26, column=0, columnspan=4, sticky="ew", pady=(8, 0)
        )

        for col in range(4):
            main.columnconfigure(col, weight=1)

        def _on_close() -> None:
            self._debug_panel_window_visible = False
            root.withdraw()

        root.protocol("WM_DELETE_WINDOW", _on_close)

        def _poll() -> None:
            if self._debug_panel_stop_event.is_set():
                root.destroy()
                return

            if self._debug_panel_window_visible:
                if not root.winfo_viewable():
                    root.deiconify()
            else:
                if root.winfo_viewable():
                    root.withdraw()

            mode_label = self._viewer_mode_human_label()
            in_debug_view = self._is_attached_debug_camera_mode(self._viewer_camera_mode)
            if in_debug_view:
                self._debug_live_preview_pending = False
            joint = self.get_debug_camera_joint_position()
            left_right_deg = float(np.rad2deg(joint[0]))
            up_down_deg = float(-np.rad2deg(joint[1]))
            roll_deg = float(np.rad2deg(joint[2]))
            zoom_fovy = self.get_debug_camera_zoom_fovy()

            view_mode_var.set(f"View mode: {mode_label}")
            if in_debug_view:
                live_preview_text = "Live preview: ATTACHED DEBUG CAMERA"
            elif self._debug_live_preview_pending:
                live_preview_text = "Live preview: switching to attached debug camera..."
            else:
                live_preview_text = "Live preview: NOT ACTIVE"
            view_on_var.set(
                f"Attached debug camera view: {'ON' if in_debug_view else 'OFF'}\n{live_preview_text}"
            )
            lr_var.set(f"Camera left/right: {left_right_deg:+.1f} deg")
            ud_var.set(f"Camera up/down: {up_down_deg:+.1f} deg")
            roll_var.set(f"Camera roll: {roll_deg:+.1f} deg")
            zoom_var.set(f"Zoom (FOV): {zoom_fovy:.1f} deg")

            slider_sync["updating"] = True
            yaw_slider_var.set(left_right_deg)
            pitch_slider_var.set(up_down_deg)
            roll_slider_var.set(roll_deg)
            zoom_slider_var.set(zoom_fovy)
            slider_sync["updating"] = False

            root.after(120, _poll)

        root.after(120, _poll)
        root.mainloop()

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

    def _queue_viewer_camera_command(self, command: Any, coalesce_key: Optional[str] = None) -> None:
        """Queue a camera command to be applied in the simulation thread."""
        with self._viewer_command_lock:
            if isinstance(command, dict):
                command = dict(command)
                key = coalesce_key if coalesce_key is not None else command.get("coalesce_key")
                if key is not None:
                    command["coalesce_key"] = key
                    self._viewer_pending_camera_commands = deque(
                        [
                            c for c in self._viewer_pending_camera_commands
                            if not (isinstance(c, dict) and c.get("coalesce_key") == key)
                        ]
                    )
            self._viewer_pending_camera_commands.append(command)

    def _pop_viewer_camera_command(self) -> Optional[Any]:
        """Pop queued camera command if available."""
        with self._viewer_command_lock:
            if not self._viewer_pending_camera_commands:
                return None
            command = self._viewer_pending_camera_commands.popleft()
        return command

    def set_viewer_camera_mode(self, mode: str) -> None:
        """Request viewer camera mode change from any thread."""
        mode_alias = {
            "third": RobotConfig.VIEWER_MODE_THIRD_PERSON,
            "third_person": RobotConfig.VIEWER_MODE_THIRD_PERSON,
            "robot": RobotConfig.VIEWER_MODE_HAND_CAMERA_FIXED,
            "robot_eye": RobotConfig.VIEWER_MODE_HAND_CAMERA_FIXED,  # backward compatibility
            "robot_eye_fixed": RobotConfig.VIEWER_MODE_HAND_CAMERA_FIXED,
            "hand": RobotConfig.VIEWER_MODE_HAND_CAMERA_FIXED,
            "hand_camera": RobotConfig.VIEWER_MODE_HAND_CAMERA_FIXED,
            "hand_camera_fixed": RobotConfig.VIEWER_MODE_HAND_CAMERA_FIXED,
            "robot_eye_debug": RobotConfig.VIEWER_MODE_HAND_CAMERA_INSPECT,  # backward compatibility
            "robot_eye_inspect": RobotConfig.VIEWER_MODE_HAND_CAMERA_INSPECT,
            "hand_camera_inspect": RobotConfig.VIEWER_MODE_HAND_CAMERA_INSPECT,
            "inspect": RobotConfig.VIEWER_MODE_HAND_CAMERA_INSPECT,
            "eye": RobotConfig.VIEWER_MODE_HAND_CAMERA_FIXED,
            "attached_debug_camera_view": RobotConfig.VIEWER_MODE_ATTACHED_DEBUG_CAMERA_VIEW,
            "debug_camera_view": RobotConfig.VIEWER_MODE_ATTACHED_DEBUG_CAMERA_VIEW,
            "attached_debug_camera_control": RobotConfig.VIEWER_MODE_ATTACHED_DEBUG_CAMERA_VIEW,
            "debug_camera_control": RobotConfig.VIEWER_MODE_ATTACHED_DEBUG_CAMERA_VIEW,
            "debug_camera_manual": RobotConfig.VIEWER_MODE_ATTACHED_DEBUG_CAMERA_VIEW,  # backward compatibility
            "debug_camera_manual_mode": RobotConfig.VIEWER_MODE_ATTACHED_DEBUG_CAMERA_VIEW,  # backward compatibility
            "toggle": "toggle",
        }
        normalized = mode_alias.get(mode.strip().lower())
        if normalized is None:
            raise ValueError(
                "mode must be one of: third_person, hand_camera_fixed, hand_camera_inspect, "
                "attached_debug_camera_view, toggle"
            )
        self._queue_viewer_camera_command(normalized)

    def toggle_viewer_camera_mode(self) -> None:
        """Request camera toggle from any thread."""
        self._queue_viewer_camera_command("toggle")

    def toggle_viewer_control_debug(self) -> None:
        """Backward-compatible alias for debug camera control window toggle."""
        self._queue_viewer_camera_command("toggle_debug_camera_panel_window")

    def toggle_viewer_compact_status(self) -> None:
        """Backward-compatible alias for debug camera control window toggle."""
        self._queue_viewer_camera_command("toggle_debug_camera_panel_window")

    def toggle_viewer_debug_camera_panel_window(self) -> None:
        """Toggle separate debug camera control window."""
        self._queue_viewer_camera_command("toggle_debug_camera_panel_window")

    def toggle_viewer_help(self) -> None:
        """Toggle extended camera help panel."""
        self._queue_viewer_camera_command("toggle_help")

    def toggle_viewer_attached_debug_camera_view(self) -> None:
        """Toggle attached debug camera view on/off."""
        self._queue_viewer_camera_command("toggle_attached_debug_camera_view")

    def toggle_viewer_attached_debug_camera_control(self) -> None:
        """Backward-compatible alias for attached debug camera view toggle."""
        self._queue_viewer_camera_command("toggle_attached_debug_camera_view")

    def toggle_viewer_debug_camera_manual_mode(self) -> None:
        """Backward-compatible alias for attached debug camera view toggle."""
        self._queue_viewer_camera_command("toggle_attached_debug_camera_view")

    def _apply_viewer_camera_command(self, viewer: Any, command: Any) -> None:
        """Apply queued command on viewer state."""
        if isinstance(command, dict):
            cmd_type = str(command.get("type", "")).strip().lower()
            if cmd_type == "panel_set_steps":
                self.set_debug_camera_direction_steps_deg(
                    left=command.get("left"),
                    right=command.get("right"),
                    up=command.get("up"),
                    down=command.get("down"),
                    roll=command.get("roll"),
                )
                if "zoom_step" in command:
                    self.set_debug_camera_zoom_step_deg(float(command["zoom_step"]))
                return
            if cmd_type == "panel_set_target_zoom":
                if command.get("ensure_view", True) and not self._is_attached_debug_camera_mode(self._viewer_camera_mode):
                    self._viewer_mode_before_attached_debug = self._normalize_non_debug_restore_mode(
                        self._viewer_camera_mode
                    )
                    self._set_viewer_camera_mode(viewer, RobotConfig.VIEWER_MODE_ATTACHED_DEBUG_CAMERA_VIEW)
                    self._debug_live_preview_pending = False
                target = np.array(command.get("target", []), dtype=float).reshape(-1)
                if target.shape[0] == 3:
                    self.set_debug_camera_target_joint(target)
                if "fovy" in command:
                    self.set_debug_camera_zoom_fovy(float(command["fovy"]))
                return

        if command == "toggle":
            self._toggle_viewer_camera_mode(viewer)
            print(f"[VIEWER] View mode: {self._viewer_mode_human_label()}.")
            return
        if command in ("toggle_debug_camera_panel_window", "toggle_compact_status"):
            if self._debug_panel_available:
                self._debug_panel_window_visible = not self._debug_panel_window_visible
                status = "ON" if self._debug_panel_window_visible else "OFF"
                print(f"[VIEWER] Debug camera control window: {status}.")
            else:
                self._viewer_show_compact_status = not self._viewer_show_compact_status
                status = "ON" if self._viewer_show_compact_status else "OFF"
                print(f"[VIEWER] Overlay fallback status: {status}.")
            return
        if command == "toggle_help":
            self._viewer_show_extended_help = not self._viewer_show_extended_help
            status = "ON" if self._viewer_show_extended_help else "OFF"
            print(f"[VIEWER] Extended help: {status}.")
            return
        if command == "toggle_attached_debug_camera_view":
            if self._is_attached_debug_camera_mode(self._viewer_camera_mode):
                restore_mode = self._normalize_non_debug_restore_mode(
                    self._viewer_mode_before_attached_debug
                )
                self._set_viewer_camera_mode(viewer, restore_mode)
                self._debug_live_preview_pending = False
                print(
                    "[VIEWER] Attached debug camera view: OFF. "
                    f"View mode: {self._viewer_mode_human_label()}."
                )
            else:
                self._viewer_mode_before_attached_debug = self._normalize_non_debug_restore_mode(
                    self._viewer_camera_mode
                )
                if self._set_viewer_camera_mode(viewer, RobotConfig.VIEWER_MODE_ATTACHED_DEBUG_CAMERA_VIEW):
                    self._debug_live_preview_pending = False
                    print("[VIEWER] Attached debug camera view: ON.")
            return
        if command == "toggle_attached_debug_camera_control_mode":
            # Backward-compatible no-op wrapper: camera controls stay in view mode.
            if not self._is_attached_debug_camera_mode(self._viewer_camera_mode):
                self._viewer_mode_before_attached_debug = self._normalize_non_debug_restore_mode(
                    self._viewer_camera_mode
                )
                if self._set_viewer_camera_mode(viewer, RobotConfig.VIEWER_MODE_ATTACHED_DEBUG_CAMERA_VIEW):
                    self._debug_live_preview_pending = False
                    print("[VIEWER] Attached debug camera view: ON.")
            return
        if command == "debug_cam_yaw_left":
            if self._is_attached_debug_camera_mode(self._viewer_camera_mode):
                self._nudge_debug_camera_target(delta_yaw=+np.deg2rad(self._debug_camera_step_left_deg))
            return
        if command == "debug_cam_yaw_right":
            if self._is_attached_debug_camera_mode(self._viewer_camera_mode):
                self._nudge_debug_camera_target(delta_yaw=-np.deg2rad(self._debug_camera_step_right_deg))
            return
        if command == "debug_cam_pitch_up":
            if self._is_attached_debug_camera_mode(self._viewer_camera_mode):
                self._nudge_debug_camera_target(delta_pitch=-np.deg2rad(self._debug_camera_step_up_deg))
            return
        if command == "debug_cam_pitch_down":
            if self._is_attached_debug_camera_mode(self._viewer_camera_mode):
                self._nudge_debug_camera_target(delta_pitch=+np.deg2rad(self._debug_camera_step_down_deg))
            return
        if command == "debug_cam_roll_left":
            if self._is_attached_debug_camera_mode(self._viewer_camera_mode):
                self._nudge_debug_camera_target(delta_roll=+np.deg2rad(self._debug_camera_roll_step_deg))
            return
        if command == "debug_cam_roll_right":
            if self._is_attached_debug_camera_mode(self._viewer_camera_mode):
                self._nudge_debug_camera_target(delta_roll=-np.deg2rad(self._debug_camera_roll_step_deg))
            return
        if command == "debug_cam_zoom_in":
            if self._is_attached_debug_camera_mode(self._viewer_camera_mode):
                self.nudge_debug_camera_zoom(-self._debug_camera_zoom_step_deg)
            return
        if command == "debug_cam_zoom_out":
            if self._is_attached_debug_camera_mode(self._viewer_camera_mode):
                self.nudge_debug_camera_zoom(+self._debug_camera_zoom_step_deg)
            return
        if command == "debug_cam_reset_home":
            self.reset_debug_camera_orientation()
            return
        if command == "debug_cam_reset_zoom":
            self.reset_debug_camera_zoom()
            return
        if command == "debug_cam_look_at_home_target":
            self.look_at_debug_camera_home_target()
            return
        if command == "debug_cam_upright_home":
            self.upright_reset_debug_camera()
            return
        if command == "debug_cam_flip_direction_180":
            self.flip_debug_camera_direction_180()
            return

        if self._is_attached_debug_camera_mode(command) and not self._is_attached_debug_camera_mode(self._viewer_camera_mode):
            self._viewer_mode_before_attached_debug = self._normalize_non_debug_restore_mode(
                self._viewer_camera_mode
            )

        self._set_viewer_camera_mode(viewer, command)
        self._debug_live_preview_pending = False
        print(f"[VIEWER] View mode: {self._viewer_mode_human_label()}.")

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
        """Get current attached debug camera angles [yaw, pitch, roll] in radians."""
        return np.array([self.data.qpos[idx] for idx in self.debug_camera_qpos_indices], dtype=float)

    def get_debug_camera_target_joint(self) -> np.ndarray:
        """Get target attached debug camera angles [yaw, pitch, roll] in radians."""
        return self._debug_camera_target_joint.copy()

    def set_debug_camera_target_joint(self, target_joint: np.ndarray) -> None:
        """Set attached debug camera target angles [yaw, pitch, roll] in radians."""
        target = np.array(target_joint, dtype=float).reshape(-1)
        if target.shape[0] == 2:
            # Backward compatibility: [yaw, pitch] keeps current roll.
            target = np.array([target[0], target[1], self._debug_camera_target_joint[2]], dtype=float)
        if target.shape[0] != 3:
            raise ValueError("target_joint must have shape [3] for [yaw, pitch, roll]")
        clipped = np.clip(target, self.debug_camera_joint_limits[:, 0], self.debug_camera_joint_limits[:, 1])
        self._debug_camera_target_joint[:] = clipped

    def get_debug_camera_joint_diff(self) -> np.ndarray:
        """Get attached debug camera joint errors [yaw, pitch, roll] in radians."""
        diff = self._debug_camera_target_joint - self.get_debug_camera_joint_position()
        diff[0] = self._wrap_to_pi(diff[0])  # yaw wraps around naturally
        diff[2] = self._wrap_to_pi(diff[2])  # roll wraps around naturally
        return diff

    def get_debug_camera_joint_velocity(self) -> np.ndarray:
        """Get attached debug camera joint velocities [yaw, pitch, roll] in rad/s."""
        return np.array([self.data.qvel[idx] for idx in self.debug_camera_dof_indices], dtype=float)

    def _compute_debug_camera_control(self) -> np.ndarray:
        """Compute attached debug camera control commands [yaw, pitch, roll]."""
        return self._debug_camera_target_joint.copy()

    def _apply_debug_camera_stabilization(self, recompute_kinematics: bool = False) -> None:
        """Keep attached debug camera joints stable for usable vision debugging."""
        clipped = np.clip(
            self._debug_camera_target_joint,
            self.debug_camera_joint_limits[:, 0],
            self.debug_camera_joint_limits[:, 1],
        )
        self._debug_camera_target_joint[:] = clipped
        for i, qpos_idx in enumerate(self.debug_camera_qpos_indices):
            self.data.qpos[qpos_idx] = clipped[i]
            self.data.qvel[self.debug_camera_dof_indices[i]] = 0.0
            self.data.ctrl[self.debug_camera_actuator_ids[i]] = clipped[i]
        if recompute_kinematics:
            mujoco.mj_forward(self.model, self.data)

    def _nudge_debug_camera_target(
        self,
        delta_yaw: float = 0.0,
        delta_pitch: float = 0.0,
        delta_roll: float = 0.0,
    ) -> None:
        """Increment attached debug camera target orientation by small deltas."""
        target = self.get_debug_camera_target_joint()
        target[0] += delta_yaw
        target[1] += delta_pitch
        target[2] += delta_roll
        self.set_debug_camera_target_joint(target)

    def get_debug_camera_zoom_fovy(self) -> float:
        """Get attached debug camera vertical field-of-view in degrees."""
        return float(self._debug_camera_fovy_deg)

    def set_debug_camera_zoom_fovy(self, fovy_deg: float) -> float:
        """Set attached debug camera vertical field-of-view in degrees."""
        fovy = float(np.clip(fovy_deg, RobotConfig.DEBUG_CAMERA_MIN_FOVY_DEG, RobotConfig.DEBUG_CAMERA_MAX_FOVY_DEG))
        self._debug_camera_fovy_deg = fovy
        self.model.cam_fovy[self.debug_camera_id] = fovy
        return self._debug_camera_fovy_deg

    def nudge_debug_camera_zoom(self, delta_fovy_deg: float) -> float:
        """Adjust attached debug camera zoom by delta FOV in degrees."""
        return self.set_debug_camera_zoom_fovy(self._debug_camera_fovy_deg + float(delta_fovy_deg))

    def reset_debug_camera_zoom(self) -> float:
        """Reset attached debug camera zoom to home FOV."""
        return self.set_debug_camera_zoom_fovy(self._debug_camera_home_fovy_deg)

    def look_at_point(self, target_xyz: np.ndarray) -> np.ndarray:
        """Set attached debug camera target so the rig looks at a world-space point."""
        target = np.array(target_xyz, dtype=float).reshape(-1)
        if target.shape[0] != 3:
            raise ValueError("target_xyz must have shape [3] for [x, y, z]")

        self._require_valid_id(self.debug_camera_yaw_body_id, RobotConfig.DEBUG_CAMERA_YAW_BODY_NAME)

        # Use yaw-parent frame to compute yaw angle.
        yaw_origin_world = self.data.xpos[self.debug_camera_yaw_body_id].copy()
        yaw_parent_rot_world = self.data.xmat[self.debug_camera_yaw_parent_body_id].reshape(3, 3).copy()
        vec_world = target - yaw_origin_world
        if np.linalg.norm(vec_world) < 1e-6:
            return self.get_debug_camera_target_joint()
        vec_parent = yaw_parent_rot_world.T @ vec_world
        yaw = np.arctan2(vec_parent[1], vec_parent[0])

        # Pitch is computed in yaw-rotated frame from pitch joint location.
        yaw_rot_local = R.from_euler("z", yaw).as_matrix()
        vec_after_yaw = yaw_rot_local.T @ vec_parent
        vec_from_pitch = vec_after_yaw - RobotConfig.DEBUG_CAMERA_PITCH_OFFSET_LOCAL
        if abs(vec_from_pitch[0]) < 1e-6 and abs(vec_from_pitch[2]) < 1e-6:
            return self.get_debug_camera_target_joint()
        pitch = np.arctan2(-vec_from_pitch[2], vec_from_pitch[0])

        # Set roll to zero for stable "look-at" behavior.
        target_joint = np.array([yaw, pitch, 0.0], dtype=float)
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

        control = current_pos + p_term + i_term - d_term

        # Hold near-setpoint joints at exact target to suppress micro-jitter
        # that propagates to attached camera view.
        hold_mask = np.abs(pos_error) < RobotConfig.ARM_HOLD_POSITION_EPS
        if np.any(hold_mask):
            self._arm_error_integral[hold_mask] = 0.0
            control[hold_mask] = self._arm_target_joint[hold_mask]

        return control

    def _stabilize_arm_for_debug_camera_view(self) -> None:
        """Suppress tiny arm tremors that visibly shake attached debug camera view."""
        pos_error = self.get_arm_joint_diff()
        vel = self.get_arm_joint_velocity()
        hold_mask = (
            np.abs(pos_error) < RobotConfig.ARM_DEBUG_STABILIZE_POS_EPS
        ) & (
            np.abs(vel) < RobotConfig.ARM_DEBUG_STABILIZE_VEL_EPS
        )
        if not np.any(hold_mask):
            return
        for i, joint_id in enumerate(self.arm_joint_ids):
            if not hold_mask[i]:
                continue
            qpos_idx = int(self.model.jnt_qposadr[joint_id])
            dof_idx = int(self.model.jnt_dofadr[joint_id])
            self.data.qpos[qpos_idx] = self._arm_target_joint[i]
            self.data.qvel[dof_idx] = 0.0

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
        self._start_debug_camera_control_panel()

        def key_callback(keycode: int) -> None:
            if keycode == RobotConfig.VIEWER_KEY_ATTACHED_DEBUG_CAMERA_VIEW_TOGGLE:
                if not self._accept_viewer_key(keycode, RobotConfig.VIEWER_FN_KEY_DEBOUNCE_SEC):
                    return
                self._queue_viewer_camera_command("toggle_attached_debug_camera_view")
            elif keycode == RobotConfig.VIEWER_KEY_TOGGLE:
                if not self._accept_viewer_key(keycode, RobotConfig.VIEWER_FN_KEY_DEBOUNCE_SEC):
                    return
                self._queue_viewer_camera_command("toggle")
            elif keycode == RobotConfig.VIEWER_KEY_THIRD_PERSON:
                if not self._accept_viewer_key(keycode, RobotConfig.VIEWER_FN_KEY_DEBOUNCE_SEC):
                    return
                self._queue_viewer_camera_command(RobotConfig.VIEWER_MODE_THIRD_PERSON)
            elif keycode == RobotConfig.VIEWER_KEY_HAND_CAMERA_FIXED:
                if not self._accept_viewer_key(keycode, RobotConfig.VIEWER_FN_KEY_DEBOUNCE_SEC):
                    return
                self._queue_viewer_camera_command(RobotConfig.VIEWER_MODE_HAND_CAMERA_FIXED)
            elif keycode == RobotConfig.VIEWER_KEY_HAND_CAMERA_INSPECT:
                if not self._accept_viewer_key(keycode, RobotConfig.VIEWER_FN_KEY_DEBOUNCE_SEC):
                    return
                self._queue_viewer_camera_command(RobotConfig.VIEWER_MODE_HAND_CAMERA_INSPECT)
            elif keycode == RobotConfig.VIEWER_KEY_TOGGLE_DEBUG_CAMERA_PANEL:
                if not self._accept_viewer_key(keycode, RobotConfig.VIEWER_FN_KEY_DEBOUNCE_SEC):
                    return
                self._queue_viewer_camera_command("toggle_debug_camera_panel_window")
            elif keycode in (RobotConfig.VIEWER_KEY_TOGGLE_HELP, ord("h"), ord("H")):
                if not self._accept_viewer_key(RobotConfig.VIEWER_KEY_TOGGLE_HELP, RobotConfig.VIEWER_FN_KEY_DEBOUNCE_SEC):
                    return
                self._queue_viewer_camera_command("toggle_help")
            elif self._is_attached_debug_camera_mode(self._viewer_camera_mode):
                if keycode == RobotConfig.VIEWER_KEY_DEBUG_CAMERA_TILT_UP:
                    if not self._accept_viewer_key(keycode, RobotConfig.VIEWER_ARROW_KEY_REPEAT_SEC):
                        return
                    self._queue_viewer_camera_command("debug_cam_pitch_up")
                elif keycode == RobotConfig.VIEWER_KEY_DEBUG_CAMERA_TILT_DOWN:
                    if not self._accept_viewer_key(keycode, RobotConfig.VIEWER_ARROW_KEY_REPEAT_SEC):
                        return
                    self._queue_viewer_camera_command("debug_cam_pitch_down")
                elif keycode == RobotConfig.VIEWER_KEY_DEBUG_CAMERA_PAN_LEFT:
                    if not self._accept_viewer_key(keycode, RobotConfig.VIEWER_ARROW_KEY_REPEAT_SEC):
                        return
                    self._queue_viewer_camera_command("debug_cam_yaw_left")
                elif keycode == RobotConfig.VIEWER_KEY_DEBUG_CAMERA_PAN_RIGHT:
                    if not self._accept_viewer_key(keycode, RobotConfig.VIEWER_ARROW_KEY_REPEAT_SEC):
                        return
                    self._queue_viewer_camera_command("debug_cam_yaw_right")
                elif keycode == RobotConfig.VIEWER_KEY_DEBUG_CAMERA_ROLL_LEFT:
                    if not self._accept_viewer_key(keycode, RobotConfig.VIEWER_ARROW_KEY_REPEAT_SEC):
                        return
                    self._queue_viewer_camera_command("debug_cam_roll_left")
                elif keycode == RobotConfig.VIEWER_KEY_DEBUG_CAMERA_ROLL_RIGHT:
                    if not self._accept_viewer_key(keycode, RobotConfig.VIEWER_ARROW_KEY_REPEAT_SEC):
                        return
                    self._queue_viewer_camera_command("debug_cam_roll_right")
                elif keycode in (RobotConfig.VIEWER_KEY_DEBUG_CAMERA_ZOOM_IN, glfw.KEY_KP_ADD):
                    if not self._accept_viewer_key(keycode, RobotConfig.VIEWER_ARROW_KEY_REPEAT_SEC):
                        return
                    self._queue_viewer_camera_command("debug_cam_zoom_in")
                elif keycode in (RobotConfig.VIEWER_KEY_DEBUG_CAMERA_ZOOM_OUT, glfw.KEY_KP_SUBTRACT):
                    if not self._accept_viewer_key(keycode, RobotConfig.VIEWER_ARROW_KEY_REPEAT_SEC):
                        return
                    self._queue_viewer_camera_command("debug_cam_zoom_out")
                elif keycode in (RobotConfig.VIEWER_KEY_DEBUG_CAMERA_RESET_ORIENTATION, ord("r"), ord("R")):
                    if not self._accept_viewer_key(RobotConfig.VIEWER_KEY_DEBUG_CAMERA_RESET_ORIENTATION, RobotConfig.VIEWER_FN_KEY_DEBOUNCE_SEC):
                        return
                    self._queue_viewer_camera_command("debug_cam_reset_home")
                elif keycode == RobotConfig.VIEWER_KEY_DEBUG_CAMERA_RESET_ZOOM:
                    if not self._accept_viewer_key(RobotConfig.VIEWER_KEY_DEBUG_CAMERA_RESET_ZOOM, RobotConfig.VIEWER_FN_KEY_DEBOUNCE_SEC):
                        return
                    self._queue_viewer_camera_command("debug_cam_reset_zoom")
                elif keycode in (RobotConfig.VIEWER_KEY_DEBUG_CAMERA_UPRIGHT_HOME, ord("u"), ord("U")):
                    if not self._accept_viewer_key(RobotConfig.VIEWER_KEY_DEBUG_CAMERA_UPRIGHT_HOME, RobotConfig.VIEWER_FN_KEY_DEBOUNCE_SEC):
                        return
                    self._queue_viewer_camera_command("debug_cam_upright_home")
                elif keycode in (RobotConfig.VIEWER_KEY_DEBUG_CAMERA_FLIP_DIRECTION, ord("y"), ord("Y")):
                    if not self._accept_viewer_key(RobotConfig.VIEWER_KEY_DEBUG_CAMERA_FLIP_DIRECTION, RobotConfig.VIEWER_FN_KEY_DEBOUNCE_SEC):
                        return
                    self._queue_viewer_camera_command("debug_cam_flip_direction_180")

        try:
            with mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_callback) as v:
                # Camera setup (default: third-person)
                with v.lock():
                    self._set_viewer_camera_mode(v, RobotConfig.VIEWER_MODE_THIRD_PERSON)
                    self._set_viewer_overlay(v)

                    # Hide geom group 0 to keep collision geoms off.
                    # Debug camera mount geoms are explicitly assigned to group 1 in XML.
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
                    "'F7' attached_debug_camera_view on/off, 'F8' third<->hand_fixed, "
                    "'F9' third_person, 'F10' hand_camera_fixed, 'F11' hand_camera_inspect, "
                    "'F12' debug control window on/off, 'H' help; "
                    "arrows rotate, ','/'.' roll, '+'/'-' zoom, 'R' reset, '0' zoom reset, 'U' upright home, "
                    "'Y' flip 180 "
                    "in attached debug camera view."
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
                    if self._is_attached_debug_camera_mode(self._viewer_camera_mode):
                        self._stabilize_arm_for_debug_camera_view()
                    if RobotConfig.DEBUG_CAMERA_ENABLE_KINEMATIC_STABILIZATION:
                        self._apply_debug_camera_stabilization(
                            recompute_kinematics=RobotConfig.DEBUG_CAMERA_STABILIZATION_RECOMPUTE_FORWARD
                        )

                    if self._viewer_camera_mode == RobotConfig.VIEWER_MODE_HAND_CAMERA_INSPECT:
                        with v.lock():
                            self._update_robot_eye_debug_camera(v)

                    now = time.time()
                    if now - self._viewer_overlay_last_update_time >= self._viewer_overlay_update_period:
                        with v.lock():
                            self._set_viewer_overlay(v)
                        self._viewer_overlay_last_update_time = now

                    v.sync()
        finally:
            self._stop_debug_camera_control_panel()

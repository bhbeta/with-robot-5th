# Mobile Manipulator Robot Control API

## Environment & Constraints
- **Sandbox**: Restricted Python. **import not allowed**. `time` and `math` are pre-loaded.
- **Coordinates**: World frame (meters, radians).
- **Mobile Base**: x, y, theta.
- **Arm**: 7-DOF Panda.
- **EE**: Position [x, y, z] only.

## Mobile Base Control

`get_mobile_position()` -> `list[float]`
Returns current base position `[x, y, theta]`.

`set_mobile_target_position(mobile_target_position, timeout=10.0, verbose=False)` -> `bool`
Sets target `[x, y, theta]`. Returns `True` if converged.

`plan_mobile_path(target_joint, grid_size=0.1)` -> `list[list[float]] | None`
Plans A* path to `[x, y]` (world coords). Returns list of `[x, y, theta]` waypoints or `None`.

`get_grid_map()` -> `list[list[int]]`
Returns 2D binary occupancy grid (0=free, 1=occupied). 0.1m cell size.

`follow_mobile_path(path_world, timeout_per_waypoint=30.0, verbose=False)` -> `bool`
Follows path from `plan_mobile_path`.

## Arm Control

`get_arm_joint_position()` -> `list[float]`
Returns 7 joint angles `[j1..j7]` in radians.

`set_arm_target_joint(arm_target_position, timeout=10.0, verbose=False)` -> `bool`
Sets 7 joint angles. Returns `True` if converged.

## End Effector & Gripper

`get_ee_position()` -> `tuple(list[float], list[float])`
Returns `([x,y,z], [roll,pitch,yaw])` in world frame.

`set_ee_target_position(target_pos, timeout=10.0, verbose=False)` -> `bool`
Sets EE position `[x, y, z]`. Orientation is not controlled.

`get_gripper_width()` -> `float`
Returns width in meters.

`set_target_gripper_width(target_width, timeout=10.0, verbose=False)` -> `bool`
Sets width (0.0=closed, 0.08=open).

## High-Level Operations

`pick_object(object_pos, approach_height=0.1, lift_height=0.2, return_to_home=True, timeout=10.0, verbose=False)` -> `bool`
Executes pick sequence at `object_pos`.

`place_object(place_pos, approach_height=0.2, retract_height=0.3, return_to_home=True, timeout=10.0, verbose=False)` -> `bool`
Executes place sequence at `place_pos`.

`get_object_positions()` -> `dict`
Returns dict of objects: `{'name': {'id': int, 'pos': [x,y,z], 'ori': [r,p,y]}}`.

## Vision Baseline (Simulator Camera)

`list_cameras()` -> `list[str]`
Returns available named MuJoCo cameras in the model.

`get_camera_intrinsics(width=320, height=240, camera_name=None)` -> `dict`
Returns pinhole intrinsics `{fx, fy, cx, cy, fov_y_deg, width, height}` for a named camera.
If `camera_name=None`, returns baseline intrinsics for a configured free camera.

`capture_camera_frame(width=320, height=240, camera_name=None, include_depth=True)` -> `dict`
Returns `{"rgb": np.ndarray, "depth": np.ndarray | None}` from offscreen rendering.

## Debug Camera Rig & Point Cloud

`get_debug_camera_joint_position()` -> `list[float]`
Returns attached debug camera orientation angles `[left_right, up_down, roll]` in radians.

`set_debug_camera_target_joint(target_joint, timeout=5.0, verbose=False)` -> `bool`
Sets attached debug camera orientation target angles. Returns `True` if converged.

`reset_debug_camera_orientation(timeout=5.0, verbose=False)` -> `bool`
Resets attached debug camera direction to home orientation.

`upright_reset_debug_camera(timeout=5.0, verbose=False)` -> `bool`
Resets attached debug camera to upright home view (`roll=0`) and home zoom.

`get_debug_camera_zoom_fovy()` -> `float`
Returns attached debug camera zoom as vertical FOV in degrees.

`set_debug_camera_zoom_fovy(fovy_deg)` -> `float`
Sets attached debug camera zoom as vertical FOV in degrees.

`reset_debug_camera_zoom()` -> `float`
Resets attached debug camera zoom to home FOV.

`look_at_point(target_xyz, timeout=5.0, verbose=False)` -> `bool`
Rotates attached debug camera so it looks at a world point `[x, y, z]`.

`get_camera_intrinsics(camera_name, width=320, height=240)` -> `dict`
Returns intrinsics for a named camera.

`get_camera_extrinsics(camera_name)` -> `dict`
Returns `world_from_camera` / `camera_from_world` transforms and camera world pose.
Frame convention is OpenCV-like camera frame (`+X` right, `+Y` down, `+Z` forward).

`get_camera_point_cloud(camera_name, max_depth=4.0, stride=4, frame="world", width=320, height=240)` -> `dict`
Generates point cloud from RGB+depth:
- `points`: Nx3 list
- `colors`: Nx3 list in `[0,1]`
- `frame`: `"world"` or `"camera"`
- `camera_name`: source camera
- `num_points`: number of valid points

Viewer helper:
- `set_viewer_camera_mode(mode)` supports:
  - `third_person`
  - `hand_camera_fixed`
  - `hand_camera_inspect`
  - `attached_debug_camera_view`
  - `toggle`
- `toggle_viewer_attached_debug_camera_view()` toggles attached debug camera view on/off.
- `toggle_viewer_attached_debug_camera_control()` is a backward-compatible alias for attached debug camera view.
- `toggle_viewer_debug_camera_manual_mode()` is a backward-compatible alias for attached debug camera view.
- `toggle_viewer_compact_status()` toggles the separate debug camera control window.
- `toggle_viewer_debug_camera_panel_window()` toggles the separate debug camera control window.
- `toggle_viewer_help()` toggles extended help panel.
- A separate debug camera control window is shown by default when tkinter is available.
- If tkinter is unavailable, compact overlay fallback is used.
- While attached debug camera view is active:
  - Arrow keys rotate camera left/right/up/down
  - `,` / `.` rotate camera roll
  - `+` / `-` zoom in/out
  - `R` resets camera orientation
  - `0` resets zoom
  - `U` restores upright home view
- Debug camera joints are stabilized kinematically each simulation step for low-shake vision debugging.
- Attached debug camera controls affect only the debug camera rig, not mobile/arm/gripper APIs.

`GET /vision/frame` query options:
- `width`, `height`: positive integers (max `1280x720`)
- `camera_name`: optional named MuJoCo camera
- `include_rgb`: include RGB array in JSON response
- `include_depth`: include depth array in JSON response

## Constants
- `PI`: 3.14159...
- `RESULT`: Dict for return values.

## Examples

### Pick & Place Workflow
```python
objects = get_object_positions()
# object keys may vary. check available keys first if needed.
object_pos = objects['object_red_0']['pos']
bowl_pos = objects['object_yellow_bowl_7']['pos']

# 1. Approach and Pick
path = plan_mobile_path(object_pos)
if path:
    follow_mobile_path(path)
pick_success = pick_object(object_pos, approach_height=0.1, lift_height=0.2)

if not pick_success:
    RESULT['status'] = 'pick_failed'
else:
    # 2. Move to Place Location
    path = plan_mobile_path(bowl_pos)
    if path:
        follow_mobile_path(path)
    # 3. Place Object
    place_success = place_object(bowl_pos, approach_height=0.2, retract_height=0.3)
    RESULT['status'] = 'completed' if place_success else 'place_failed'
```

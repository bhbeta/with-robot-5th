"""FastAPI server for MuJoCo robot simulation with REST API control."""

import time
import queue
import threading
import uvicorn
from typing import Dict, Any, Optional
from fastapi import FastAPI, Response, status, HTTPException
from simulator import MujocoSimulator
import code_repository


# Server configuration
HOST = "0.0.0.0"  # Listen on all network interfaces
PORT = 8800       # API server port
VERSION = "0.0.1"

# FastAPI application instance
app = FastAPI(
    title="MuJoCo Robot Simulator API",
    description="Control Panda-Omron mobile robot via REST API",
    version=VERSION
)

# Create simulator instance and inject into code_repository
simulator = MujocoSimulator()
code_repository.simulator = simulator


def process_actions(action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process action."""
    RESULT = {}
    if action["type"] == "run_code":
        code_str = action["payload"].get("code")
        try:
            RESULT = code_repository.exec_code(code_str)
            print(f"Code execution completed: {RESULT}")
        except Exception as e:
            # Log errors without crashing the simulator
            print(f"\n[EXECUTION ERROR]")
            print(f"  Type: {type(e).__name__}")
            print(f"  Message: {e}")
            import traceback
            print(f"\n[TRACEBACK]")
            traceback.print_exc()
    print("=" * 60 + "\n")
    return RESULT


def run_simulator() -> None:
    """Run MuJoCo simulator in background thread."""
    simulator.run()


@app.get("/")
def read_root() -> Dict[str, str]:
    """Get server info."""
    return {"name": "MuJoCo Robot Simulator", "version": VERSION, "status": "running"}


@app.get("/env")
def get_environment():
    """Collect environment snapshot with object poses and robot state."""
    objects = simulator.get_object_positions()
    for obj in objects.values():
        obj['pos'] = obj['pos'].tolist()
        obj['ori'] = obj['ori'].tolist()
    return {
        "timestamp": time.time(),
        "objects": objects,
    }


@app.get("/vision/cameras")
def get_cameras() -> Dict[str, Any]:
    """List available named cameras and free-camera baseline intrinsics."""
    width = 320
    height = 240
    return {
        "timestamp": time.time(),
        "named_cameras": simulator.list_cameras(),
        "free_camera_intrinsics": simulator.get_camera_intrinsics(width=width, height=height, camera_name=None),
    }


@app.get("/vision/frame")
def get_vision_frame(
    width: int = 320,
    height: int = 240,
    camera_name: Optional[str] = None,
    include_depth: bool = True,
    include_rgb: bool = True
) -> Dict[str, Any]:
    """Capture RGB/depth frame for vision-baseline experiments."""
    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="width/height must be positive integers")
    if width > 1280 or height > 720:
        raise HTTPException(status_code=400, detail="width/height too large (max: 1280x720)")
    if not include_rgb and not include_depth:
        raise HTTPException(status_code=400, detail="At least one of include_rgb/include_depth must be true")

    try:
        frame = simulator.capture_camera_frame(
            width=width,
            height=height,
            camera_name=camera_name,
            include_depth=include_depth
        )
        intrinsics = simulator.get_camera_intrinsics(width=width, height=height, camera_name=camera_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    response: Dict[str, Any] = {
        "timestamp": time.time(),
        "camera_name": camera_name or "free_camera",
        "intrinsics": intrinsics,
    }
    if include_rgb:
        response["rgb"] = frame["rgb"].tolist()
    if include_depth and frame["depth"] is not None:
        response["depth"] = frame["depth"].astype(float).tolist()
    return response


@app.get("/vision/extrinsics")
def get_vision_extrinsics(camera_name: str) -> Dict[str, Any]:
    """Return camera extrinsics for a named camera."""
    try:
        extrinsics = simulator.get_camera_extrinsics(camera_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "timestamp": time.time(),
        "camera_name": camera_name,
        "extrinsics": extrinsics,
    }


@app.get("/vision/point-cloud")
def get_vision_point_cloud(
    camera_name: str,
    width: int = 320,
    height: int = 240,
    max_depth: float = 4.0,
    stride: int = 4,
    frame: str = "world"
) -> Dict[str, Any]:
    """Generate point cloud from camera RGBD frame."""
    if width <= 0 or height <= 0:
        raise HTTPException(status_code=400, detail="width/height must be positive integers")
    if width > 1280 or height > 720:
        raise HTTPException(status_code=400, detail="width/height too large (max: 1280x720)")
    if stride <= 0:
        raise HTTPException(status_code=400, detail="stride must be a positive integer")
    if frame not in ("world", "camera"):
        raise HTTPException(status_code=400, detail="frame must be one of: world, camera")

    try:
        point_cloud = simulator.get_camera_point_cloud(
            camera_name=camera_name,
            width=width,
            height=height,
            max_depth=max_depth,
            stride=stride,
            frame=frame,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    return {
        "timestamp": time.time(),
        "point_cloud": point_cloud,
    }


@app.post("/send_action")
def receive_action(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Queue action for execution.

    Expected format:
        {
            "action": {
                "type": "run_code",
                "payload": {"code": "get_mobile_target_joint([0, 0, PI])"}
            }
        }
    """
    # Validate action format
    if "action" in payload and "type" in payload["action"] and "payload" in payload["action"]:
        RESULT = process_actions(payload["action"])
        return {"status": "success", "result": RESULT}
    
    return {"status": "error", "message": "Invalid action format"}


def main() -> None:
    """
    Start simulator and FastAPI server.

    Creates three concurrent threads:
        1. Main thread: FastAPI uvicorn server
        2. Simulator thread: MuJoCo physics simulation with 3D viewer
    """
    # Start background threads (daemon=True ensures cleanup on exit)
    threading.Thread(target=run_simulator, daemon=True).start()

    # Display startup information
    print("\n" + "=" * 60)
    print(f"MuJoCo Robot Simulator API")
    print("=" * 60)
    print(f"Server: http://{HOST}:{PORT}")
    print(f"API docs: http://{HOST}:{PORT}/docs")
    print("=" * 60 + "\n")

    # Start FastAPI server (blocking call)
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
| File: 8_camera_vehicle.py
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: This files serves as an example on how to build an app that makes use of the Pegasus API to run a simulation
with a single vehicle equipped with a camera, producing rgb and camera info ROS2 Humble topics.
"""

# Imports to start Isaac Sim from this script
import carb
from omni.isaac.kit import SimulationApp

# Start Isaac Sim's simulation environment
# Note: this simulation app must be instantiated right after the SimulationApp import, otherwise the simulator will crash
# as this is the object that will load all the extensions and load the actual simulator.
simulation_app = SimulationApp({"headless": False, "width": 640, "height": 480})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
from omni.isaac.core.world import World
from omni.isaac.core.utils.extensions import disable_extension, enable_extension

# Enable/disable ROS bridge extensions to keep only ROS2 Humble Bridge
disable_extension("omni.isaac.ros_bridge")
disable_extension("omni.isaac.ros2_bridge")
enable_extension("omni.isaac.ros2_bridge-humble")

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.backends.mavlink_backend import MavlinkBackend, MavlinkBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.graphs import ROS2Camera, ROS2Tf, ROS2Odometry, ROS2Lidar
from pegasus.simulator.logic.sensors import Magnetometer, IMU, Barometer, Vision, Camera, Lidar

# Auxiliary scipy and numpy modules
from scipy.spatial.transform import Rotation

class PegasusApp:
    """
    A Template class that serves as an example on how to build a simple Isaac Sim standalone App.
    """

    def __init__(self):
        """
        Method that initializes the PegasusApp and is used to setup the simulation environment.
        """

        # Acquire the timeline that will be used to start/stop the simulation
        self.timeline = omni.timeline.get_timeline_interface()

        # Start the Pegasus Interface
        self.pg = PegasusInterface()

        # Acquire the World, .i.e, the singleton that controls that is a one stop shop for setting up physics,
        # spawning asset primitives, etc.
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Launch one of the worlds provided by NVIDIA
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

        # Create the vehicle
        # Try to spawn the selected robot in the world to the specified namespace
        config_multirotor = MultirotorConfig()

        # Create the multirotor configuration
        mavlink_config = MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": True,
            "px4_dir": "/home/fstec/Projects/PX4-Autopilot",
            "px4_vehicle_model": 'iris_vision'
        })
        config_multirotor.backends = [MavlinkBackend(mavlink_config)]

        # Sensors
        camera_prim_path = "body/camera"
        camera_config = {
            "position": [0.1, 0.0, 0.0],
            "orientation": Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            "focal_length": 16.0,
            "overwrite_params": True
        }
        lidar_prim_path = "body/lidar"
        lidar_config = {
            "position": [-0.1, 0.0, 0.0],
            "yaw_offset": 180.0,
            "horizontal_fov": 27.0,
            "vertical_fov": 27.0,
            "min_range": 0.01,
            "max_range": 5.0,
            "draw_lines": True
        }
        config_multirotor.sensors = [
            Magnetometer(), IMU(), Barometer(), Vision(), 
            Camera(camera_prim_path, camera_config),
            Lidar(lidar_prim_path, lidar_config)]

        # Graphs
        config_multirotor.graphs = [
            ROS2Tf(), ROS2Odometry(),
            ROS2Camera(camera_prim_path, config={"types": ['rgb', 'camera_info', 'depth']}),
            ROS2Lidar(lidar_prim_path)
        ]

        Multirotor(
            "/World/quadrotor",
            ROBOTS['Iris'],
            0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )

        # Reset the simulation environment so that all articulations (aka robots) are initialized
        self.world.reset()

        # Auxiliar variable for the timeline callback example
        self.stop_sim = False

    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """

        # Start the simulation
        self.timeline.play()

        # The "infinite" loop
        while simulation_app.is_running() and not self.stop_sim:
            # Update the UI of the app and perform the physics step
            self.world.step(render=True)

        # Cleanup and stop
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()

def main():

    # Instantiate the template app
    pg_app = PegasusApp()

    # Run the application loop
    pg_app.run()

if __name__ == "__main__":
    main()

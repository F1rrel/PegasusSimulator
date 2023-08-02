"""
| File: rgb_camera.py
| Author: Micah Nye (micahn@andrew.cmu.edu)
| License: BSD-3-Clause. Copyright (c) 2023, Micah Nye. All rights reserved.
| Description: Simulates an RGB camera using the Omnigraph framework provided in Isaacsim
"""
__all__ = ["Camera"]

import carb

from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.prims import XFormPrim
from omni.isaac.sensor import Camera as CameraPrim

from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.sensors import Sensor
from pegasus.simulator.logic.vehicles import Vehicle
import numpy as np

class Camera(Sensor):
    """The class that implements the Camera sensor. This class inherits the base class Sensor.
    """
    def __init__(self, camera_prim_path: str, config={}):
        """Initialize the Camera class
        Args:
            camera_prim_path (str): Path to the camera prim. Global path when it starts with `/`, else local to vehicle prim path
            config (dict): A Dictionary that contains all the parameters for configuring the Camera - it can be empty or only have some of the parameters used by the Camera.
        Examples:
            The dictionary default parameters are
            >>> {"position": [0.0, 0.0, 0.0],         # Meters
            >>>  "orientation": [0.0, 0.0, 0.0, 1.0], # Quaternion [qx, qy, qz, qw]
            >>>  "focal_length": 24.0,                # Millimeters
            >>>  "focus_distance", 400.0,             # Stage units
            >>>  "resolution": [640, 480],            # Pixels
            >>>  "set_projection_type": "pinhole",    # pinhole, fisheyeOrthographic, fisheyeEquidistant, fisheyeEquisolid, fisheyePolynomial, fisheyeSpherical
            >>>  "update_rate": 30.0}                 # Hz
        """

        # Initialize the Super class "object" attribute
        # update_rate not necessary
        super().__init__(sensor_type="Camera", update_rate=config.get("update_rate", 30))

        # Save the id of the sensor
        self._camera_prim_path = camera_prim_path
        self._frame_id = camera_prim_path.rpartition("/")[-1] # frame_id of the camera is the last prim path part after `/`

        # Reference to the actual camera object. This is set when the camera is initialized
        self.camera = None

        # Set the position of the camera relative to the vehicle
        self._position = np.array(config.get("position", [0.0, 0.0, 0.0]))
        self._orientation = np.array(config.get("orientation", [0.0, 0.0, 0.0, 1.0]))  # Quaternion [qx, qy, qz, qw]

        # Set the camera parameters
        self._focal_length = config.get("focal_length", 24.0)
        self._focus_distance = config.get("focus_distance", 400.0)
        self._clipping_range = config.get("clipping_range", [0.05, 1000000.0])
        self._resolution = config.get("resolution", [640, 480])
        self._set_projection_type = config.get("set_projection_type", "pinhole")
        self._horizonal_aperture = config.get("horizontal_aperture", 20.9550)
        self._vertical_aperture = config.get("vertical_aperture", 15.2908)

        # Save the current state of the camera sensor
        self._state = {
            "frame_id": self._frame_id
        }
    
    def initialize(self, vehicle: Vehicle):
        """Method that initializes the action graph of the camera. It also initalizes the sensor latitude, longitude and
        altitude attributes as well as the vehicle that the sensor is attached to.
        
        Args:
            vehicle (Vehicle): The vehicle that this sensor is attached to.
        """

        # Set the prim paths for the camera
        # - camera (XForm) - ENU-FLU representation
        #   - camera (Camera)
        if self._camera_prim_path[0] != '/':
            camera_xform_path = f"{vehicle.prim_path}/{self._camera_prim_path}"
        else:
            camera_xform_path = self._camera_prim_path
        self._camera_prim_path = f"{camera_xform_path}/{self._frame_id}"

        # Create camera xform prim in ENU frame
        if not is_prim_path_valid(camera_xform_path):
            XFormPrim(
                prim_path=camera_xform_path,
                translation=np.array(self._position),
                orientation=[self._orientation[3], self._orientation[0], self._orientation[1], self._orientation[2]]
            )
        else:
            carb.log_error(f"Cannot create Camera prim, the prim path \"{camera_xform_path}\" already exists")
            return

        # Create the camera prim
        self.camera = CameraPrim(
            prim_path=self._camera_prim_path,
            frequency=self._update_rate,
            resolution=self._resolution,
        )

        # Set camera parameters
        self.camera.set_focal_length(self._focal_length)
        self.camera.set_focus_distance(self._focus_distance)
        self.camera.set_clipping_range(self._clipping_range[0], self._clipping_range[1])
        self.camera.set_projection_type(self._set_projection_type)
        self.camera.set_horizontal_aperture(self._horizonal_aperture)
        self.camera.set_vertical_aperture(self._vertical_aperture)

        # Set the sensor's frame path
        self.frame_path = camera_xform_path

    @property
    def state(self):
        """
        (dict) The 'state' of the sensor, i.e. the data produced by the sensor at any given point in time
        """
        return self._state

    @Sensor.update_at_rate
    def update(self, state: State, dt: float):
        """
        Args:
            state (State): The current state of the vehicle. UNUSED IN THIS SENSOR
            dt (float): The time elapsed between the previous and current function calls (s). UNUSED IN THIS SENSOR
        Returns:
            None
        """

        return None
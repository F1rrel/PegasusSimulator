"""
| File: rep_shm_writer.py
| License: BSD-3-Clause. Copyright (c) 2023, Micah Nye. All rights reserved.
| Description: Creates or connects to a Camera prim for higher level functionality
"""
__all__ = ["RepShmWriter"]

import omni.replicator.core as rep

from pegasus.simulator.logic.vehicles import Vehicle

from pegasus.simulator.logic.replicator.writers import PsShmWriter

class RepShmWriter():
    
    def __init__(self, camera_prim_path: str, config: dict = {}):
        """Initialize the ShmWriter class

        Args:
            camera_prim_path (str): Path to the camera prim. Global path when it starts with `/`, else local to vehicle prim path
            config (dict): A Dictionary that contains all the parameters for configuring the Camera - it can be empty or only have some of the parameters used by the Camera.
        
        Examples:
            The dictionary default parameters are

            >>> {"annotator": "rgb",                         # Annotator to use [LdrColor, rgb]
            >>>  "socket_path": "/tmp/ps_shm.sock",          # Path to the socket
            >>>  "shm_size": 67108864,                       # Size of the shared memory
            >>>  "width": 640,                               # Width of the output stream
            >>>  "height": 480,                              # Height of the output stream
            >>>  "fps": 30}                                  # FPS of the output stream
        """

        # Save the id of the sensor
        self._camera_prim_path = camera_prim_path
        self._annotator = config.get("annotator", "rgb")
        self._socket_path = config.get("socket_path", "/tmp/ps_shm.sock")
        self._shm_size = config.get("shm_size", 67108864)
        self._width = config.get("width", 640)
        self._height = config.get("height", 480)
        self._fps = config.get("fps", 30)
    
    def initialize(self, vehicle: Vehicle):
        """Initialize the ShmWriter class

        Args:
            vehicle (Vehicle): Vehicle object to attach the RTSPWriter to
        """
        self._vehicle = vehicle

        # Set the prim path for the camera
        if self._camera_prim_path[0] != "/":
            self._camera_prim_path = f"{vehicle.prim_path}/{self._camera_prim_path}"
        else:
            self._camera_prim_path = self._camera_prim_path

        # Append camera render product
        render_products = []
        render_products.append(
            rep.create.render_product(
                self._camera_prim_path,
                resolution=(self._width, self._height)))

        writer_config = {
            "annotator": self._annotator,
            "socket_path": self._socket_path,
            "shm_size": self._shm_size,
            "fps": self._fps
        }

        # Create Shm writer
        self._writer = PsShmWriter(
            writer_config
        )
    
        self._writer.attach(render_products)

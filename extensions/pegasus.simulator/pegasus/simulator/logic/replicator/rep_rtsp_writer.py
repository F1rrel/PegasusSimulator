"""
| File: rep_rtsp_writer.py
| License: BSD-3-Clause. Copyright (c) 2023, Micah Nye. All rights reserved.
| Description: Creates or connects to a Camera prim for higher level functionality
"""
__all__ = ["RepRTSPWriter"]

import omni.replicator.core as rep
from omni.isaac.core.utils.extensions import enable_extension

from pegasus.simulator.logic.vehicles import Vehicle

# enable_extension("omni.replicator.agent.core")
# import omni.replicator.agent.core.writers

from pegasus.simulator.logic.replicator.writers import PsRTSPWriter

class RepRTSPWriter():
    
    def __init__(self, camera_prim_path: str, config: dict = {}):
        """Initialize the RTSPWriter class

        Args:
            camera_prim_path (str): Path to the camera prim. Global path when it starts with `/`, else local to vehicle prim path
            config (dict): A Dictionary that contains all the parameters for configuring the Camera - it can be empty or only have some of the parameters used by the Camera.
        
        Examples:
            The dictionary default parameters are

            >>> {"annotator": "rgb",                         # Annotator to use [LdrColor, rgb]
            >>>  "server": "localhost",                      # RTSP server to connect to
            >>>  "port": 8554,                               # Port to connect to
            >>>  "topic": "Pegasus",                         # Topic to connect to
            >>>  "width": 640,                               # Width of the output stream
            >>>  "height": 480,                              # Height of the output stream
            >>>  "fps": 30.0,                                # FPS of the output stream
            >>>  "bitrate": 2000,                            # Bitrate of the output stream [Kbps]
            >>>  "enc_profile": "fast",                      # Encoding profile to use [fast, slow]
            >>>  "device": 0}                                # GPU to use for rendering
        """

        # Save the id of the sensor
        self._camera_prim_path = camera_prim_path
        self._annotator = config.get("annotator", "rgb")
        self._server = config.get("server", "localhost")
        self._port = config.get("port", 8554)
        self._topic = config.get("topic", "Pegasus")
        self._width = config.get("width", 640)
        self._height = config.get("height", 480)
        self._fps = config.get("fps", 30.0)
        self._bitrate = config.get("bitrate", 2000)
        self._enc_profile = config.get("enc_profile", "fast")
        self._device = config.get("device", 0)
    
    def initialize(self, vehicle: Vehicle):
        """Initialize the RTSPWriter class

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
            "server": self._server,
            "port": self._port,
            "topic": self._topic,
        }

        camera_config = {
            "device": self._device,
            "fps": self._fps,
            "bitrate": self._bitrate,
            "enc_profile": self._enc_profile,
        }

        # Create RTSP writer
        self._writer = PsRTSPWriter(
            writer_config, camera_config
        )
    
        self._writer.attach(render_products)

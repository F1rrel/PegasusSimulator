"""
| File: ps_rtsp_writer.py
| License: BSD-3-Clause. Copyright (c) 2023, Micah Nye. All rights reserved.
| Description: Replicator RTSP Writer optimized for Pegasus Simulator
"""

import subprocess
import os
import time

import carb  # carb logging

import omni.timeline
import omni.replicator.core as rep
from omni.replicator.core import AnnotatorRegistry, Writer

# https://gitlab-master.nvidia.com/omniverse/synthetic-data/omni.replicator/-/blob/develop/source/extensions/omni.replicator.core/python/scripts/annotators_default.py#L9
_annotator_type_uint8_channel_4 = [
    "LdrColor",
    "rgb",
]

_supported_annotators = _annotator_type_uint8_channel_4

class PsShmWriter(Writer):
    """Publish annotations of attached render products to shared memory."""

    _default_annotator = "LdrColor"
    _default_socket_path = "/tmp/ps_shm.sock"
    _default_shm_size = 67108864 # 64MB
    _default_fps = 30

    _name = "PsShmWriter"

    def __init__(self, config: dict = {}):
        self.initialize(config)

    def __del__(self):
        self.on_final_frame()
    
    def initialize(self, config: dict = {}):
        """Initialize the PsShmWriter class

        Args:
            config (dict): A Dictionary that contains all the parameters for configuring the PsShmWriter.
        
        Example:
            The dictionary default parameters are
            >>> {"annotator": "rgb",                         # Annotator to use [LdrColor, rgb]
            >>>  "socket_path": "/tmp/ps_shm.sock",          # Path to the socket
            >>>  "shm_size": 67108864,                       # Size of the shared memory
            >>>  "fps": 30.0}                                # FPS of the output stream
        """

        self.annotator = config.get("annotator", self._default_annotator)
        self.socket_path = config.get("socket_path", self._default_socket_path)
        self.shm_size = config.get("shm_size", self._default_shm_size)

        self.fps = config.get("fps", self._default_fps)

        # Create annotators
        self._annotators = []
        if self.annotator in ["semantic_segmentation", "instance_id_segmentation", "instance_segmentation"]:
            self.annotators.append(AnnotatorRegistry.get_annotator(self.annotator, init_params={"colorize": True}))
        else:
            self.annotators.append(AnnotatorRegistry.get_annotator(self.annotator))

        self._process = None
        self._frame_id = 0
    
    @property
    def annotator(self):
        return self._annotator

    @annotator.setter
    def annotator(self, annotator: str):
        if not isinstance(annotator, str) or annotator not in _supported_annotators:
            self._annotator = self._default_annotator
        else:
            self._annotator = annotator
    
    @property
    def socket_path(self):
        return self._socket_path

    @property
    def shm_size(self):
        return self._shm_size

    @property
    def fps(self):
        return self._fps

    def on_final_frame(self):
        super().on_final_frame()

        if self._process is not None:
            try:
                self._process.stdin.close()
            except Exception:
                pass
            self._process.terminate()
            self._process.wait()
            self._process = None
        
        if os.path.exists(self.socket_path):
            try:
                os.remove(self.socket_path)
            except Exception as e:
                carb.log_error(f"[{self._name}] Failed to remove socket file {self.socket_path}: {e}")
        
        self._frame_id = 0

    def write(self, data):
        """Write the data to the socket."""
        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing():
            return

        if self._frame_id == 0:
            self._launch_gstreamer_pipeline(data)
            self._time_start = time.time()

        # Convert RGBA to RGB
        rgb = data[self.annotator][:, :, :3]

        try:
            self._process.stdin.write(rgb.tobytes())
            self._process.stdin.flush()
        except BrokenPipeError:
            carb.log_error(f"[{self._name}] Broken pipe error: {self._process.stdin}")
        
        self._frame_id += 1
        if self._frame_id % self.fps == 0:
            elapsed_time = time.time() - self._time_start
            carb.log_verbose(f"[{self._name}] Frame {self._frame_id} written to socket {self.socket_path}")
            carb.log_verbose(f"[{self._name}] FPS: {(self._frame_id - 1) / elapsed_time:.2f}")

    def _launch_gstreamer_pipeline(self, data):
        if self._process is not None:
            self._process.stdin.close()
            self._process.terminate()
            self._process.wait()
            self._process = None
        
        render_products = [k for k in data.keys() if k.startswith("rp_")]
        if not isinstance(render_products, list):
            render_products = [render_products]
        
        for rp_path in render_products:
            camera_prim_path = data[rp_path]["camera"]
            (width, height) = data[rp_path]["resolution"]

            gst_cmd = [
                "gst-launch-1.0",
                "fdsrc", "fd=0",
                "!", f"rawvideoparse",
                f"width={width}",
                f"height={height}",
                "format=rgb",
                f"framerate={self.fps}/1",
                "!", "videoconvert",
                "!", "shmsink",
                f"socket-path={self.socket_path}",
                "sync=false",
                "wait-for-connection=true",
                f"shm-size={self.shm_size}",
            ]

            carb.log_info(f"[{self._name}] Launching GStreamer pipeline: {' '.join(gst_cmd)}")

            process = subprocess.Popen(
                gst_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=False,
                bufsize=0
            )

            if process is None:
                raise Exception(f"[{self._name}] Cannot start Shared Memory Writer on {camera_prim_path}.")
            
            self._process = process
            carb.log_info(f"[{self._name}] Started GStreamer pipeline: {process.pid}")

            receive_pipeline = [
                "gst-launch-1.0",
                "shmsrc",
                f"socket-path={self.socket_path}",
                "do-timestamp=true",
                "is-live=true",
                "!", f"video/x-raw,format=RGB,width={width},height={height},framerate={self.fps}/1",
                "!", "videoconvert",
                "!", "autovideosink",
            ]
            carb.log_info(f"[{self._name}] Sample receive pipeline: {' '.join(receive_pipeline)}")

rep.WriterRegistry.register(PsShmWriter)

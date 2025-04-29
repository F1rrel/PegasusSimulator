"""
| File: ps_rtsp_writer.py
| License: BSD-3-Clause. Copyright (c) 2023, Micah Nye. All rights reserved.
| Description: Replicator RTSP Writer optimized for Pegasus Simulator
"""

import re
import socket
import subprocess as sp
import sys
from shutil import which

import carb  # carb logging

import omni.timeline
import omni.replicator.core as rep
from omni.replicator.core import AnnotatorRegistry, Writer


# https://gitlab-master.nvidia.com/omniverse/synthetic-data/omni.replicator/-/blob/develop/source/extensions/omni.replicator.core/python/scripts/annotators_default.py#L9
_annotator_type_uint8_channel_4 = [
    "LdrColor",
    "rgb",
    "semantic_segmentation",  # colorize = True
    "instance_id_segmentation",  # colorize = True
    "instance_segmentation",  # colorize = True
    "DiffuseAlbedo",  # ??? pixel format is wrong ???
    "Roughness",
]

_annotator_type_float16_channel_1 = ["EmissionAndForegroundMask"]

_annotator_type_float32_channel_1 = [
    "distance_to_camera",
    "distance_to_image_plane",
    "DepthLinearized",  # ??? pixel format is wrong ???
]

_annotator_type_float16_channel_4 = [
    "HdrColor",
]

_nvenc_annotators = _annotator_type_uint8_channel_4
_software_annotators = (
    _annotator_type_float16_channel_1 + _annotator_type_float32_channel_1 + _annotator_type_float16_channel_4
)
_supported_annotators = _nvenc_annotators + _software_annotators

# Default values
_default_device = 0
_default_annotator = "LdrColor"

class RTSPCamera:
    """The class records a render products (HydraTexture) by its prim path.
    The class also records the ffmpeg subprocess command which is customized
    by the render product's camera parameters, e.g. fps, width, height, annotator.
    The published RTSP URL of each RTSPCamera instance is constructed by appending
    the render product's camera prim path and the annotator name to the base output directory.

    Notes:
        The supported annotators are:
            'LdrColor' / 'rgb',
            'semantic_segmentation',
            'instance_id_segmentation',
            'instance_segmentation',
            'DiffuseAlbedo',
            'Roughness',
            'EmissionAndForegroundMask'
            'distance_to_camera',
            'distance_to_image_plane',
            'DepthLinearized',
            'HdrColor'
        Please refer to https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html
        for more details on the supported annotators. Annotators: 'LdrColor' / 'rgb', 'semantic_segmentation',
        'instance_id_segmentation', 'instance_segmentation', are accelerated by NVENC while the rest annotators
        are software encoded by CPU. The default video stream format is HEVC.
    """

    _default_device = _default_device
    _default_fps = 30.0
    _default_bitrate = 2000
    _default_enc_profile = "fast"
    
    _name = "RTSPCamera"

    def __init__(self, camera_prim_path: str, width: int, height: int, annotator: str, output_dir: str, config: dict = {}):
        """Initialize the RTSPCamera class

        Args:
            camera_prim_path (str): Path to the camera prim. Global path when it starts with `/`, else local to vehicle prim path
            config (dict): A Dictionary that contains all the parameters for configuring the Camera - it can be empty or only have some of the parameters used by the Camera.
        """

        self.initialize(camera_prim_path, width, height, annotator, output_dir, config)

    def initialize(self, camera_prim_path: str, width: int, height: int, annotator: str, output_dir: str, config: dict = {}):
        """Initialize the RTSPCamera class

        Args:
            camera_prim_path (str): Path to the camera prim. Global path when it starts with `/`, else local to vehicle prim path
            config (dict): A Dictionary that contains all the parameters for configuring the Camera - it can be empty or only have some of the parameters used by the Camera.
        
        Example:
            The dictionary default parameters are
            >>> {"device": 0,                                # GPU to use for rendering
            >>>  "fps": 30.0,                                # FPS of the output stream
            >>>  "bitrate": 2000,                            # Bitrate of the output stream [Kbps]
            >>>  "enc_profile": "fast"}                      # Encoding profile
        """

        self.camera_prim_path = camera_prim_path
        self.width = width
        self.height = height
        self.annotator = annotator
        self.output_dir = output_dir + "_" + annotator

        self.device = config.get("device", self._default_device)
        self.fps = config.get("fps", self._default_fps)
        self.bitrate = config.get("bitrate", self._default_bitrate)
        self.enc_profile = config.get("enc_profile", self._default_enc_profile)

        if self.annotator in _annotator_type_uint8_channel_4:
            self.pixel_fmt = "rgba"
        elif self.annotator in _annotator_type_float16_channel_1:
            if sys.byteorder == "little":
                self.pixel_fmt = "gray16le"
            else:
                self.pixel_fmt = "gray16be"
        elif self.annotator in _annotator_type_float32_channel_1:
            if sys.byteorder == "little":
                self.pixel_fmt = "grayf32le"
            else:
                self.pixel_fmt = "grayf32be"
        elif self.annotator in _annotator_type_float16_channel_4:
            if sys.byteorder == "little":
                self.pixel_fmt = "rgba64le"
            else:
                self.pixel_fmt = "rgba64be"
        else:
            raise ValueError(f"Publishing {self.annotator} annotator is not supported.")
        
        # Verify ffmpeg installed
        if not which("ffmpeg"):
            raise ValueError(f"ffmpeg cannot be found in the system.")

        # Video writer
        self.command = ["ffmpeg"]

        # Setup encoder based on annotator
        if self.annotator in _nvenc_annotators:
            self.command += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-hwaccel_device", str(self.device)]
            vcodec = "hevc_nvenc"  # 'h264_nvenc'
        elif self.annotator in _software_annotators:  # Software encoding
            vcodec = "libx265"  # 'libx264'
        else:
            raise ValueError(f"Publishing {self.annotator} annotator is not supported.")

        # Finish the command
        self.command += [
            "-use_wallclock_as_timestamps",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            self.pixel_fmt,
            "-s",
            f"{self.width}x{self.height}",
            "-i",
            "-",
            "-c:a",
            "copy",
            "-c:v",
            vcodec,
            "-preset",
            self.enc_profile,
            "-maxrate:v",
            f"{self.bitrate}k",
            "-bufsize:v",
            "64M",
            "-vsync",
            "1",
            "-r",
            str(self.fps),
            "-f",
            "rtsp",
            "-rtsp_transport",
            "udp",
            self.output_dir
        ]

        carb.log_info(f'[{self._name}] command = {self.command}')

        carb.log_info(
            f'"{self.annotator}" of "{self.camera_prim_path}" will be published to "{output_dir}" encoded by "{vcodec}".'
        )
        
        self.pipe = None

    @property
    def device(self) -> int:
        return self._device

    @device.setter
    def device(self, device: int):
        if not isinstance(device, int) or device < 0:
            self._device = self._default_device
        else:
            self._device = device

    @property
    def fps(self) -> float:
        return self._fps

    @fps.setter
    def fps(self, fps: float):
        if not isinstance(fps, (int, float)) or fps <= 0:
            self._fps = self._default_fps
        else:
            self._fps = fps

    @property
    def bitrate(self) -> int:
        return self._bitrate

    @bitrate.setter
    def bitrate(self, bitrate: int):
        if not isinstance(bitrate, int) or bitrate < 0:
            self._bitrate = self._default_bitrate
        else:
            self._bitrate = bitrate

class PsRTSPWriter(Writer):
    """Publish annotations of attached render products to an RTSP server.

    The Writer tracks a dictionary of render products (HydraTexture) by the combo of the
    annotator name and the render product's prim path. Each render product is recorded as
    an instance of RTSPCamera. The published RTSP URL of each RTSPCamera instance is
    constructed by appending the render product's camera prim path and the annotator name
    to the base output directory.

    The supported annotators are:
        'LdrColor' / 'rgb',
        'semantic_segmentation',
        'instance_id_segmentation',
        'instance_segmentation',
        'DiffuseAlbedo',
        'Roughness',
        'EmissionAndForegroundMask'
        'distance_to_camera',
        'distance_to_image_plane',
        'DepthLinearized',
        'HdrColor'
    Please refer to https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/annotators_details.html
    for more details on the supported annotators. Annotators: 'LdrColor' / 'rgb', 'semantic_segmentation',
    'instance_id_segmentation', 'instance_segmentation', are accelerated by NVENC while the rest annotators
    are software encoded by CPU. The video stream format is HEVC.
    """

    # Default RTSP server, port, and topic.
    _default_annotator = _default_annotator
    _default_server = "localhost"
    _default_port = 8554
    _default_topic = "PsRTSPWriter"

    _name = "PsRTSPWriter"

    def __init__(self, config: dict = {}, camera_config: dict = {}):
        """Initialize the writer."""
        self.initialize(config, camera_config)

    def initialize(self, config: dict = {}, camera_config: dict = {}):
        """Initialize the writer.

        Args:
            config (dict): A Dictionary that contains all the parameters for configuring the PsRTSPWriter.
            camera_config (dict): A Dictionary that contains all the parameters for configuring the RTSPCamera.

        Example config:
            The dictionary default parameters are
            >>>  {"annotator": "LdrColor",                   # Annotator to use [LdrColor, rgb]
            >>>  "server": "localhost",                      # RTSP server
            >>>  "port": 8554,                               # RTSP port
            >>>  "topic": "PsRTSPWriter"}                    # RTSP topic
        
        Example camera_config:
            The dictionary default parameters are
            >>> {"device": 0,                                # GPU to use for rendering
            >>>  "fps": 30.0,                                # FPS of the output stream
            >>>  "bitrate": 2000,                            # Bitrate of the output stream [Kbps]
            >>>  "enc_profile": "fast"}                      # Encoding profile
        """

        # Save the id of the sensor
        self.annotator = config.get("annotator", self._default_annotator)
        self.server = config.get("server", self._default_server)
        self.port = config.get("port", self._default_port)
        self.topic = config.get("topic", self._default_topic)

        self.camera_config = camera_config

        self._output_dir = f"rtsp://{self.server}:{self.port}/{self.topic}"

        # Create annotators
        self._annotators = []
        if self.annotator in ["semantic_segmentation", "instance_id_segmentation", "instance_segmentation"]:
            self.annotators.append(AnnotatorRegistry.get_annotator(self.annotator, init_params={"colorize": True}))
        else:
            self.annotators.append(AnnotatorRegistry.get_annotator(self.annotator))
        
        print("Annotators: ", [x.get_name() for x in self.annotators])

        # Verify RTSP
        self.verify_rtsp_server()

        self.cameras = {}  # map render_product name to a subprocess
        self._frame_id = 0

    @property
    def annotator(self):
        return self._annotator

    @annotator.setter
    def annotator(self, annotator: str):
        if not isinstance(annotator, str) or annotator not in _supported_annotators:
            self._annotator = self._default_annotator
        elif annotator == "rgb":
            self._annotator = "LdrColor"
        else:
            self._annotator = annotator

    @property
    def output_dir(self):
        return self._output_dir
    
    def verify_rtsp_server(self):
        # Verify RTSP URL
        match = re.search(r"^rtsp\://(.+)\:([0-9]+)/(.+)$", self.output_dir)
        if not match:
            raise ValueError(
                f'{self.output_dir} is not a valid RTSP stream URL. The format is "rtsp://<hostname>:<port>/<topic>".'
            )
        
        # Verify RTSP server is live
        sock = socket.socket()
        try:
            sock.connect((self.server, int(self.port)))
        except Exception as e:
            raise ValueError(f"RTSP server at {self.server}:{self.port} is not accessible with exception {e}.")
        finally:
            sock.close()

    def on_final_frame(self):
        """Run after final frame is written.

        Notes:
            When "stop" button is clicked in Isaac Sim UI, this function is called.
            The ffmpeg subprocesses are killed (SIGKILL).
        """
        super().on_final_frame()

        for _, camera in self.cameras.items():
            if camera.pipe:
                camera.pipe.stdin.close()
                camera.pipe.wait()
                camera.pipe.kill()
                camera.pipe = None
                carb.log_info(f'Subprocess on "{camera.prim_path}" has been terminated.')

        self.cameras.clear()
        self._frame_id = 0

    def write(self, data):
        """Write function called from the OgnWriter node on every frame to process annotator output.

        The function publishes one of _supported_annotators of each attached render product
        to the RTSP server. If the render product name does not exist in self.pipes (dict),
        the function creates a new ffmpeg subprocess.

        Args:
            data: A dictionary containing the annotator data for the current frame.

        Returns:
            None

        Raises:
            Exception: Can't open RTSP client writer.
        """

        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing():
            return
    
        # Create RTSPCamera's on the first frame.
        if self._frame_id == 0:
            self._config_cameras(data)

        for key, camera in self.cameras.items():
            if (
                key.startswith("semantic_segmentation")
                or key.startswith("instance_id_segmentation")
                or key.startswith("instance_segmentation")
            ):
                camera.pipe.stdin.write(data[key]["data"].tobytes())
            else:
                camera.pipe.stdin.write(data[key].tobytes())

        self._frame_id += 1
    
    def _config_cameras(self, data):
        """Attach one or a list of render products to the writer.

        This is the base function called by either attach() or attach_async().
        The function constructs the ffmpeg command for each render product.
        Each render product associates with an unique RTSP URL which is built
        by appending render product's camera prim path to self.output_dir.

        Args:
            data: A dictionary containing the annotator data for the current frame.

        Returns:
            None
        """

        # Clear existing camera pipes
        if self.cameras:
            self.on_final_frame()

        annotator_name = self.annotators[0].get_name()
        print(f"Annotator: {annotator_name}")

        render_products = [k for k in data.keys() if k.startswith("rp_")]
        if not isinstance(render_products, list):
            render_products = [render_products]
        
        for rp_path in render_products:
            camera_prim_path = data[rp_path]["camera"]
            resolution = data[rp_path]["resolution"]

            if len(render_products) == 1:
                key = annotator_name
            else:
                key = annotator_name + "-" + rp_path[3:]

            camera = RTSPCamera(
                camera_prim_path=camera_prim_path,
                width=resolution[0],
                height=resolution[1],
                annotator=annotator_name,
                output_dir=self.output_dir,
                config=self.camera_config
            )

            camera.pipe = sp.Popen(camera.command, stdin=sp.PIPE)
            if not camera.pipe:
                raise Exception(f"Can't start ffmpeg RTSP client writer on {camera.prim_path}.")

            self.cameras[key] = camera

rep.WriterRegistry.register(PsRTSPWriter)

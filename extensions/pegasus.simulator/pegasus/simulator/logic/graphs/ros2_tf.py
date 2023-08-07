"""
| File: ros2_tf.py
| License: BSD-3-Clause. Copyright (c) 2023, Micah Nye. All rights reserved.
"""
__all__ = ["ROS2Tf"]

import carb

from omni.isaac.core.utils import stage
import omni.graph.core as og
from omni.isaac.core.utils.prims import is_prim_path_valid, set_targets
from omni.isaac.core.prims import XFormPrim

from pegasus.simulator.logic.graphs import Graph
from pegasus.simulator.logic.vehicles import Vehicle

class ROS2Tf(Graph):
    """The class that implements the ROS2 TF graph. This class inherits the base class Graph.
    """
    def __init__(self):
        """Initialize the ROS2 TF class
        """

        # Initialize the Super class "object" attribute
        super().__init__(graph_type="ROS2Tf")

    def initialize(self, vehicle: Vehicle):
        """Method that initializes the graph.

        Args:
            vehicle (Vehicle): The vehicle that this graph is attached to.
        """

        self._namespace = f"/{vehicle.vehicle_name}"

        # The vehicle uses body instead of standardized base_link,
        # so we need to create the base_link and connect the body to it
        base_link_xform_path = f"{vehicle.prim_path}/body/base_link"
        XFormPrim(
            prim_path=base_link_xform_path
        )

        # Create the graph under vehicle with graph name tf and allow only one per vehicle.
        graph_path = f"{vehicle.prim_path}/tf_pub"
        if is_prim_path_valid(graph_path):
            carb.log_warn(f"ROS2 TF Graph for vehicle {vehicle.vehicle_name} already exists")
            return
    
        # Graph configuration
        graph_specs = {
            "graph_path": graph_path,
            "evaluator_name": "execution",
        }

        # Creating a graph edit configuration with transform tree publishers
        keys = og.Controller.Keys
        graph_config = {
            keys.CREATE_NODES: [
                ("on_playback_tick", "omni.graph.action.OnPlaybackTick"),
                ("isaac_read_simulation_time", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                ("publish_transform_tree", "omni.isaac.ros2_bridge.ROS2PublishTransformTree")
            ],
            keys.CONNECT: [
                ("on_playback_tick.outputs:tick", "publish_transform_tree.inputs:execIn"),
                ("isaac_read_simulation_time.outputs:simulationTime", "publish_transform_tree.inputs:timeStamp")
            ],
            keys.SET_VALUES: [
                ("publish_transform_tree.inputs:nodeNamespace", self._namespace)
            ]
        }

        # Create the camera graph
        (graph, _, _, _) = og.Controller.edit(
            graph_specs,
            graph_config
        )

        # Set the parent frame, it should be the base_link
        set_targets(
            prim=stage.get_current_stage().GetPrimAtPath(f"{graph_path}/publish_transform_tree"),
            attribute="inputs:parentPrim",
            target_prim_paths=[base_link_xform_path]
        )

        # Create list of target prims, which will contain articulation root 
        # and all sensors with frame_path filled
        target_prim_paths = [vehicle.prim_path]

        for sensor in vehicle._sensors:
            if len(sensor.frame_path) and is_prim_path_valid(sensor.frame_path):
                target_prim_paths.append(sensor.frame_path)

        set_targets(
            prim=stage.get_current_stage().GetPrimAtPath(f"{graph_path}/publish_transform_tree"),
            attribute="inputs:targetPrims",
            target_prim_paths=target_prim_paths
        )

        # Run the ROS Camera graph once to generate ROS image publishers in SDGPipeline
        og.Controller.evaluate_sync(graph)

        # Also initialize the Super class with updated prim path (only camera graph path)
        super().initialize(graph_path)

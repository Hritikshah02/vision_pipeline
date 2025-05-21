import json
import os
import re
import logging
from typing import List, Dict, Any, Union, Optional
from datetime import datetime


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline_generator.log")
    ]
)
logger = logging.getLogger("vision_pipeline")

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import BitsAndBytesConfig  # For quantization if needed
    LLM_AVAILABLE = True
    logger.info("PyTorch and Transformers libraries successfully imported")
except ImportError as e:
    logger.warning(f"Failed to import LLM dependencies: {e}")
    LLM_AVAILABLE = False

# Predefined mappings (for fallback and reference)
PROMPT_TO_PIPELINE = {
    "detect objects": [
        {"module": "obj_detection", "mode": "live", "input_from": "camera", "input_type": "camera"}
    ],
    "segment everything after detecting objects": [
        {"module": "obj_detection", "mode": "live", "input_from": "camera", "input_type": "camera"},
        {"module": "nano_seg_track", "mode": "live", "input_from": "obj_detection", "input_type": "BBOX"}
    ],
    "find cats and plan how to grasp them": [
        {"module": "gdino", "mode": "live", "input_from": "camera", "input_type": "camera"},
        {"module": "neugraspnet", "mode": "single", "input_from": "gdino", "input_type": "BBOX"}
    ],
    "recognize text and track human pose": [
        {"module": "human_pose", "mode": "live", "input_from": "camera", "input_type": "camera"},
        {"module": "nano_seg_track", "mode": "live", "input_from": "human_pose", "input_type": "POINTS"},
        {"module": "scene_seg_track", "mode": "live", "input_from": "camera", "input_type": "camera"},
        {"module": "ocr", "mode": "single", "input_from": "camera", "input_type": "camera"}
    ],
    "estimate the depth of objects in the scene": [
        {"module": "obj_detection", "mode": "live", "input_from": "camera", "input_type": "camera"},
        {"module": "scene_seg_track", "mode": "live", "input_from": "camera", "input_type": "camera"},
        {"module": "monocular_depth", "mode": "single", "input_from": "camera", "input_type": "camera"}
    ],
    "track hand movements and identify gestures": [
        {"module": "obj_detection", "mode": "live", "input_from": "camera", "input_type": "camera"},
        {"module": "hand_pose", "mode": "live", "input_from": "camera", "input_type": "camera"},
        {"module": "nano_seg_track", "mode": "live", "input_from": "obj_detection", "input_type": "BBOX"},
        {"module": "scene_seg_track", "mode": "live", "input_from": "camera", "input_type": "camera"}
    ],
    "detect oriented objects in the image": [
        {"module": "obj_detection", "mode": "live", "input_from": "camera", "input_type": "camera"},
        {"module": "oriented_obj_detection", "mode": "live", "input_from": "camera", "input_type": "camera"}
    ],
    "describe what you see in the image": [
        {"module": "llava", "mode": "single", "prompt": "Describe what you see in this image", "input_from": "camera", "input_type": "camera"}
    ],
    "find books and read the text on them": [
        {"module": "obj_detection", "mode": "live", "input_from": "camera", "input_type": "camera"},
        {"module": "ocr", "mode": "single", "input_from": "camera", "input_type": "camera"}
    ],
    "segment objects in the scene": [
        {"module": "obj_detection", "mode": "live", "input_from": "camera", "input_type": "camera"},
        {"module": "obj_segmentation", "mode": "single", "input_from": "obj_detection", "input_type": "BBOX"}
    ],
    "analyze object affordances": [
        {"module": "obj_detection", "mode": "live", "input_from": "camera", "input_type": "camera"},
        {"module": "obj_affordance", "mode": "single", "input_from": "obj_detection", "input_type": "BBOX"}
    ],
    "estimate 6dof poses of objects": [
        {"module": "obj_detection", "mode": "live", "input_from": "camera", "input_type": "camera"},
        {"module": "nano_seg_track", "mode": "live", "input_from": "obj_detection", "input_type": "BBOX"},
        {"module": "obj_pose_6dof", "mode": "live", "input_from": "nano_seg_track", "input_type": "MASK"}
    ],
    # Add the complex example directly from few-shot examples
    "detect an object from my prompt then estimate grasp and physical properties for that object once then estimate live pose for that object continuously": [
        {"module": "gdino", "mode": "live", "input_from": "camera", "input_type": "camera", "level": 0},
        {"module": "neugraspnet", "mode": "single", "input_from": "gdino", "input_type": "BBOX", "level": 1},
        {"module": "llava", "mode": "single", "input_from": "gdino", "input_type": "BBOX", 
         "prompt": "Describe the physical properties of the detected object", "level": 1},
        {"module": "obj_pose_6dof", "mode": "live", "input_from": "gdino", "input_type": "BBOX", "level": 1}
    ]
}

# Module configuration information - Updated with all modules from the table
MODULES_INFO = {
    "obj_detection": {
        "name": "Object Detection",
        "model": "Yolo",
        "mode": "Color",
        "inference_type": "Live",
        "input": "none",
        "outputs": "BBOX",
        "keywords": ["detect", "object", "find", "identify"]
    },
    "oriented_obj_detection": {
        "name": "Oriented Object Detection",
        "model": "Yolo",
        "mode": "Color",
        "inference_type": "Live",
        "input": "none",
        "outputs": "OBBOX",
        "keywords": ["oriented", "orientation", "angle"]
    },
    "gdino": {
        "name": "Zero-shot Object Detection",
        "model": "Grounding Dino",
        "mode": "Color",
        "inference_type": "Single Inference",
        "input": "Prompt",
        "outputs": "BBOX",
        "keywords": ["specific", "particular", "custom"]
    },
    "vit_detection": {
        "name": "ViT-based Object Detection",
        "model": "Owl",
        "mode": "Color",
        "inference_type": "Single Inference",
        "input": "Prompt",
        "outputs": "BBOX",
        "keywords": ["vit", "transformer"]
    },
    "llava": {
        "name": "Vision Language Model",
        "model": "LLaVa",
        "mode": "Color",
        "inference_type": "Single Inference",
        "input": "Prompt",
        "outputs": "Text",
        "keywords": ["describe", "explain", "caption", "tell", "physical"]
    },
    "hand_pose": {
        "name": "Hand Pose Tracking",
        "model": "Mediapipe",
        "mode": "Color",
        "inference_type": "Live",
        "input": "none",
        "outputs": "POINTS",
        "keywords": ["hand", "gesture", "finger"]
    },
    "human_pose": {
        "name": "Human Pose Tracking",
        "model": "Mediapipe",
        "mode": "Color",
        "inference_type": "Live",
        "input": "none",
        "outputs": "POINTS",
        "keywords": ["pose", "human", "body", "person"]
    },
    "obj_segmentation": {
        "name": "Object Segmentation",
        "model": "NanoSam",
        "mode": "Color",
        "inference_type": "Single Inference",
        "input": "BBOX",
        "outputs": "MASK",
        "keywords": ["segment", "segmentation", "mask", "object"]
    },
    "scene_segmentation": {
        "name": "Scene Segmentation",
        "model": "NanoSam",
        "mode": "Color",
        "inference_type": "Single Inference",
        "input": "none",
        "outputs": "MASK",
        "keywords": ["scene", "background", "environment", "segmentation"]
    },
    "agnostic_obj_segmentation": {
        "name": "Image Agnostic Object Segmentation",
        "model": "dounseep",
        "mode": "Color",
        "inference_type": "Single Inference",
        "input": "Object Gallery",
        "outputs": "MASK",
        "keywords": ["agnostic", "segmentation", "gallery"]
    },
    "nano_seg_track": {
        "name": "Object Segmentation & Tracking",
        "model": "NanoSam",
        "mode": "Color",
        "inference_type": "Live",
        "input": "BBOX",
        "outputs": "MASK",
        "keywords": ["segment", "segmentation", "mask", "tracking"]
    },
    "scene_seg_track": {
        "name": "Scene Segmentation & Tracking",
        "model": "Yolo",
        "mode": "Color",
        "inference_type": "Live",
        "input": "none",
        "outputs": "MASK,ID",
        "keywords": ["scene", "background", "environment", "tracking"]
    },
    "ocr": {
        "name": "OCR",
        "model": "Paddle",
        "mode": "Color",
        "inference_type": "Single Inference",
        "input": "none",
        "outputs": "BBOX,Text",
        "keywords": ["text", "ocr", "read", "character"]
    },
    "monocular_depth": {
        "name": "Monocular Depth Estimation",
        "model": "Midas",
        "mode": "Color",
        "inference_type": "Single Inference",
        "input": "none",
        "outputs": "Depth",
        "keywords": ["depth", "distance", "3d"]
    },
    "obj_affordance": {
        "name": "Object Affordance",
        "model": "Custom",
        "mode": "Color",
        "inference_type": "Single Inference",
        "input": "none",
        "outputs": "MASK,Class",
        "keywords": ["affordance", "interaction", "usage"]
    },
    "obj_pose_6dof": {
        "name": "6-DoF Object Pose Estimation",
        "model": "Free Pose",
        "mode": "Depth",
        "inference_type": "Live",
        "input": "MASK",
        "outputs": "4x4 Matrix",
        "keywords": ["pose", "6dof", "orientation", "position"]
    },
    "neugraspnet": {
        "name": "Grasp Planning",
        "model": "NeuGraspnet",
        "mode": "Depth",
        "inference_type": "Single Inference",
        "input": "MASK",
        "outputs": "4x4 Matrices",
        "keywords": ["grasp", "grip", "hold", "pick"]
    }
}

# Define task combinations for rule-based decisions
TASK_COMBINATIONS = {
    "object_detection": ["obj_detection", "oriented_obj_detection", "gdino", "vit_detection"],
    "segmentation": ["obj_detection", "oriented_obj_detection", "nano_seg_track", "scene_seg_track", "obj_segmentation", "scene_segmentation", "agnostic_obj_segmentation"],
    "reading_text": ["obj_detection", "ocr"],
    "hand_tracking": ["obj_detection", "hand_pose", "nano_seg_track", "scene_seg_track"],
    "depth_analysis": ["obj_detection", "scene_seg_track", "monocular_depth"],
    "human_tracking": ["human_pose", "nano_seg_track", "scene_seg_track"],
    "grasp_planning": ["obj_detection", "neugraspnet"],
    "affordance_analysis": ["obj_detection", "obj_affordance"],
    "pose_estimation": ["obj_detection", "obj_pose_6dof"]
}

# Few-shot learning examples for the LLM
FEW_SHOT_EXAMPLES = """
User request: "Detect objects."
JSON pipeline: [{"module": "obj_detection", "mode": "live", "input_from": "camera", "input_type": "camera", "level": 0}]

User request: "Segment everything after detecting objects."
JSON pipeline: [
  {"module": "obj_detection", "mode": "live", "input_from": "camera", "input_type": "camera", "level": 0},
  {"module": "nano_seg_track", "mode": "live", "input_from": "obj_detection", "input_type": "BBOX", "level": 1}
]

User request: "Find cats and plan how to grasp them."
JSON pipeline: [
  {"module": "gdino", "mode": "live", "input_from": "camera", "input_type": "camera", "level": 0},
  {"module": "neugraspnet", "mode": "single", "input_from": "gdino", "input_type": "BBOX", "level": 1}
]

User request: "Recognize text and track human pose."
JSON pipeline: [
  {"module": "human_pose", "mode": "live", "input_from": "camera", "input_type": "camera", "level": 0},
  {"module": "ocr", "mode": "single", "input_from": "camera", "input_type": "camera", "level": 0},
  {"module": "nano_seg_track", "mode": "live", "input_from": "human_pose", "input_type": "POINTS", "level": 1}
]

User request: "Estimate the depth of objects in the scene."
JSON pipeline: [
  {"module": "obj_detection", "mode": "live", "input_from": "camera", "input_type": "camera", "level": 0},
  {"module": "scene_seg_track", "mode": "live", "input_from": "camera", "input_type": "camera", "level": 0},
  {"module": "monocular_depth", "mode": "single", "input_from": "camera", "input_type": "camera", "level": 0}
]

User request: "Segment objects without tracking them."
JSON pipeline: [
  {"module": "obj_detection", "mode": "live", "input_from": "camera", "input_type": "camera", "level": 0},
  {"module": "obj_segmentation", "mode": "single", "input_from": "obj_detection", "input_type": "BBOX", "level": 1}
]

User request: "Analyze what I can do with this object."
JSON pipeline: [
  {"module": "obj_detection", "mode": "live", "input_from": "camera", "input_type": "camera", "level": 0},
  {"module": "obj_affordance", "mode": "single", "input_from": "obj_detection", "input_type": "BBOX", "level": 1}
]

User request: "Find the 3D pose of objects in the scene."
JSON pipeline: [
  {"module": "obj_detection", "mode": "live", "input_from": "camera", "input_type": "camera", "level": 0},
  {"module": "nano_seg_track", "mode": "live", "input_from": "obj_detection", "input_type": "BBOX", "level": 1},
  {"module": "obj_pose_6dof", "mode": "live", "input_from": "nano_seg_track", "input_type": "MASK", "level": 2}
]

User request: "Analyze the physical properties of objects I point to in the scene."
JSON pipeline: [
  {"module": "gdino", "mode": "live", "input_from": "camera", "input_type": "camera", "level": 0},
  {"module": "llava", "mode": "single", "input_from": ["camera", "gdino"], "input_type": ["camera", "BBOX"], 
   "prompt": "Describe the material, texture, approximate weight, and possible use cases of this object", "level": 1}
]

User request: "Detect multiple objects in the scene and compare their physical properties."
JSON pipeline: [
  {"module": "obj_detection", "mode": "live", "input_from": "camera", "input_type": "camera", "level": 0},
  {"module": "nano_seg_track", "mode": "live", "input_from": "obj_detection", "input_type": "BBOX", "level": 1},
  {"module": "llava", "mode": "single", "input_from": ["camera", "nano_seg_track"], "input_type": ["camera", "MASK"], 
   "prompt": "Compare the physical properties of these objects. Which would be more durable? Which would be heavier?", "level": 2}
]

User request: "Tell me how I can interact with the objects in the scene."
JSON pipeline: [
  {"module": "gdino", "mode": "live", "input_from": "camera", "input_type": "camera", "level": 0},
  {"module": "nano_seg_track", "mode": "live", "input_from": "gdino", "input_type": "BBOX", "level": 1},
  {"module": "llava", "mode": "single", "input_from": ["camera", "nano_seg_track"], "input_type": ["camera", "MASK"], 
   "prompt": "Describe how a human could interact with each object. What affordances does each object have?", "level": 2}
]

User request: "Track the pose of objects in the scene."
JSON pipeline: [
  {"module": "gdino", "mode": "live", "input_from": "camera", "input_type": "camera", "level": 0},
  {"module": "nano_seg_track", "mode": "live", "input_from": "gdino", "input_type": "BBOX", "level": 1},
  {"module": "obj_pose_6dof", "mode": "live", "input_from": ["camera", "nano_seg_track"], 
   "input_type": ["camera", "MASK"], "level": 2}
]

User request: "Detect specific objects from my 3D model collection and estimate their pose."
JSON pipeline: [
  {"module": "gdino", "mode": "live", "input_from": "camera", "input_type": "camera", "level": 0},
  {"module": "nano_seg_track", "mode": "live", "input_from": "gdino", "input_type": "BBOX", "level": 1},
  {"module": "obj_pose_6dof", "mode": "live", "input_from": ["camera", "nano_seg_track"], 
   "input_type": ["camera", "MASK"], "model_library": "user_models", "level": 2}
]

User request: "Help me assemble objects by tracking their pose and understanding their physical properties."
JSON pipeline: [
  {"module": "gdino", "mode": "live", "input_from": "camera", "input_type": "camera", "level": 0},
  {"module": "nano_seg_track", "mode": "live", "input_from": "gdino", "input_type": "BBOX", "level": 1},
  {"module": "obj_pose_6dof", "mode": "live", "input_from": ["camera", "nano_seg_track"], 
   "input_type": ["camera", "MASK"], "level": 2},
  {"module": "llava", "mode": "single", "input_from": ["camera", "nano_seg_track"], 
   "input_type": ["camera", "MASK"], 
   "prompt": "Analyze how these parts fit together based on their physical properties", "level": 2}
]

User request: "Track the movement and orientation of multiple objects as they move in the scene."
JSON pipeline: [
  {"module": "obj_detection", "mode": "live", "input_from": "camera", "input_type": "camera", "level": 0},
  {"module": "nano_seg_track", "mode": "live", "input_from": "obj_detection", "input_type": "BBOX", "level": 1},
  {"module": "obj_pose_6dof", "mode": "live", "input_from": ["camera", "nano_seg_track"], 
   "input_type": ["camera", "MASK"], "tracking_id": "all", "level": 2}
]

User request: "Analyze the materials and textures of objects in the scene."
JSON pipeline: [
  {"module": "gdino", "mode": "live", "input_from": "camera", "input_type": "camera", "level": 0},
  {"module": "llava", "mode": "single", "input_from": ["camera", "gdino"], "input_type": ["camera", "BBOX"], 
   "prompt": "Describe in detail the material composition, texture properties, and surface characteristics of each detected object", "level": 1}
]

User request: "Detect an object from my prompt then estimate grasp and physical properties for that Object once then estimate live pose for that Object continuously."
JSON pipeline: [
  {"module": "gdino", "mode": "live", "input_from": "camera", "input_type": "camera", "level": 0},
  {"module": "neugraspnet", "mode": "single", "input_from": ["camera", "gdino"], "input_type": ["camera", "BBOX"], "level": 1},
  {"module": "llava", "mode": "single", "input_from": ["camera", "gdino"], "input_type": ["camera", "BBOX"], 
   "prompt": "Describe the physical properties of the detected object", "level": 1},
  {"module": "obj_pose_6dof", "mode": "live", "input_from": ["camera", "gdino"], "input_type": ["camera", "BBOX"], "level": 1}
]
"""

def assign_levels_to_pipeline(pipeline):
    """
    Assign level numbers to each module in the pipeline based on dependencies.
    Ensures all modules have input_type and input_from fields.
    
    Args:
        pipeline: List of module configurations
        
    Returns:
        Updated pipeline with level field and required input fields
    """
    if not pipeline:
        logger.warning("Empty pipeline passed to assign_levels_to_pipeline")
        return []
    
    # Step 1: Check if levels are already assigned
    if all("level" in module for module in pipeline):
        logger.info("Levels already assigned to pipeline, skipping level assignment")
        return pipeline
    
    # Step 2: Ensure all modules have input_type and input_from fields
    for module in pipeline:
        # If input_from exists but input_type doesn't, try to determine it
        if "input_from" in module and "input_type" not in module:
            input_from = module["input_from"]
            if input_from == "camera":
                module["input_type"] = "camera"
            else:
                input_module_info = MODULES_INFO.get(input_from, {})
                
                # Use the first output type if multiple are available
                if input_module_info:
                    output_types = input_module_info.get("outputs", "").split(",")
                    if output_types and output_types[0]:
                        module["input_type"] = output_types[0]
                    else:
                        module["input_type"] = "camera"
                else:
                    module["input_type"] = "camera"
        
        # If neither field exists, set both to "camera"
        elif "input_from" not in module and "input_type" not in module:
            module["input_from"] = "camera"
            module["input_type"] = "camera"
        
        # If only input_type exists but not input_from, set input_from to "camera"
        elif "input_type" in module and "input_from" not in module:
            module["input_from"] = "camera"
        
        # If only input_from exists with value "camera", ensure input_type is also "camera"
        elif module.get("input_from") == "camera" and "input_type" not in module:
            module["input_type"] = "camera"
    
    # Step 3: Create a dependency graph
    dependency_graph = {}
    module_indices = {}
    
    # Map module names to their indices for quick lookup
    for i, module in enumerate(pipeline):
        module_name = module["module"]
        module_indices[module_name] = i
        dependency_graph[module_name] = []
    
    # Build dependency relationships
    for module in pipeline:
        module_name = module["module"]
        if "input_from" in module:
            if isinstance(module["input_from"], list):
                for input_source in module["input_from"]:
                    if input_source != "camera" and input_source in dependency_graph:
                        dependency_graph[module_name].append(input_source)
            elif module["input_from"] != "camera" and module["input_from"] in dependency_graph:
                dependency_graph[module_name].append(module["input_from"])
    
    # Step 4: Perform topological sorting to assign levels
    # Initialize all modules with no level
    levels = {module_name: -1 for module_name in dependency_graph}
    
    # Modules with no dependencies (or "camera" dependencies) start at level 0
    for module_name in dependency_graph:
        module_index = module_indices[module_name]
        module = pipeline[module_index]
        if not dependency_graph[module_name] or (
            isinstance(module.get("input_from"), str) and module.get("input_from") == "camera"):
            levels[module_name] = 0
    
    # Keep assigning levels until all modules have levels
    changed = True
    max_iterations = len(dependency_graph) * 2  # Safety to prevent infinite loops
    iteration = 0
    
    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        
        for module_name in dependency_graph:
            # Skip modules that already have levels assigned
            if levels[module_name] >= 0:
                continue
                
            # Check if all dependencies have levels assigned
            if all(levels.get(dep, -1) >= 0 for dep in dependency_graph[module_name]):
                # Assign level as max(dependency_levels) + 1
                max_dep_level = max((levels.get(dep, -1) for dep in dependency_graph[module_name]), default=-1)
                levels[module_name] = max_dep_level + 1
                changed = True
    
    # Check for unassigned levels due to circular dependencies
    unassigned = [module_name for module_name, level in levels.items() if level < 0]
    if unassigned:
        logger.warning(f"Could not assign levels to modules due to circular dependencies: {unassigned}")
        # Assign default level 0 to unassigned modules
        for module_name in unassigned:
            levels[module_name] = 0
    
    # Step 5: Update the pipeline with level assignments
    for module_name, level in levels.items():
        if module_name in module_indices:
            pipeline[module_indices[module_name]]["level"] = level
    
    logger.info(f"Assigned levels to pipeline with {len(pipeline)} modules")
    return pipeline

def validate_hierarchical_pipeline(pipeline):
    """
    Validate that the hierarchical pipeline structure is correct with required input fields.
    
    Args:
        pipeline: List of module configurations with level field and input fields
        
    Returns:
        (is_valid, error_message)
    """
    if not pipeline:
        return False, "Empty pipeline"
    
    # Check that all modules have required fields
    for i, module in enumerate(pipeline):
        if "module" not in module:
            return False, f"Module at index {i} missing 'module' field"
            
        if "mode" not in module:
            return False, f"Module {module.get('module', f'at index {i}')} missing 'mode' field"
            
        if "level" not in module:
            return False, f"Module {module.get('module', f'at index {i}')} missing 'level' field"
            
        if "input_from" not in module:
            return False, f"Module {module.get('module', f'at index {i}')} missing 'input_from' field"
            
        if "input_type" not in module:
            return False, f"Module {module.get('module', f'at index {i}')} missing 'input_type' field"
        
        # Validate module name exists in MODULES_INFO
        if module["module"] not in MODULES_INFO:
            return False, f"Module {module['module']} is not a valid module type"
            
        # Check if input_from references an existing module or is 'camera'
        if isinstance(module["input_from"], str):
            if module["input_from"] != "camera" and module["input_from"] not in [m["module"] for m in pipeline]:
                return False, f"Module {module['module']} depends on non-existent module {module['input_from']}"
        elif isinstance(module["input_from"], list):
            for input_source in module["input_from"]:
                if input_source != "camera" and input_source not in [m["module"] for m in pipeline]:
                    return False, f"Module {module['module']} depends on non-existent module {input_source}"
        else:
            return False, f"Module {module['module']} has invalid input_from type: {type(module['input_from'])}"
            
        # Ensure input_from and input_type consistency
        if isinstance(module["input_from"], str) and module["input_from"] == "camera":
            if isinstance(module["input_type"], str) and module["input_type"] != "camera":
                return False, f"Module {module['module']} has input_from='camera' but input_type is not 'camera'"
    
    # Check that dependencies are consistent with levels
    for module in pipeline:
        module_level = module["level"]
        
        # If a module has real dependencies, its level should be higher than its dependencies
        if module["input_from"] != "camera":
            if isinstance(module["input_from"], str):
                input_modules = [m for m in pipeline if m["module"] == module["input_from"]]
                
                if not input_modules:
                    return False, f"Module {module['module']} depends on non-existent module {module['input_from']}"
                    
                input_level = input_modules[0]["level"]
                if module_level <= input_level:
                    return False, f"Module {module['module']} at level {module_level} depends on " \
                                 f"module {module['input_from']} at level {input_level}, but should be at a higher level"
            elif isinstance(module["input_from"], list):
                for input_source in module["input_from"]:
                    if input_source != "camera":
                        input_modules = [m for m in pipeline if m["module"] == input_source]
                        
                        if not input_modules:
                            return False, f"Module {module['module']} depends on non-existent module {input_source}"
                            
                        input_level = input_modules[0]["level"]
                        if module_level <= input_level:
                            return False, f"Module {module['module']} at level {module_level} depends on " \
                                         f"module {input_source} at level {input_level}, but should be at a higher level"
    
    return True, "Hierarchical pipeline is valid"

class LocalLLMPipelineGenerator:
    """
    Class for using TinyLLaMA to generate pipeline configurations
    """
    def __init__(self, model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self.model = None
        self.tokenizer = None
        self.initialized = False
        self.model_path = model_path
        
        if LLM_AVAILABLE:
            try:
                self._initialize_model(model_path)
            except Exception as e:
                logger.error(f"Error initializing TinyLLaMA: {e}")
    
    def _initialize_model(self, model_path):
        """Initialize the TinyLLaMA model with optimizations"""
        try:
            logger.info(f"Loading TinyLLaMA from {model_path}...")
            
            # Advanced quantization configuration
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",  # Use normalized float 4 for better quality
                bnb_4bit_use_double_quant=True  # Use nested quantization for further memory savings
            )
            
            # Load tokenizer with caching
            tokenizer_kwargs = {
                "use_fast": True,  # Use the fast Rust-based tokenizer
                "padding_side": "left",  # Better for causal LM outputs
            }
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
            
            # Performance optimization flags
            model_kwargs = {
                "quantization_config": quantization_config,
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "low_cpu_mem_usage": True,
            }
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            
            # Set evaluation mode
            self.model.eval()
            
            self.initialized = True
            logger.info(f"TinyLLaMA initialized successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize TinyLLaMA: {e}")
            raise
    
    def extract_json_from_text(self, text):
        """Enhanced JSON extraction with better error handling"""
        logger.debug(f"Attempting to extract JSON from text of length {len(text)}")
        
        # First try direct regex with better pattern handling
        json_patterns = [
            r'\[(\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*)\]',  # Standard JSON array with relaxed whitespace
            r'```json\s*(\[.*?\])',  # Markdown code block
            r'```\s*(\[.*?\])',      # Generic code block
            r'(\[{"module":.*?}\])'  # Specific structure match
        ]
        
        for pattern in json_patterns:
            try:
                matches = re.findall(pattern, text, re.DOTALL)
                if matches:
                    for match in matches:
                        try:
                            # For patterns that capture the inner content of the array
                            if pattern == r'\[(\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*)\]':
                                json_str = f"[{match}]"
                            else:
                                json_str = match
                            
                            pipeline_config = json.loads(json_str)
                            if isinstance(pipeline_config, list) and len(pipeline_config) > 0:
                                if all(isinstance(item, dict) and "module" in item for item in pipeline_config):
                                    logger.info(f"Successfully extracted JSON array with {len(pipeline_config)} pipeline modules")
                                    return pipeline_config
                        except json.JSONDecodeError as e:
                            logger.debug(f"Failed to parse match as JSON: {e}")
                            continue
            except Exception as e:
                logger.debug(f"Error with pattern {pattern}: {e}")
                continue
        
        # If regex approach failed, try more advanced parsing
        try:
            # Look for bracket-enclosed content that might be JSON
            start_idx = text.find('[')
            if start_idx >= 0:
                # Find matching closing bracket
                bracket_count = 0
                for i in range(start_idx, len(text)):
                    if text[i] == '[':
                        bracket_count += 1
                    elif text[i] == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_idx = i + 1
                            json_candidate = text[start_idx:end_idx]
                            try:
                                config = json.loads(json_candidate)
                                if isinstance(config, list) and len(config) > 0:
                                    if all(isinstance(item, dict) and "module" in item for item in config):
                                        logger.info(f"Successfully extracted JSON using bracket matching")
                                        return config
                            except:
                                pass
        except Exception as e:
            logger.debug(f"Advanced JSON extraction failed: {e}")
        
        logger.warning("Could not extract valid JSON pipeline from text")
        return None
    
    def generate_pipeline(self, user_request):
        """Generate hierarchical pipeline JSON using TinyLLaMA with required input fields"""
        if not self.initialized or not LLM_AVAILABLE:
            logger.warning("LLM not initialized or not available, cannot generate pipeline")
            return None
        
        # Create prompt for TinyLLaMA with hierarchical structure and required input fields
        normalized_request = user_request.strip().lower()
        
        # Check for direct examples first to improve accuracy
        for example_request, example_pipeline in PROMPT_TO_PIPELINE.items():
            if normalized_request == example_request.lower() or normalized_request == example_request.lower() + '.':
                logger.info(f"Found exact match in predefined examples for: '{user_request}'")
                return example_pipeline
        
        prompt = f"""<|system|>
You are an AI assistant that converts natural language requests into JSON pipeline configurations for computer vision tasks.
Each module in the pipeline MUST have the following fields:
- "module": the module identifier
- "mode": either "live" or "single"
- "input_from": identifies which module provides input to this module, or "camera" if none
- "input_type": specifies the type of input expected (e.g., "BBOX", "MASK"), or "camera" if none
- "level": indicates the execution level, where modules at the same level run in parallel

Level assignments follow these rules:
- Modules that are to run first are assigned level 0
- A module that depends on modules from level N is assigned level N+1
- Modules at the same level can run in parallel

Here are some example hierarchical pipelines:

{FEW_SHOT_EXAMPLES}

Convert the following request into a hierarchical JSON pipeline configuration:
<|user|>
{user_request}
<|assistant|>
```json
"""
        
        try:
            # Tokenize the input
            logger.info(f"Generating pipeline for request: '{user_request}'")
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=1024,  # Increased token limit
                    temperature=0.2,      # Lower temperature for more deterministic outputs
                    top_p=0.95,
                    repetition_penalty=1.2,
                    do_sample=True
                )
            
            # Decode the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            logger.debug(f"Raw model response: {response}")
            
            # Extract the assistant's response
            if "<|assistant|>" in response:
                response = response.split("<|assistant|>")[1].strip()
            
            # Extract JSON from response
            pipeline_config = self.extract_json_from_text(response)
            
            if pipeline_config:
                # Ensure all modules have required fields
                for module in pipeline_config:
                    if "input_from" not in module:
                        module["input_from"] = "camera"
                    if "input_type" not in module:
                        if module["input_from"] == "camera":
                            module["input_type"] = "camera"
                        else:
                            # Try to determine input type based on input_from
                            input_from = module["input_from"]
                            if isinstance(input_from, str):
                                input_module_info = MODULES_INFO.get(input_from, {})
                                output_types = input_module_info.get("outputs", "").split(",")
                                module["input_type"] = output_types[0] if output_types and output_types[0] else "camera"
                            else:
                                module["input_type"] = "camera"
                
                # Validate pipeline before returning
                is_valid, error_message = validate_hierarchical_pipeline(pipeline_config)
                if is_valid:
                    logger.info(f"Generated valid pipeline with {len(pipeline_config)} modules")
                    return pipeline_config
                else:
                    logger.warning(f"Generated pipeline fails validation: {error_message}")
                    # We'll still return it for fixing later
                    return pipeline_config
            
            logger.warning("Failed to extract valid JSON pipeline from LLM response")
            return None
        except Exception as e:
            logger.error(f"Error generating pipeline with TinyLLaMA: {e}", exc_info=True)
            return None

    def batch_generate_pipelines(self, user_requests, batch_size=4):
        """Generate pipelines for multiple requests in batches"""
        if not self.initialized or not LLM_AVAILABLE:
            logger.warning("LLM not initialized or not available, cannot batch generate pipelines")
            return [None] * len(user_requests)
        
        results = []
        logger.info(f"Batch generating pipelines for {len(user_requests)} requests")
        
        # Process in batches
        for i in range(0, len(user_requests), batch_size):
            batch = user_requests[i:i+batch_size]
            batch_results = []
            
            # Process each request in the batch sequentially to avoid resource issues
            for request in batch:
                pipeline = self.generate_pipeline(request)
                batch_results.append(pipeline)
            
            results.extend(batch_results)
        
        return results

class HybridPipelineGenerator:
    """
    Hybrid approach combining rule-based matching and LLM generation
    """
    def __init__(self, llm_model_path=None):
        # Initialize the LLM component if available
        self.llm_generator = LocalLLMPipelineGenerator(llm_model_path)
        self.use_llm = self.llm_generator.initialized
        logger.info(f"Initialized HybridPipelineGenerator, LLM available: {self.use_llm}")
    
    def find_best_match(self, text):
        """Find the best matching predefined prompt based on similarity"""
        text = text.lower().strip()
        logger.debug(f"Finding best predefined match for: '{text}'")
        
        # Direct match (case insensitive)
        for prompt in PROMPT_TO_PIPELINE.keys():
            normalized_prompt = prompt.lower()
            if (text == normalized_prompt or 
                text == normalized_prompt + '.' or 
                text == normalized_prompt + '?'):
                logger.info(f"Found direct match: '{prompt}'")
                return prompt
        
        # Task-based matching
        tasks_found = []
        
        # Check for each task type based on keywords
        task_keywords = {
            "object_detection": ["detect", "find", "locate", "object"],
            "segmentation": ["segment", "mask", "track"],
            "reading_text": ["text", "read", "ocr", "character"],
            "hand_tracking": ["hand", "gesture", "finger"],
            "depth_analysis": ["depth", "distance", "3d"],
            "human_tracking": ["human", "pose", "person", "body"],
            "grasp_planning": ["grasp", "grab", "hold", "pick"],
            "physical_properties": ["physical", "material", "property", "describe", "analyze"]
        }
        
        for task, keywords in task_keywords.items():
            if any(keyword in text for keyword in keywords):
                tasks_found.append(task)
                logger.debug(f"Detected task '{task}' in text")
        
        # Specifically match the complex example
        if ("detect" in text and "grasp" in text and "physical" in text and "pose" in text and
            ("properties" in text or "property" in text) and "continuous" in text):
            logger.info("Matched complex task pattern with detecting, grasping, physical properties and pose")
            return "detect an object from my prompt then estimate grasp and physical properties for that object once then estimate live pose for that object continuously"
        
        # Map tasks to example prompts based on combinations
        if tasks_found:
            if "segmentation" in tasks_found and "object_detection" in tasks_found:
                return "segment everything after detecting objects"
            
            if "reading_text" in tasks_found and "human_tracking" in tasks_found:
                return "recognize text and track human pose"
            
            if "grasp_planning" in tasks_found:
                return "find cats and plan how to grasp them"
            
            if "depth_analysis" in tasks_found:
                return "estimate the depth of objects in the scene"
            
            if "hand_tracking" in tasks_found:
                return "track hand movements and identify gestures"
            
            if "physical_properties" in tasks_found and "object_detection" in tasks_found:
                return "analyze the materials and textures of objects in the scene"
            
            if "object_detection" in tasks_found:
                for keyword in ["oriented", "orientation", "rotated", "angle"]:
                    if keyword in text:
                        return "detect oriented objects in the image"
                return "detect objects"
            
            if "physical_properties" in tasks_found:
                return "describe what you see in the image"
        
        # Default to object detection
        logger.info("No specific match found, defaulting to 'detect objects'")
        return "detect objects"
    
    def extract_modules_from_text(self, text):
        """Extract modules mentioned in the text"""
        text = text.lower().strip()
        modules = []
        logger.debug(f"Extracting modules from text: '{text}'")
        
        # Extract modules by keyword matching
        for module_id, module_info in MODULES_INFO.items():
            if any(keyword in text for keyword in module_info["keywords"]):
                mode = "live" if module_info["inference_type"] == "Live" else "single"
                module_config = {
                    "module": module_id, 
                    "mode": mode,
                    "input_from": "camera",
                    "input_type": "camera"
                }
                
                # Add prompt for modules that need it
                if module_info["input"] == "Prompt":
                    if module_id == "llava":
                        module_config["prompt"] = "Describe what you see in this image"
                        
                    elif module_id == "gdino" or module_id == "vit_detection":
                        # Try to extract specific object names
                        object_pattern = r"(?:find|detect|locate)\s+(?:a|an|the)?\s*([a-zA-Z\s]+)"
                        match = re.search(object_pattern, text)
                        if match:
                            module_config["prompt"] = match.group(1).strip()
                
                modules.append(module_config)
                logger.debug(f"Extracted module: {module_id}")
        
        return modules
    
    def rule_based_generate(self, text):
        """Generate pipeline using rule-based approach"""
        logger.info("Using rule-based pipeline generation")
        
        # Find best matching example
        best_match = self.find_best_match(text)
        
        # Return the corresponding predefined pipeline
        if best_match in PROMPT_TO_PIPELINE:
            pipeline = PROMPT_TO_PIPELINE[best_match]
            logger.info(f"Using predefined pipeline for '{best_match}' with {len(pipeline)} modules")
            return pipeline
        
        # Fallback to basic obj_detection if no match found
        logger.info("No matching predefined pipeline, defaulting to 'detect objects'")
        return PROMPT_TO_PIPELINE["detect objects"]
    
    def validate_pipeline(self, pipeline):
        """Validate that the pipeline configuration is correct"""
        if not pipeline:
            logger.warning("Empty pipeline in validation")
            return False
        
        # Check that all modules exist and have required fields
        for module in pipeline:
            if "module" not in module:
                logger.warning("Pipeline module missing 'module' field")
                return False
                
            if module["module"] not in MODULES_INFO:
                logger.warning(f"Unknown module type in pipeline: {module.get('module')}")
                return False
                
            if "mode" not in module:
                logger.warning(f"Module {module['module']} missing 'mode' field")
                return False
        
        logger.info(f"Pipeline validation passed for {len(pipeline)} modules")
        return True
    
    def generate_pipeline(self, user_request):
        """
        Generate pipeline JSON using hybrid approach.
        First tries LLM-based generation if available, falls back to rule-based if needed.
        """
        normalized_request = user_request.strip().lower()
        logger.info(f"Generating pipeline for: '{user_request}'")
        
        # Check for direct match in predefined pipelines first (most reliable)
        for example_request, example_pipeline in PROMPT_TO_PIPELINE.items():
            if (normalized_request == example_request.lower() or 
                normalized_request == example_request.lower() + '.' or
                normalized_request == example_request.lower() + '?'):
                logger.info(f"Found exact match in predefined examples: '{example_request}'")
                return example_pipeline
        
        # Step 1: Try LLM-based generation if available
        if self.use_llm:
            logger.info("Attempting LLM-based generation")
            llm_pipeline = self.llm_generator.generate_pipeline(user_request)
            if llm_pipeline and self.validate_pipeline(llm_pipeline):
                logger.info("Successfully generated pipeline using LLM")
                return llm_pipeline
            else:
                logger.warning("LLM pipeline generation failed or produced invalid result")
        else:
            logger.info("LLM not available, skipping LLM-based generation")
        
        # Step 2: Fall back to rule-based approach
        logger.info("Falling back to rule-based generation")
        return self.rule_based_generate(user_request)

class HierarchicalPipelineGenerator(HybridPipelineGenerator):
    """
    Extension of HybridPipelineGenerator that produces hierarchical pipelines
    with level assignments for parallel execution
    """
    def __init__(self, llm_model_path=None):
        super().__init__(llm_model_path)
        logger.info("Initialized HierarchicalPipelineGenerator")
    
    def generate_pipeline(self, user_request):
        """
        Generate a hierarchical pipeline with level assignments
        """
        # Get base pipeline from parent class
        base_pipeline = super().generate_pipeline(user_request)
        
        if not base_pipeline:
            logger.warning("Failed to generate base pipeline")
            return None
        
        # Ensure all modules have input_type and input_from fields
        for module in base_pipeline:
            # If input_from exists but input_type doesn't, try to determine it
            if "input_from" in module and "input_type" not in module:
                input_from = module["input_from"]
                if input_from == "camera":
                    module["input_type"] = "camera"
                else:
                    input_module_info = MODULES_INFO.get(input_from, {})
                    
                    # Use the first output type if multiple are available
                    if input_module_info:
                        output_types = input_module_info.get("outputs", "").split(",")
                        if output_types and output_types[0]:
                            module["input_type"] = output_types[0]
                        else:
                            module["input_type"] = "camera"
                    else:
                        module["input_type"] = "camera"
            
            # If neither field exists, set both to "camera"
            elif "input_from" not in module and "input_type" not in module:
                module["input_from"] = "camera"
                module["input_type"] = "camera"
            
            # If only input_type exists but not input_from, set input_from to "camera"
            elif "input_type" in module and "input_from" not in module:
                module["input_from"] = "camera"
            
            # If only input_from exists with value "camera", ensure input_type is also "camera"
            elif module.get("input_from") == "camera" and "input_type" not in module:
                module["input_type"] = "camera"
        
        # Check if level is already assigned to all modules
        if all("level" in module for module in base_pipeline):
            logger.info("Pipeline already has level assignments, skipping level assignment")
            hierarchical_pipeline = base_pipeline
        else:
            # Assign hierarchical levels based on dependencies
            logger.info("Assigning hierarchical levels to pipeline")
            hierarchical_pipeline = assign_levels_to_pipeline(base_pipeline)
        
        # Validate the hierarchical structure
        is_valid, error_message = validate_hierarchical_pipeline(hierarchical_pipeline)
        if not is_valid:
            logger.warning(f"Invalid hierarchical pipeline: {error_message}, attempting to fix")
            # Try to fix the pipeline
            hierarchical_pipeline = self.fix_hierarchical_pipeline(hierarchical_pipeline)
            
            # Re-validate after fixing
            is_valid, error_message = validate_hierarchical_pipeline(hierarchical_pipeline)
            if not is_valid:
                logger.error(f"Failed to fix pipeline: {error_message}")
                return base_pipeline  # Return original as fallback
        
        return hierarchical_pipeline
    
    def fix_hierarchical_pipeline(self, pipeline):
        """
        Attempt to fix common issues in a hierarchical pipeline
        """
        if not pipeline:
            logger.warning("Cannot fix empty pipeline")
            return []
        
        logger.info(f"Attempting to fix pipeline with {len(pipeline)} modules")
        
        # Deep copy to avoid modifying original
        fixed_pipeline = json.loads(json.dumps(pipeline))
        
        # Fix 1: Add level field if missing
        for module in fixed_pipeline:
            if "level" not in module:
                module["level"] = 0
                logger.debug(f"Added missing level=0 to module {module['module']}")
        
        # Fix 2: Ensure all modules have valid input_from and input_type
        for module in fixed_pipeline:
            # Add missing input_from
            if "input_from" not in module:
                module["input_from"] = "camera"
                logger.debug(f"Added missing input_from=camera to module {module['module']}")
            
            # Fix input_from referencing non-existent modules
            if isinstance(module["input_from"], str) and module["input_from"] != "camera":
                if module["input_from"] not in [m["module"] for m in fixed_pipeline]:
                    logger.warning(f"Module {module['module']} references non-existent module {module['input_from']}, "
                                  f"changing to camera")
                    module["input_from"] = "camera"
            elif isinstance(module["input_from"], list):
                valid_inputs = ["camera"]
                valid_inputs.extend([m["module"] for m in fixed_pipeline])
                module["input_from"] = [inp for inp in module["input_from"] if inp in valid_inputs]
                if not module["input_from"]:
                    module["input_from"] = "camera"
                    logger.debug(f"Fixed invalid input_from list in module {module['module']}")
            
            # Add missing input_type
            if "input_type" not in module:
                if module["input_from"] == "camera":
                    module["input_type"] = "camera"
                else:
                    # Try to determine input type based on input_from
                    if isinstance(module["input_from"], str):
                        input_from = module["input_from"]
                        input_module_info = MODULES_INFO.get(input_from, {})
                        output_types = input_module_info.get("outputs", "").split(",")
                        module["input_type"] = output_types[0] if output_types and output_types[0] else "camera"
                    else:
                        module["input_type"] = "camera"
                logger.debug(f"Added missing input_type to module {module['module']}")
        
        # Fix 3: Ensure mode field
        for module in fixed_pipeline:
            if "mode" not in module:
                module_info = MODULES_INFO.get(module["module"], {})
                if module_info.get("inference_type") == "Live":
                    module["mode"] = "live"
                else:
                    module["mode"] = "single"
                logger.debug(f"Added missing mode to module {module['module']}")
        
        # Fix 4: Add required LLaVa prompt if missing
        for module in fixed_pipeline:
            if module["module"] == "llava" and "prompt" not in module:
                module["prompt"] = "Describe what you see in this image"
                logger.debug("Added missing prompt to LLaVa module")
        
        # Fix 5: Assign levels based on dependencies (full recomputation)
        fixed_pipeline = assign_levels_to_pipeline(fixed_pipeline)
        
        logger.info(f"Fixed pipeline now has {len(fixed_pipeline)} modules")
        return fixed_pipeline

class CachingPipelineGenerator(HierarchicalPipelineGenerator):
    """Extension of HierarchicalPipelineGenerator with caching"""
    def __init__(self, llm_model_path=None, cache_size=100):
        super().__init__(llm_model_path)
        self.cache = {}
        self.cache_size = cache_size
        self.cache_keys = []  # To track LRU
        logger.info(f"Initialized CachingPipelineGenerator with cache_size={cache_size}")
    
    def generate_pipeline(self, user_request):
        """Generate pipeline with caching"""
        # Normalize the request to improve cache hits
        normalized_request = user_request.lower().strip()
        
        # Check cache
        if normalized_request in self.cache:
            # Move this key to the end (most recently used)
            self.cache_keys.remove(normalized_request)
            self.cache_keys.append(normalized_request)
            logger.info(f"Cache hit for: '{normalized_request}'")
            return self.cache[normalized_request]
        
        # Generate pipeline
        start_time = datetime.now()
        pipeline = super().generate_pipeline(user_request)
        generation_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Pipeline generation took {generation_time:.2f} seconds")
        
        # Update cache
        if pipeline:
            # If cache is full, remove least recently used item
            if len(self.cache) >= self.cache_size:
                oldest_key = self.cache_keys.pop(0)
                del self.cache[oldest_key]
                logger.debug(f"Removed oldest cache entry: '{oldest_key}'")
            
            # Add to cache
            self.cache[normalized_request] = pipeline
            self.cache_keys.append(normalized_request)
            logger.info(f"Added to cache: '{normalized_request}'")
        
        return pipeline
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache = {}
        self.cache_keys = []
        logger.info("Cache cleared")
    
    def get_cache_stats(self):
        """Return cache statistics"""
        return {
            "cache_size": len(self.cache),
            "max_cache_size": self.cache_size,
            "cache_keys": self.cache_keys
        }

def validate_pipeline_semantics(prompt, pipeline, modules_info=MODULES_INFO):
    """
    Validate that the pipeline is semantically correct for the given prompt
    Returns (is_valid, list_of_issues)
    """
    prompt = prompt.lower()
    logger.debug(f"Validating pipeline semantics for prompt: '{prompt}'")
    
    # Extract tasks from prompt
    tasks = []
    if any(keyword in prompt for keyword in ["detect", "find", "object"]):
        tasks.append("object_detection")
    if any(keyword in prompt for keyword in ["segment", "mask"]):
        tasks.append("segmentation")
    if any(keyword in prompt for keyword in ["text", "read"]):
        tasks.append("text_recognition")
    if any(keyword in prompt for keyword in ["depth", "distance"]):
        tasks.append("depth_estimation")
    if any(keyword in prompt for keyword in ["human", "pose", "body"]):
        tasks.append("human_pose")
    if any(keyword in prompt for keyword in ["hand", "gesture"]):
        tasks.append("hand_pose")
    if any(keyword in prompt for keyword in ["grasp", "pick", "hold"]):
        tasks.append("grasp_planning")
    if any(keyword in prompt for keyword in ["describe", "what", "tell"]):
        tasks.append("description")
    
    # Check if pipeline satisfies the required tasks
    pipeline_modules = [m["module"] for m in pipeline]
    
    checks = []
    if "object_detection" in tasks and not any(m in pipeline_modules for m in ["obj_detection", "gdino"]):
        checks.append("Missing object detection module")
    
    if "segmentation" in tasks and not any(m in pipeline_modules for m in ["nano_seg_track", "scene_seg_track", "obj_segmentation"]):
        checks.append("Missing segmentation module")
    
    if "text_recognition" in tasks and "ocr" not in pipeline_modules:
        checks.append("Missing OCR module")
    
    if "depth_estimation" in tasks and "monocular_depth" not in pipeline_modules:
        checks.append("Missing depth estimation module")
    
    if "human_pose" in tasks and "human_pose" not in pipeline_modules:
        checks.append("Missing human pose module")
    
    if "hand_pose" in tasks and "hand_pose" not in pipeline_modules:
        checks.append("Missing hand pose module")
    
    if "grasp_planning" in tasks and "neugraspnet" not in pipeline_modules:
        checks.append("Missing grasp planning module")
    
    if "description" in tasks and "llava" not in pipeline_modules:
        checks.append("Missing description module")
    
    is_valid = len(checks) == 0
    logger.info(f"Pipeline semantic validation: {'PASSED' if is_valid else 'FAILED'}")
    if not is_valid:
        logger.warning(f"Semantic issues found: {checks}")
    
    return is_valid, checks

def nl_to_hierarchical_vision_pipeline(natural_language_request: str, llm_model_path: Optional[str] = None, use_cache: bool = True) -> List[Dict[str, Any]]:
    """
    Convert a natural language request to a hierarchical vision pipeline configuration
    with level assignments for parallel execution.
    
    Args:
        natural_language_request: User's natural language description of desired vision processing
        llm_model_path: Optional path to a local LLM model
        use_cache: Whether to use caching for pipeline generation
        
    Returns:
        A list of module configurations with hierarchical level assignments
    """
    # Check for direct match in predefined pipelines (case-insensitive)
    normalized_request = natural_language_request.strip().lower()
    for example_request, example_pipeline in PROMPT_TO_PIPELINE.items():
        if (normalized_request == example_request.lower() or 
            normalized_request == example_request.lower() + '.' or
            normalized_request == example_request.lower() + '?'):
            logger.info(f"Direct match found in predefined examples for: '{natural_language_request}'")
            return example_pipeline
    
    # Initialize the generator based on caching preference
    if use_cache:
        generator = CachingPipelineGenerator(llm_model_path)
    else:
        generator = HierarchicalPipelineGenerator(llm_model_path)
    
    # Generate hierarchical pipeline configuration
    pipeline = generator.generate_pipeline(natural_language_request)
    
    # Validate pipeline semantics
    if pipeline:
        is_valid, issues = validate_pipeline_semantics(natural_language_request, pipeline)
        if not is_valid:
            logger.warning(f"Generated pipeline has semantic issues: {issues}")
            # We still return the pipeline, as the semantic validation is advisory
    
    return pipeline

# Utility function to normalize prompt text for better matching
def normalize_prompt(text):
    """Normalize prompt text for better matching"""
    # Convert to lowercase
    text = text.lower().strip()
    
    # Remove punctuation at the end
    if text.endswith('.') or text.endswith('?') or text.endswith('!'):
        text = text[:-1]
    
    # Remove extra spaces
    text = ' '.join(text.split())
    
    # Replace specific terms for consistency
    replacements = {
        "objects": "object",
        "estimate": "detect",
        "physical properties": "physical property",
        "identify": "detect",
        "continuously": "continuous"
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    return text

def debug_pipeline_generation(prompt, model_path=None):
    """
    Debug utility function to test pipeline generation step by step
    """
    print(f"\n=== DEBUG: Pipeline Generation for '{prompt}' ===")
    
    # Step 1: Normalize the prompt
    normalized_prompt = normalize_prompt(prompt)
    print(f"Normalized prompt: '{normalized_prompt}'")
    
    # Step 2: Check for direct matches
    for key in PROMPT_TO_PIPELINE.keys():
        norm_key = normalize_prompt(key)
        if normalized_prompt == norm_key:
            print(f"DIRECT MATCH found in predefined pipelines: '{key}'")
            result = PROMPT_TO_PIPELINE[key]
            print(f"Predefined pipeline: {json.dumps(result, indent=2)}")
            return result
    
    print("No direct match found in predefined pipelines")
    
    # Step 3: Initialize generator and try LLM generation
    generator = HierarchicalPipelineGenerator(model_path)
    
    if generator.use_llm:
        print("Attempting LLM-based generation...")
        llm_pipeline = generator.llm_generator.generate_pipeline(prompt)
        
        if llm_pipeline:
            print(f"LLM generated pipeline: {json.dumps(llm_pipeline, indent=2)}")
            
            # Validate the LLM pipeline
            is_valid = generator.validate_pipeline(llm_pipeline)
            print(f"LLM pipeline validation: {'PASSED' if is_valid else 'FAILED'}")
            
            if is_valid:
                # Assign levels if needed
                if not all("level" in module for module in llm_pipeline):
                    print("Assigning hierarchical levels...")
                    llm_pipeline = assign_levels_to_pipeline(llm_pipeline)
                
                # Validate hierarchical structure
                h_valid, h_error = validate_hierarchical_pipeline(llm_pipeline)
                print(f"Hierarchical validation: {'PASSED' if h_valid else 'FAILED'}")
                if not h_valid:
                    print(f"Hierarchical error: {h_error}")
                    print("Attempting to fix pipeline...")
                    llm_pipeline = generator.fix_hierarchical_pipeline(llm_pipeline)
                
                return llm_pipeline
        else:
            print("LLM failed to generate a valid pipeline")
    else:
        print("LLM not available")
    
    # Step 4: Fall back to rule-based approach
    print("Falling back to rule-based generation...")
    best_match = generator.find_best_match(prompt)
    print(f"Best rule-based match: '{best_match}'")
    
    rule_pipeline = generator.rule_based_generate(prompt)
    print(f"Rule-based pipeline: {json.dumps(rule_pipeline, indent=2)}")
    
    return rule_pipeline

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("pipeline_generator.log")
        ]
    )
    
    # Set this to the TinyLLaMA model path
    model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Test with the problematic example
    test_examples = [
        "Detect an object from my prompt then estimate grasp and physical properties for that Object once then estimate live pose for that Object continuously."
    ]
    
    for example in test_examples:
        # Log original and normalized prompt for comparison
        original_prompt = example
        normalized_prompt = normalize_prompt(original_prompt)
        print(f"\nOriginal prompt: '{original_prompt}'")
        print(f"Normalized prompt: '{normalized_prompt}'")
        
        # Check for exact matches in predefined pipelines
        exact_match_found = False
        for key in PROMPT_TO_PIPELINE.keys():
            norm_key = normalize_prompt(key)
            if normalized_prompt == norm_key:
                print(f"EXACT MATCH found in predefined pipelines: '{key}'")
                exact_match_found = True
                result = PROMPT_TO_PIPELINE[key]
                break
        
        if not exact_match_found:
            # Try to generate the pipeline
            print(f"No exact match found, generating pipeline...")
            result = nl_to_hierarchical_vision_pipeline(example, model_path, use_cache=False)
        
        # Print the result
        if result:
            print(f"Pipeline generated with {len(result)} modules:")
            print(json.dumps(result, indent=2))
            
            # Save the result to input.json
            try:
                output_file = "C:/Users/vrund/OneDrive/Desktop/sastra/final/input.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"Pipeline JSON saved to '{output_file}' successfully.")
            except Exception as e:
                print(f"Error saving pipeline to JSON file: {str(e)}")
        else:
            print("Failed to generate pipeline")


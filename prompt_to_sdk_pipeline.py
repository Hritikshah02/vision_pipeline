import json
import sys
import os
import re
import logging
import argparse
import subprocess
from typing import List, Dict, Any, Union, Optional
from datetime import datetime
from collections import defaultdict

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

# Import the necessary functions from both files
# You'll need to make sure these imports work in your environment
try:
    # Import from prompt_to_json_5.py
    from prompt_to_json_5 import (
        nl_to_hierarchical_vision_pipeline,
        normalize_prompt,
        PROMPT_TO_PIPELINE
    )
    
    # Import from command_line_pipeline_generator.py
    from command_line_pipeline_generator import generate_pipeline_code

    DIRECT_IMPORTS = True
    logger.info("Successfully imported functions from both script files")
except ImportError as e:
    DIRECT_IMPORTS = False
    logger.warning(f"Could not import functions directly: {e}. Will use embedded functions.")
    
    # Here we would include all the necessary functions from both files
    # For brevity, I'm not including them all here, but in a real implementation
    # you would copy the necessary functions from both files.
    
    # Placeholder for nl_to_hierarchical_vision_pipeline function from prompt_to_json_5.py
    def nl_to_hierarchical_vision_pipeline(natural_language_request, llm_model_path=None, use_cache=True):
        # This is where you'd include the actual implementation from prompt_to_json_5.py
        pass
    
    # Placeholder for generate_pipeline_code function from command_line_pipeline_generator.py
    def generate_pipeline_code(pipeline_json):
        # This is where you'd include the actual implementation from command_line_pipeline_generator.py
        pass

def prompt_to_pipeline(prompt, 
                      model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                      use_cache=True,
                      save_json=True,
                      json_file="input.json",
                      save_python=True,
                      python_file="generated_pipeline.py"):
    """
    Convert a natural language prompt directly to Python pipeline code.
    
    Args:
        prompt (str): Natural language description of the vision pipeline
        model_path (str): Path to the LLM model
        use_cache (bool): Whether to use caching for pipeline generation
        save_json (bool): Whether to save the intermediate JSON
        json_file (str): Path to save the JSON file
        save_python (bool): Whether to save the generated Python code
        python_file (str): Path to save the Python file
        
    Returns:
        tuple: (JSON pipeline, Python code)
    """
    logger.info(f"Processing prompt: '{prompt}'")
    
    # Step 1: Convert natural language to JSON pipeline
    try:
        # Normalize prompt
        normalized_prompt = normalize_prompt(prompt)
        logger.info(f"Normalized prompt: '{normalized_prompt}'")
        
        # Check for exact matches in predefined pipelines
        exact_match_found = False
        for key in PROMPT_TO_PIPELINE.keys():
            norm_key = normalize_prompt(key)
            if normalized_prompt == norm_key:
                logger.info(f"EXACT MATCH found in predefined pipelines: '{key}'")
                exact_match_found = True
                pipeline_json = PROMPT_TO_PIPELINE[key]
                break
        
        if not exact_match_found:
            # Generate pipeline using LLM or rule-based methods
            logger.info(f"No exact match found, generating pipeline...")
            pipeline_json = nl_to_hierarchical_vision_pipeline(prompt, model_path, use_cache)
        
        if not pipeline_json:
            logger.error("Failed to generate pipeline JSON")
            return None, None
        
        # Save JSON to file if requested
        if save_json and json_file:
            try:
                with open(json_file, 'w') as f:
                    json.dump(pipeline_json, f, indent=2)
                logger.info(f"Pipeline JSON saved to '{json_file}'")
            except Exception as e:
                logger.error(f"Error saving JSON: {str(e)}")
        
        # Step 2: Convert JSON pipeline to Python code
        python_code = generate_pipeline_code(pipeline_json)
        
        # Save Python code to file if requested
        if save_python and python_file:
            try:
                with open(python_file, 'w') as f:
                    f.write(python_code)
                logger.info(f"Python code saved to '{python_file}'")
            except Exception as e:
                logger.error(f"Error saving Python code: {str(e)}")
        
        return pipeline_json, python_code
    
    except Exception as e:
        logger.error(f"Error in prompt_to_pipeline: {str(e)}", exc_info=True)
        return None, None

def main():
    """Main function to handle command line arguments and run the pipeline"""
    parser = argparse.ArgumentParser(description='Convert natural language to vision pipeline Python code')
    parser.add_argument('prompt', nargs='?', help='Natural language prompt describing the vision pipeline')
    parser.add_argument('-i', '--input_file', help='File containing the prompt (alternative to providing prompt directly)')
    parser.add_argument('-j', '--json_file', default='input.json', help='Path to save intermediate JSON file')
    parser.add_argument('-o', '--output', default='generated_pipeline.py', help='Output Python file path')
    parser.add_argument('-m', '--model', default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help='Path to LLM model')
    parser.add_argument('--no-cache', action='store_true', help='Disable cache for pipeline generation')
    parser.add_argument('--no-save-json', action='store_true', help='Do not save intermediate JSON file')
    parser.add_argument('--print-json', action='store_true', help='Print the generated JSON to console')
    parser.add_argument('--print-code', action='store_true', help='Print the generated code to console')
    parser.add_argument('--examples', action='store_true', help='Run with preset example prompts')
    
    args = parser.parse_args()
    
    # Get the prompt from either command line or file
    prompt = None
    if args.prompt:
        prompt = args.prompt
    elif args.input_file:
        try:
            with open(args.input_file, 'r') as f:
                prompt = f.read().strip()
        except Exception as e:
            logger.error(f"Error reading prompt file: {str(e)}")
            sys.exit(1)
    elif args.examples:
        # Run with preset examples
        examples = [
            "Detect an object from my prompt then estimate grasp and physical properties for that Object once then estimate live pose for that Object continuously.",
            "detect objects",
            "Detect objects and read text in parallel at level 0, then analyze properties at level 1."
        ]
        
        # Process all examples with a consistent naming scheme
        for i, example in enumerate(examples):
            example_num = i + 1
            print(f"\n--- Example {example_num}: '{example}' ---\n")
            json_file = f"example_{example_num}.json"
            python_file = f"example_{example_num}.py"
            
            pipeline_json, python_code = prompt_to_pipeline(
                example, 
                args.model, 
                not args.no_cache,
                not args.no_save_json,
                json_file,
                True,
                python_file
            )
            
            if args.print_json and pipeline_json:
                print("\nGenerated JSON:")
                print(json.dumps(pipeline_json, indent=2))
            
            if args.print_code and python_code:
                print("\nGenerated Python Code:")
                print(python_code)
        
        # If the user requested specific output files with -j and -o options, also save the last example there
        if not args.no_save_json and args.json_file != "input.json":
            try:
                with open(args.json_file, 'w') as f:
                    json.dump(pipeline_json, f, indent=2)
                print(f"Last example also saved to: {args.json_file}")
            except Exception as e:
                logger.error(f"Error saving to requested JSON file: {str(e)}")
                
        if args.output != "generated_pipeline.py":
            try:
                with open(args.output, 'w') as f:
                    f.write(python_code)
                print(f"Last example also saved to: {args.output}")
            except Exception as e:
                logger.error(f"Error saving to requested Python file: {str(e)}")
        
        return
    else:
        parser.print_help()
        sys.exit(1)
    
    # Process the prompt
    pipeline_json, python_code = prompt_to_pipeline(
        prompt, 
        args.model, 
        not args.no_cache,
        not args.no_save_json,
        args.json_file,
        True,
        args.output
    )
    
    # Print outputs if requested
    if args.print_json and pipeline_json:
        print("\nGenerated JSON:")
        print(json.dumps(pipeline_json, indent=2))
    
    if args.print_code and python_code:
        print("\nGenerated Python Code:")
        print(python_code)
    
    # Success message
    if pipeline_json and python_code:
        print(f"\nSuccessfully generated pipeline from prompt: '{prompt}'")
        if not args.no_save_json:
            print(f"JSON saved to: {args.json_file}")
        print(f"Python code saved to: {args.output}")
    else:
        print("Failed to generate pipeline. Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
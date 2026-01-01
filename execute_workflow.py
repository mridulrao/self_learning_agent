"""
Execute Workflow with Automatic Validation
Validates and normalizes workflow before execution
"""
import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from anthropic import Anthropic
from orchestrator import WorkflowOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('workflow_execution.log')
    ]
)

logger = logging.getLogger(__name__)

load_dotenv()


class WorkflowValidator:
    """Validates and normalizes workflows to match orchestrator expectations"""
    
    # Reference workflow template
    REFERENCE_WORKFLOW = {
        "description": "Search for information, extract content, and save to TextEdit",
        "steps": [
            {
                "id": "step_1",
                "action": "search",
                "args": {"query": "example search query"},
                "agent": "browser",
                "description": "Search on DuckDuckGo",
                "retries": 2,
                "delay_after": 2.0
            },
            {
                "id": "step_2",
                "action": "click_first_result",
                "args": {"result_type": "any"},
                "agent": "browser",
                "description": "Click first search result",
                "retries": 3,
                "delay_after": 3.0
            },
            {
                "id": "step_3",
                "action": "extract_page_content",
                "args": {},
                "agent": "browser",
                "description": "Extract content from the page",
                "retries": 2,
                "delay_after": 1.0
            },
            {
                "id": "step_4",
                "action": "launch_fullscreen",
                "args": {"app_name": "TextEdit"},
                "agent": "desktop",
                "description": "Launch TextEdit",
                "retries": 2,
                "delay_after": 1.0
            },
            {
                "id": "step_5",
                "action": "click_element",
                "args": {"element_description": "New Document button"},
                "agent": "desktop",
                "description": "Create a new document",
                "retries": 3,
                "delay_after": 0.5
            },
            {
                "id": "step_6",
                "action": "type_text",
                "args": {"text_from_ctx": "extracted_content", "fallback_text": "No content found.\n"},
                "agent": "desktop",
                "description": "Type extracted content into document",
                "retries": 1,
                "delay_after": 0.5
            },
            {
                "id": "step_7",
                "action": "hotkey",
                "args": {"keys": ["CMD", "S"]},
                "agent": "desktop",
                "description": "Save (CMD+S)",
                "retries": 1,
                "delay_after": 0.5
            },
            {
                "id": "step_8",
                "action": "type_text",
                "args": {"text": "output.txt"},
                "agent": "desktop",
                "description": "Enter filename",
                "retries": 1,
                "delay_after": 0.3
            },
            {
                "id": "step_9",
                "action": "hotkey",
                "args": {"keys": ["ENTER"]},
                "agent": "desktop",
                "description": "Confirm save",
                "retries": 1,
                "delay_after": 0.5
            }
        ]
    }

    def __init__(self, anthropic_api_key: str):
        """Initialize with Anthropic API key"""
        self.client = Anthropic(api_key=anthropic_api_key)
        
    def validate_and_normalize(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate workflow and return normalized version
        
        Args:
            workflow: Input workflow to validate
            
        Returns:
            Dict with validation results and corrected workflow
        """

        logger.info(f"Validating workflow: {workflow.get('description', 'No description')}")
        
        # Build prompt directly without .format() to avoid issues with curly braces
        prompt = f"""You are a workflow validation expert. Your task is to validate and normalize a workflow to match the expected orchestrator format.

REFERENCE WORKFLOW (The ideal structure):
{json.dumps(self.REFERENCE_WORKFLOW, indent=2)}

INPUT WORKFLOW (To be validated):
{json.dumps(workflow, indent=2)}

VALIDATION RULES:
1. The workflow MUST have exactly 9 steps (no more, no less)
2. Steps must follow this exact sequence:
   - Step 1: search (browser) - Any search query is allowed (make sure to change this query similar to input query so that it can be validated)
   - Step 2: click_first_result (browser)
   - Step 3: extract_page_content (browser) - Generic content extraction (can be any extraction action like extract_page_content, etc.)
   - Step 4: launch_fullscreen (desktop) - TextEdit
   - Step 5: click_element (desktop) - New Document button
   - Step 6: type_text (desktop) - With text_from_ctx parameter for extracted content
   - Step 7: hotkey (desktop) - CMD+S
   - Step 8: type_text (desktop) - filename
   - Step 9: hotkey (desktop) - ENTER

3. Each step must have the correct action, agent, and args structure
4. Customization allowed:
   - Step 1: "query" value in args (any search query)
   - Step 3: "action" can be any extraction action (extract_page_content, extract_text, etc.)
   - Step 3: "args" can have any extraction-specific parameters
   - Step 6: "text_from_ctx" should match what Step 3 produces (e.g., "page_content", "extracted_content", "page_text", etc.)
   - Step 8: "text" value in args (the filename)
   
5. Other parameters (retries, delay_after) should be reasonable values (can vary slightly from reference)
6. Step IDs should be "step_1", "step_2", etc.

TASK:
1. Check if the input workflow follows the 9-step structure
2. If invalid, create a corrected version that:
   - Preserves the search query from the input
   - Preserves the extraction action and parameters from Step 3
   - Ensures Step 6's text_from_ctx matches what Step 3 produces
   - Generates an appropriate filename based on the search query
   - Matches the reference structure for all other steps

CRITICAL OUTPUT FORMAT:
Return ONLY a valid JSON object with these three fields:
- is_valid: boolean (true or false)
- issues: array of strings describing problems found
- corrected_workflow: object with "description" and "steps" array

The corrected_workflow must have a "steps" field that is an ARRAY of step objects.

Do not include any markdown formatting, code blocks, or explanatory text. Just return the raw JSON object.

IMPORTANT:
- Ensure the "steps" field is an ARRAY of step objects, not a string
- If the workflow is already valid, set is_valid to true but still return it as corrected_workflow
- Preserve the search query and extraction logic from the input workflow
- Generate a sensible filename from the search query (lowercase, underscores, .txt extension)
"""
        
        # Call Claude API
        logger.info("Calling Claude API for validation...")
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Extract response text
            response_text = response.content[0].text.strip()
            
            logger.debug(f"Raw API response: {response_text[:500]}...")
            
            # Parse JSON response
            # Remove markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            
            # Validate the result structure
            if not isinstance(result, dict):
                raise ValueError("Result is not a dictionary")
            
            if 'corrected_workflow' not in result:
                raise ValueError("Result missing 'corrected_workflow' field")
            
            corrected = result['corrected_workflow']
            
            if not isinstance(corrected, dict):
                raise ValueError("corrected_workflow is not a dictionary")
            
            if 'steps' not in corrected:
                raise ValueError("corrected_workflow missing 'steps' field")
            
            if not isinstance(corrected['steps'], list):
                logger.error(f"Steps field is not a list, it's: {type(corrected['steps'])}")
                raise ValueError("corrected_workflow.steps is not a list")
            
            if result.get('issues'):
                logger.warning(f"Issues found ({len(result['issues'])}):")
                for i, issue in enumerate(result['issues'], 1):
                    logger.warning(f"  {i}. {issue}")
            else:
                logger.info("No issues found")
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text: {response_text}")
            
            # Return original workflow as fallback
            return {
                "is_valid": False,
                "issues": [f"Validation failed - JSON parse error: {str(e)}"],
                "corrected_workflow": workflow
            }
            
        except Exception as e:
            logger.error(f"Error during validation: {e}", exc_info=True)
            
            # Return original workflow as fallback
            return {
                "is_valid": False,
                "issues": [f"Validation failed: {str(e)}"],
                "corrected_workflow": workflow
            }


def validate_workflow_file(workflow_file: str, api_key: str, 
                           save_corrected: bool = True) -> Dict[str, Any]:
    """
    Validate a workflow file and optionally save the corrected version
    
    Args:
        workflow_file: Path to workflow JSON file
        api_key: Anthropic API key
        save_corrected: If True, save corrected workflow to new file
        
    Returns:
        Corrected workflow dict ready for execution
    """
    # Load workflow
    logger.info(f"Loading workflow from: {workflow_file}")
    with open(workflow_file, 'r') as f:
        workflow = json.load(f)
    
    # Validate and normalize
    validator = WorkflowValidator(api_key)
    result = validator.validate_and_normalize(workflow)
    
    corrected_workflow = result['corrected_workflow']
    
    # Save corrected workflow if requested
    if save_corrected:
        # Generate output filename
        if '_corrected' not in workflow_file:
            corrected_file = workflow_file.replace('.json', '_corrected.json')
        else:
            corrected_file = workflow_file
        
        with open(corrected_file, 'w') as f:
            json.dump(corrected_workflow, f, indent=2)
        
        logger.info(f"Corrected workflow saved to: {corrected_file}")
    
    return corrected_workflow


def execute_validated_workflow(workflow_file: str, 
                               anthropic_api_key: str,
                               browser_headless: bool = False,
                               auto_cleanup: bool = True,
                               stop_on_error: bool = True,
                               save_result: bool = True,
                               skip_validation: bool = False):
    """
    Execute workflow with automatic validation and correction
    
    Args:
        workflow_file: Path to workflow JSON file
        anthropic_api_key: Anthropic API key for validation and execution
        browser_headless: Run browser in headless mode
        auto_cleanup: Automatically cleanup after execution
        stop_on_error: Stop workflow on first error
        save_result: Save execution results
        skip_validation: Skip validation (use original workflow as-is)
    """
    
    # Load original workflow
    logger.info(f"Loading workflow from: {workflow_file}")
    with open(workflow_file, 'r') as f:
        original_workflow = json.load(f)
    
    
    # Validate and normalize workflow (unless skipped)
    if skip_validation:
        logger.warning("Validation skipped - using original workflow")
        workflow = original_workflow
    else:
        
        validator = WorkflowValidator(anthropic_api_key)
        validation_result = validator.validate_and_normalize(original_workflow)
        
        workflow = validation_result['corrected_workflow']
        
        # Verify workflow structure before saving
        if not isinstance(workflow.get('steps'), list):
            logger.error(f"Validation failed - steps is not a list: {type(workflow.get('steps'))}")
            logger.error("Falling back to original workflow")
            workflow = original_workflow
        else:
            # Save corrected workflow
            corrected_file = workflow_file.replace('.json', '_corrected.json')
            with open(corrected_file, 'w') as f:
                json.dump(workflow, f, indent=2)
            logger.info(f"Corrected workflow saved to: {corrected_file}")
            
            if not validation_result['is_valid']:
                logger.warning("Workflow was corrected - using validated version")
            else:
                logger.info("Workflow is valid - proceeding with original")
    
    orchestrator = WorkflowOrchestrator(
        anthropic_api_key=anthropic_api_key,
        browser_headless=browser_headless,
        auto_cleanup=auto_cleanup
    )
    
    logger.info(f"Executing workflow: {workflow.get('description')}")
    logger.info(f"Total steps: {len(workflow.get('steps', []))}")
    
    result = orchestrator.execute_workflow(
        workflow,
        workflow_id='validated_workflow',
        stop_on_error=stop_on_error,
        save_result=save_result
    )
    
    # Display results
    logger.info(f"Status: {result.status}")
    logger.info(f"Completed: {result.completed_steps}/{result.total_steps} steps")
    
    if result.status == "completed":
        logger.info("Workflow completed successfully!")
    
    elif result.status == "failed":
        logger.error(f"Workflow failed at step {result.completed_steps + 1}")
        if hasattr(result, 'error') and result.error:
            logger.error(f"Error: {result.error}")
    
    else:
        logger.warning(f"Workflow ended with status: {result.status}")
    
    return result


def get_latest_recording_folder(base_dir="recordings"):
    """Find the latest recording folder based on creation time"""
    from pathlib import Path
    
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Recordings directory '{base_dir}' not found")
    
    # Get all subdirectories in recordings folder
    folders = [f for f in Path(base_dir).iterdir() if f.is_dir()]
    
    if not folders:
        raise FileNotFoundError(f"No recording folders found in '{base_dir}'")
    
    # Sort by modification time (most recent first)
    latest_folder = max(folders, key=lambda x: x.stat().st_mtime)
    
    return latest_folder


def find_workflow_file(folder_path):
    """Find the workflow file in the given folder"""
    from pathlib import Path
    
    folder = Path(folder_path)
    
    # Look for workflow files in order of preference
    workflow_patterns = [
        "events_workflow_v2_orchestrator.json",
        "events_workflow_v2.json",
        "*_workflow_v2_orchestrator.json",
        "*_workflow_v2.json",
        "*_workflow.json"
    ]
    
    for pattern in workflow_patterns:
        matches = list(folder.glob(pattern))
        if matches:
            return matches[0]
    
    raise FileNotFoundError(f"No workflow file found in {folder}")


def main():
    """Main entry point"""
    
    # Get API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not found in environment")
        logger.error("Please set it in your .env file")
        sys.exit(1)
    
    # Get workflow file from command line or use latest recording
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        workflow_file = sys.argv[1]
    else:
        # Find latest recording folder
        try:
            latest_folder = get_latest_recording_folder()
            workflow_file = find_workflow_file(latest_folder)
            logger.info(f"Using latest recording: {latest_folder.name}")
            logger.info(f"Workflow file: {workflow_file.name}")
        except Exception as e:
            logger.error(f"Error finding latest workflow: {e}")
            sys.exit(1)
    
    # Check if file exists
    if not os.path.exists(workflow_file):
        logger.error(f"Workflow file not found: {workflow_file}")
        sys.exit(1)
    
    # Parse additional arguments
    validate_only = '--validate-only' in sys.argv
    skip_validation = '--skip-validation' in sys.argv
    headless = '--headless' in sys.argv
    
    try:
        if validate_only:
            logger.info("Running validation only (no execution)")
            corrected_workflow = validate_workflow_file(
                str(workflow_file), 
                api_key, 
                save_corrected=True
            )
            logger.info("Validation complete - corrected workflow saved")
        else:
            # Execute with validation
            result = execute_validated_workflow(
                workflow_file=str(workflow_file),
                anthropic_api_key=api_key,
                browser_headless=headless,
                auto_cleanup=True,
                stop_on_error=True,
                save_result=True,
                skip_validation=skip_validation
            )
            
            logger.info("Execution complete")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
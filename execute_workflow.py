import os
from orchestrator import WorkflowOrchestrator
import json
from dotenv import load_dotenv

load_dotenv()


# Load generated workflow
with open('recordings/20251231_192615/events_workflow_orchestrator.json', 'r') as f:
    workflow = json.load(f)

# Execute with orchestrator
orchestrator = WorkflowOrchestrator(
    anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
    browser_headless=False,
    auto_cleanup=True
)

result = orchestrator.execute_workflow(
    workflow,
    workflow_id='recorded_workflow',
    stop_on_error=True,
    save_result=True
)

print(f'Status: {result.status}')
print(f'Completed: {result.completed_steps}/{result.total_steps} steps')
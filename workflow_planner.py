"""
Unified Workflow Planner v2 - Generates improved orchestrator-compatible workflows
"""
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class Step:
    """Single automation step - compatible with WorkflowOrchestrator"""
    id: str
    action: str
    args: Dict[str, Any]
    retries: int = 2
    description: Optional[str] = None
    agent: str = "browser"  # "browser" or "desktop"
    delay_after: float = 0.5  # Delay after step completion (seconds)
    timestamp: Optional[str] = None
    
    def to_dict(self):
        result = asdict(self)
        # Remove None values to keep output clean
        return {k: v for k, v in result.items() if v is not None}


class UnifiedEventAnalyzer:
    """Analyzes events and maintains chronological order"""
    
    # Browser window titles to detect
    BROWSER_WINDOWS = ['Google Chrome', 'Chrome', 'Safari', 'Firefox', 'Edge', 'Brave']
    
    # Browser-specific event types
    BROWSER_EVENT_TYPES = ['navigation', 'page_load']
    
    def __init__(self, events: List[Dict]):
        self.events = sorted(events, key=lambda x: x.get('timestamp', ''))
        self.timeline = self._build_timeline()
    
    def _is_browser_event(self, event: Dict) -> bool:
        """Check if event is a browser event"""
        window_title = event.get('window_title', '')
        event_type = event.get('event_type', '')
        url = event.get('url', '')
        
        # Check window title
        if window_title and any(browser in window_title for browser in self.BROWSER_WINDOWS):
            return True
        
        # Check event type (navigation, page_load are browser-only)
        if event_type in self.BROWSER_EVENT_TYPES:
            return True
        
        # Check if URL exists (only browser events have URLs)
        if url and url != '':
            return True
        
        return False
    
    def _build_timeline(self) -> List[Dict]:
        """Build chronological timeline with categorized events"""
        timeline = []
        
        for event in self.events:
            window_title = event.get('window_title', '')
            app_name = event.get('app_name', '')
            
            # Categorize based on multiple signals
            if self._is_browser_event(event):
                agent = 'browser'
            else:
                agent = 'desktop'
            
            timeline.append({
                'event': event,
                'agent': agent,
                'timestamp': event.get('timestamp', ''),
                'app_name': app_name,
                'window_title': window_title
            })
        
        return timeline
    
    def get_search_query(self, url: str) -> Optional[str]:
        """Extract search query from URL"""
        try:
            if 'google.com/search' in url and 'q=' in url:
                query = url.split('q=')[1].split('&')[0]
                return query.replace('+', ' ')
        except:
            pass
        return None
    
    def get_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return url


class UnifiedWorkflowPlanner:
    """Generates orchestrator-compatible workflows from user activity"""
    
    def __init__(self, events: List[Dict]):
        self.analyzer = UnifiedEventAnalyzer(events)
        self.step_counter = 0
        self.steps = []
        
    def _next_step_id(self) -> str:
        self.step_counter += 1
        return f"step_{self.step_counter}"
    
    def _add_step(self, action: str, args: Dict, agent: str, 
                  description: str = None, retries: int = 2,
                  delay_after: float = 0.5, timestamp: str = None):
        """Add a step to the workflow"""
        step = Step(
            id=self._next_step_id(),
            action=action,
            args=args,
            retries=retries,
            description=description,
            agent=agent,
            delay_after=delay_after,
            timestamp=timestamp
        )
        self.steps.append(step.to_dict())
    
    def analyze_and_plan(self) -> Dict[str, Any]:
        """Analyze events and create unified workflow plan"""
        
        # Track state
        current_window = None
        last_search_query = None
        final_search_query = None  # The query we actually want to use
        text_buffer = []
        in_browser = False
        navigated_to_site = False
        last_url = None
        search_result_clicked = False
        browser_session_ended = False
        desktop_app_launched = False
        
        # Collect all search queries to find the most relevant one
        all_searches = []
        
        # Debug: Count agent types
        browser_events = sum(1 for item in self.analyzer.timeline if item['agent'] == 'browser')
        desktop_events = sum(1 for item in self.analyzer.timeline if item['agent'] == 'desktop')
        
        print(f"\nTimeline Analysis:")
        print(f"  Browser events: {browser_events}")
        print(f"  Desktop events: {desktop_events}")
        
        # First pass: collect all searches and find the most relevant one
        print(f"\nDetecting searches...")
        for item in self.analyzer.timeline:
            event = item['event']
            if item['agent'] == 'browser' and event.get('event_type') == 'navigation':
                url = event.get('url', '')
                if 'google.com/search' in url and 'q=' in url:
                    query = self.analyzer.get_search_query(url)
                    if query and query not in all_searches:
                        all_searches.append(query)
                        print(f"  Found search: '{query}'")
        
        # Pick the last search as the main one (usually most relevant)
        # Skip searches about "Python activity recording" or other meta searches
        skip_keywords = ['python activity', 'recording', 'monitor', 'tracker']
        for search in reversed(all_searches):
            search_lower = search.lower()
            if not any(keyword in search_lower for keyword in skip_keywords):
                final_search_query = search
                break
        
        if not final_search_query and all_searches:
            final_search_query = all_searches[-1]
        
        if final_search_query:
            print(f"\nSelected primary search query: '{final_search_query}'")
        
        print("\nProcessing events...\n")
        
        # Second pass: generate workflow steps
        for item in self.analyzer.timeline:
            event = item['event']
            agent = item['agent']
            event_type = event.get('event_type')
            timestamp = event.get('timestamp')
            window_title = item['window_title']
            
            # BROWSER EVENTS
            if agent == 'browser':
                in_browser = True
                
                # Navigation events
                if event_type == 'navigation':
                    url = event.get('url', '')
                    
                    # Skip empty/blank URLs
                    if not url or 'about:blank' in url or 'about:srcdoc' in url:
                        continue
                    
                    # Skip duplicate URLs
                    if url == last_url:
                        continue
                    last_url = url
                    
                    # Detect search - but only add the final/primary search
                    if 'google.com/search' in url and 'q=' in url:
                        query = self.analyzer.get_search_query(url)
                        
                        # Only add search step if this is our selected primary query
                        # and we haven't already added a search step
                        if query == final_search_query and not last_search_query:
                            last_search_query = query
                            print(f"Adding search step for: '{query}'")
                            
                            self._add_step(
                                action="search",
                                args={"query": query, "engine": "duckduckgo"},
                                agent="browser",
                                description=f"Search for: {query}",
                                retries=2,
                                delay_after=2.0,
                                timestamp=timestamp
                            )
                    
                    # Detect navigation to actual site (not Google)
                    elif last_search_query and 'google.com' not in url and not navigated_to_site:
                        domain = self.analyzer.get_domain(url)
                        
                        # Skip technical/tracking domains
                        skip_domains = ['captcha', 'googletagmanager', 'doubleclick', 'analytics']
                        if domain and not any(skip in domain.lower() for skip in skip_domains):
                            navigated_to_site = True
                            print(f"Detected navigation to: {domain}")
                            
                            # Add improved click step with bot detection handling
                            if not search_result_clicked:
                                self._add_step(
                                    action="click_first_result",
                                    args={
                                        "result_type": "any",
                                        "max_attempts": 5,  # Try up to 5 results
                                        "skip_domains": ["yelp.com", "groupon.com", "tripadvisor.com"]
                                    },
                                    agent="browser",
                                    description="Click first quality search result (skip sponsored/bot-checked sites)",
                                    retries=3,
                                    delay_after=3.0,
                                    timestamp=timestamp
                                )
                                search_result_clicked = True
                            
                            # Add scroll to see content
                            self._add_step(
                                action="scroll",
                                args={"direction": "down", "amount": 300},
                                agent="browser",
                                description="Scroll to view content",
                                retries=1,
                                delay_after=1.0,
                                timestamp=timestamp
                            )
            
            # DESKTOP EVENTS
            elif agent == 'desktop':
                # Window switch/launch (only for non-browser windows)
                if window_title and window_title != current_window:
                    print(f"Window switch: '{current_window}' -> '{window_title}'")
                    
                    # If we were in browser and navigated to a site, extract info before switching
                    if in_browser and navigated_to_site and not browser_session_ended:
                        print(f"Adding extraction step before switching to desktop")
                        self._add_step(
                            action="extract_restaurants_from_html",
                            args={"max_restaurants": 5},
                            agent="browser",
                            description="Extract restaurant information from the page",
                            retries=2,
                            delay_after=1.0,
                            timestamp=timestamp
                        )
                        browser_session_ended = True
                    
                    current_window = window_title
                    in_browser = False
                    
                    # Launch desktop app
                    app_name = event.get('app_name', window_title)
                    
                    # Only launch app once
                    if not desktop_app_launched:
                        self._add_step(
                            action="launch_fullscreen",
                            args={"app_name": app_name},
                            agent="desktop",
                            description=f"Launch {app_name}",
                            retries=2,
                            delay_after=1.5,
                            timestamp=timestamp
                        )
                        desktop_app_launched = True
                        
                        # For text editors, create new document with improved click
                        if any(editor in app_name for editor in ['TextEdit', 'Notes', 'Notepad']):
                            self._add_step(
                                action="click_element",
                                args={
                                    "element_description": "New Document button",
                                    "prefer_position": "bottom"  # IMPROVED: Always pick bottom button
                                },
                                agent="desktop",
                                description="Create new document (click bottom New Document button)",
                                retries=3,
                                delay_after=0.8,
                                timestamp=timestamp
                            )
                
                # Text input - collect for typing step
                elif event_type == 'key_press':
                    key = event.get('key', '')
                    
                    # Skip modifier keys and save commands
                    if key in ['Key.cmd', 'Key.ctrl', 'Key.shift', 'Key.alt']:
                        continue
                        
                    # Regular text character
                    if len(key) == 1:
                        text_buffer.append(key)
                    elif key == 'Key.space':
                        text_buffer.append(' ')
                    elif key == 'Key.enter':
                        text_buffer.append('\n')
                    elif key == 'Key.backspace' and text_buffer:
                        text_buffer.pop()
        
        # Add typing step with context-aware text
        if text_buffer or browser_session_ended:
            # If we extracted restaurants, use formatted output
            if browser_session_ended:
                print(f"Adding typing step with restaurant extraction context")
                self._add_step(
                    action="type_text",
                    args={
                        "text": "{{restaurants_formatted}}"  # Will be replaced by orchestrator
                    },
                    agent="desktop",
                    description="Type extracted restaurant information",
                    retries=1,
                    delay_after=0.5
                )
            # Otherwise use the collected text
            elif text_buffer:
                text = ''.join(text_buffer).strip()
                if text:
                    print(f"Adding typing step with collected text: '{text[:50]}...'")
                    self._add_step(
                        action="type_text",
                        args={"text": text},
                        agent="desktop",
                        description=f"Type text",
                        retries=1,
                        delay_after=0.5
                    )
        
        # Detect save command (CMD+S or CTRL+S)
        save_detected = self._detect_save_command()
        
        if save_detected:
            print("Save command detected")
            self._add_save_workflow(final_search_query or "notes")
        
        # Build final workflow
        workflow = self._build_workflow_output(final_search_query)
        return workflow
    
    def _detect_save_command(self) -> bool:
        """Detect if CMD+S or CTRL+S was pressed"""
        modifier_pressed = False
        
        for i, item in enumerate(self.analyzer.timeline):
            event = item['event']
            if event.get('event_type') == 'key_press':
                key = event.get('key', '')
                
                if key in ['Key.cmd', 'Key.ctrl']:
                    modifier_pressed = True
                elif modifier_pressed and key == 's':
                    return True
                elif key not in ['Key.cmd', 'Key.ctrl', 'Key.shift', 'Key.alt']:
                    modifier_pressed = False
        
        return False
    
    def _add_save_workflow(self, context: str):
        """Add save workflow steps"""
        filename = self._infer_filename(context)
        
        self._add_step(
            action="hotkey",
            args={"keys": ["cmd", "s"]},  # lowercase for compatibility
            agent="desktop",
            description="Save the document (Cmd+S)",
            retries=2,
            delay_after=0.8
        )
        
        self._add_step(
            action="sleep",
            args={"duration": 0.5},
            agent="desktop",
            description="Wait for save dialog",
            retries=1,
            delay_after=0.0
        )
        
        self._add_step(
            action="type_text",
            args={"text": filename},
            agent="desktop",
            description=f"Enter filename: {filename}",
            retries=1,
            delay_after=0.5
        )
        
        self._add_step(
            action="hotkey",
            args={"keys": ["enter"]},  # lowercase for compatibility
            agent="desktop",
            description="Confirm save",
            retries=2,
            delay_after=0.8
        )
        
        # Add file validation step
        # Construct expected file path
        import os
        username = os.path.expanduser("~").split('/')[-1]
        file_path = f"/Users/{username}/Library/Mobile Documents/com~apple~TextEdit/Documents/{filename}"
        
        self._add_step(
            action="validate_file",
            args={
                "file_path": file_path,
                "min_size": 10,
                "expected_content": None  # Optional: could check for keywords
            },
            agent="desktop",
            description=f"Verify file was saved successfully",
            retries=3,
            delay_after=0.5
        )
    
    def _infer_filename(self, context: str) -> str:
        """Infer filename from context"""
        import re
        
        # Clean context text
        context = context.lower().strip()
        
        # Extract keywords
        words = context.split()[:3]  # First 3 words
        if words:
            # Clean and join
            filename = '_'.join(words)
            filename = re.sub(r'[^\w\s-]', '', filename)
            filename = re.sub(r'[-\s]+', '_', filename)
            
            # Add appropriate extension
            return f"{filename}_info.rtf"  # TextEdit default is RTF
        
        return "notes.rtf"
    
    def _build_workflow_output(self, search_query: Optional[str]) -> Dict[str, Any]:
        """Build the final workflow output (orchestrator-compatible format)"""
        
        # Determine user intent
        if search_query:
            intent = f"Search for '{search_query}', extract information, and save to document"
        else:
            intent = "Perform automated task"
        
        # Return orchestrator-compatible format
        workflow = {
            "description": intent,
            "steps": self.steps
        }
        
        return workflow
    
    def get_metadata(self, search_query: Optional[str] = None) -> Dict[str, Any]:
        """Get workflow metadata (separate from orchestrator workflow)"""
        browser_count = sum(1 for s in self.steps if s['agent'] == 'browser')
        desktop_count = sum(1 for s in self.steps if s['agent'] == 'desktop')
        
        return {
            "generated_at": datetime.now().isoformat(),
            "user_intent": search_query or "Perform task",
            "total_steps": len(self.steps),
            "browser_steps": browser_count,
            "desktop_steps": desktop_count,
            "total_events_analyzed": len(self.analyzer.events),
            "sequential": True,
            "mixed_agents": True,
            "improvements": [
                "Uses click_first_quality_result with bot detection",
                "Uses prefer_position for desktop clicks",
                "Filters out meta-searches (e.g., 'python activity recording')",
                "Adds file validation after save",
                "Better context-aware text typing"
            ]
        }


def load_events_from_file(events_file: str) -> List[Dict]:
    """Load events from JSONL file"""
    import os
    
    # Check if file exists
    if not os.path.exists(events_file):
        # Try in uploads directory
        uploads_path = f"/mnt/user-data/uploads/{events_file}"
        if os.path.exists(uploads_path):
            events_file = uploads_path
        else:
            raise FileNotFoundError(f"Events file not found: {events_file}")
    
    # Load events from JSONL file
    events = []
    print(f"Reading events from: {events_file}")
    
    with open(events_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    event = json.loads(line)
                    events.append(event)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
                    continue
    
    if not events:
        raise ValueError("No valid events found in file")
    
    print(f"Loaded {len(events)} events")
    return events


def generate_workflow_from_events(events_file: str, output_file: str = None) -> Dict[str, Any]:
    """
    Generate orchestrator-compatible workflow from events file
    
    Args:
        events_file: Path to events.jsonl file
        output_file: Optional path to save workflow JSON
        
    Returns:
        Orchestrator-compatible workflow dict
    """
    # Load events
    events = load_events_from_file(events_file)
    
    print("\n" + "="*80)
    print("UNIFIED WORKFLOW PLANNER V2 (IMPROVED)")
    print("="*80)
    
    # Create planner and generate workflow
    planner = UnifiedWorkflowPlanner(events)
    workflow = planner.analyze_and_plan()
    
    # Get metadata
    metadata = planner.get_metadata(workflow.get('description', '').split("'")[1] if "'" in workflow.get('description', '') else None)
    
    # Display results
    print("\n" + "="*80)
    print("WORKFLOW METADATA")
    print("-"*80)
    print(f"User Intent: {metadata['user_intent']}")
    print(f"Total Steps: {metadata['total_steps']}")
    print(f"  - Browser Steps: {metadata['browser_steps']}")
    print(f"  - Desktop Steps: {metadata['desktop_steps']}")
    print(f"Events Analyzed: {metadata['total_events_analyzed']}")
    print(f"\nImprovements Applied:")
    for imp in metadata['improvements']:
        print(f"  ✓ {imp}")
    
    print("\n" + "="*80)
    print("GENERATED WORKFLOW (Orchestrator-Compatible)")
    print("="*80)
    print(f"\nDescription: {workflow['description']}")
    print(f"Total Steps: {len(workflow['steps'])}")
    
    for i, step in enumerate(workflow['steps'], 1):
        agent_tag = "[BROWSER]" if step['agent'] == 'browser' else "[DESKTOP]"
        print(f"\n{i}. {agent_tag} {step['action'].upper()}")
        if step.get('description'):
            print(f"   → {step['description']}")
        
        # Pretty print args
        args_str = json.dumps(step['args'], indent=2)
        for line in args_str.split('\n'):
            print(f"   {line}")
        
        print(f"   Retries: {step['retries']}, Delay: {step['delay_after']}s")
    
    print("\n" + "="*80)
    print("WORKFLOW GENERATION COMPLETE")
    print("="*80)
    
    # Save to file if specified
    if output_file:
        # Save full version with metadata
        full_output = {
            "metadata": metadata,
            "workflow": workflow
        }
        
        with open(output_file, 'w') as f:
            json.dump(full_output, f, indent=2)
        print(f"\nFull workflow saved to: {output_file}")
        
        # Also save orchestrator-only version
        orch_output_file = output_file.replace('.json', '_orchestrator.json')
        with open(orch_output_file, 'w') as f:
            json.dump(workflow, f, indent=2)
        print(f"Orchestrator-compatible workflow saved to: {orch_output_file}")
    
    return workflow


def main():
    """CLI entry point"""
    import sys
    
    # Get file path from command line or use default
    if len(sys.argv) > 1:
        events_file = sys.argv[1]
    else:
        events_file = "recordings/20251231_192615/events.jsonl"
    
    # Generate output filename
    output_file = events_file.replace('.jsonl', '_workflow_v2.json')
    
    try:
        workflow = generate_workflow_from_events(events_file, output_file)
        
        print("\n" + "="*80)
        print("USAGE WITH ORCHESTRATOR")
        print("="*80)
        print("\nPython example:")
        print("```python")
        print("from orchestrator import WorkflowOrchestrator")
        print("import json")
        print()
        print(f"# Load generated workflow")
        print(f"with open('{output_file.replace('.json', '_orchestrator.json')}', 'r') as f:")
        print("    workflow = json.load(f)")
        print()
        print("# Execute with orchestrator")
        print("orchestrator = WorkflowOrchestrator(")
        print("    anthropic_api_key='your-api-key',")
        print("    browser_headless=False,")
        print("    desktop_headless=False,  # Set to True for headless mode")
        print("    auto_cleanup=True")
        print(")")
        print()
        print("result = orchestrator.execute_workflow(")
        print("    workflow,")
        print("    workflow_id='recorded_workflow',")
        print("    stop_on_error=True,")
        print("    save_result=True")
        print(")")
        print()
        print("print(f'Status: {result.status}')")
        print("print(f'Completed: {result.completed_steps}/{result.total_steps} steps')")
        print()
        print("# Access extracted data")
        print("if 'restaurants' in result.context.get('vars', {}):")
        print("    restaurants = result.context['vars']['restaurants']")
        print("    print(f'Extracted {len(restaurants)} restaurants')")
        print("```")
        
        return workflow
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
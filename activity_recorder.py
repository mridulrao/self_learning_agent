"""
Comprehensive Activity Recorder
Captures screen recording, browser instrumentation, and OS-level events
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
import threading
import queue

# Screen recording
import cv2
import numpy as np
from PIL import ImageGrab

# Browser instrumentation
from playwright.async_api import async_playwright, Page, Browser

# OS-level event tracking
from pynput import mouse, keyboard
import psutil
import platform

if platform.system() == "Linux":
    try:
        from Xlib import display, X
        from Xlib.error import XError
    except ImportError:
        print("Warning: python-xlib not installed. Window tracking may not work on Linux.")
elif platform.system() == "Windows":
    import win32gui
    import win32process
elif platform.system() == "Darwin":
    try:
        from AppKit import NSWorkspace
    except ImportError:
        print("Warning: pyobjc not installed. Window tracking may not work on macOS.")


class ActivityRecorder:
    def __init__(self, output_dir: str = "./recordings", browser_type: str = "chromium", 
                 browser_channel: str = "chrome", log_browser_events: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Browser configuration - default to Chrome channel for macOS compatibility
        self.browser_type = browser_type  # chromium, firefox, webkit
        self.browser_channel = browser_channel  # chrome, msedge, etc.
        self.log_browser_events = log_browser_events  # Log network/console events?
        
        # Generate session ID
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / self.session_id
        self.session_dir.mkdir(exist_ok=True)
        
        # Recording state
        self.is_recording = False
        self.event_queue = queue.Queue()
        
        # File paths
        self.video_path = self.session_dir / "screen_recording.avi"
        self.events_path = self.session_dir / "events.jsonl"
        self.browser_trace_path = self.session_dir / "browser_trace.zip"
        
        # Components
        self.screen_recorder = None
        self.mouse_listener = None
        self.keyboard_listener = None
        self.browser_context = None
        self.browser = None
        self.playwright = None
        self.page = None
        
    def get_active_window_info(self) -> dict:
        """Get active window title and application name"""
        try:
            if platform.system() == "Windows":
                hwnd = win32gui.GetForegroundWindow()
                window_title = win32gui.GetWindowText(hwnd)
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                try:
                    process = psutil.Process(pid)
                    app_name = process.name()
                except:
                    app_name = "Unknown"
                return {"window_title": window_title, "app_name": app_name}
                
            elif platform.system() == "Darwin":
                workspace = NSWorkspace.sharedWorkspace()
                active_app = workspace.activeApplication()
                return {
                    "window_title": active_app.get("NSApplicationName", "Unknown"),
                    "app_name": active_app.get("NSApplicationName", "Unknown")
                }
                
            elif platform.system() == "Linux":
                d = display.Display()
                window = d.get_input_focus().focus
                wmname = window.get_wm_name()
                wmclass = window.get_wm_class()
                return {
                    "window_title": wmname if wmname else "Unknown",
                    "app_name": wmclass[1] if wmclass else "Unknown"
                }
        except Exception as e:
            return {"window_title": "Error", "app_name": str(e)}
        
        return {"window_title": "Unknown", "app_name": "Unknown"}
    
    def log_event(self, event_type: str, data: dict):
        """Log an event with timestamp"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "time_ms": int(time.time() * 1000),
            "event_type": event_type,
            **data
        }
        self.event_queue.put(event)
    
    def start_screen_recording(self):
        """Start screen recording in a separate thread"""
        def record_screen():
            # Get screen size
            screen = ImageGrab.grab()
            width, height = screen.size
            
            # Video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(
                str(self.video_path),
                fourcc,
                10.0,  # FPS
                (width, height)
            )
            
            print(f"Screen recording started: {width}x{height}")
            
            while self.is_recording:
                # Capture screen
                img = ImageGrab.grab()
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                out.write(frame)
                time.sleep(0.1)  # 10 FPS
            
            out.release()
            print("Screen recording stopped")
        
        self.screen_recorder = threading.Thread(target=record_screen, daemon=True)
        self.screen_recorder.start()
    
    def start_input_tracking(self):
        """Start tracking mouse and keyboard events"""
        def on_mouse_move(x, y):
            window_info = self.get_active_window_info()
            self.log_event("mouse_move", {
                "x": x,
                "y": y,
                **window_info
            })
        
        def on_mouse_click(x, y, button, pressed):
            window_info = self.get_active_window_info()
            self.log_event("mouse_click", {
                "x": x,
                "y": y,
                "button": str(button),
                "pressed": pressed,
                **window_info
            })
        
        def on_mouse_scroll(x, y, dx, dy):
            window_info = self.get_active_window_info()
            self.log_event("mouse_scroll", {
                "x": x,
                "y": y,
                "dx": dx,
                "dy": dy,
                **window_info
            })
        
        def on_key_press(key):
            window_info = self.get_active_window_info()
            try:
                key_char = key.char
            except AttributeError:
                key_char = str(key)
            
            self.log_event("key_press", {
                "key": key_char,
                **window_info
            })
        
        def on_key_release(key):
            window_info = self.get_active_window_info()
            try:
                key_char = key.char
            except AttributeError:
                key_char = str(key)
            
            self.log_event("key_release", {
                "key": key_char,
                **window_info
            })
        
        # Mouse listener
        self.mouse_listener = mouse.Listener(
            on_move=on_mouse_move,
            on_click=on_mouse_click,
            on_scroll=on_mouse_scroll
        )
        self.mouse_listener.start()
        
        # Keyboard listener
        self.keyboard_listener = keyboard.Listener(
            on_press=on_key_press,
            on_release=on_key_release
        )
        self.keyboard_listener.start()
        
        print("Input tracking started")
    
    async def setup_browser_instrumentation(self):
        """Setup Playwright browser with tracing"""
        try:
            self.playwright = await async_playwright().start()
            
            # Get browser type (chromium, firefox, webkit)
            browser_launcher = getattr(self.playwright, self.browser_type)
            
            # Build launch arguments
            launch_args = {
                "headless": False,
                "args": ['--disable-blink-features=AutomationControlled']
            }
            
            # Add channel if specified (e.g., 'chrome' for Google Chrome)
            if self.browser_channel:
                launch_args["channel"] = self.browser_channel
            
            # Launch browser
            self.browser = await browser_launcher.launch(**launch_args)
            
            # Create context with tracing
            self.browser_context = await self.browser.new_context(
                viewport={'width': 1280, 'height': 720},
                ignore_https_errors=True
            )
            
            # Start tracing
            await self.browser_context.tracing.start(
                screenshots=True,
                snapshots=True,
                sources=True
            )
            
            # Setup page instrumentation
            page = await self.browser_context.new_page()
            
            # Store page reference
            self.page = page
            
            # Navigate to DuckDuckGo by default
            await self.page.goto("https://duckduckgo.com", wait_until="domcontentloaded")
            
            # Always log page navigation (important for DSL)
            def sync_log_page_load():
                self.log_event("page_load", {
                    "url": page.url,
                    "title": "Loading..."
                })
            
            page.on("load", sync_log_page_load)
            
            page.on("framenavigated", lambda frame: self.log_event("navigation", {
                "url": frame.url,
                "name": frame.name
            }))
            
            # Only log console/network events if enabled
            if self.log_browser_events:
                # Log console messages
                page.on("console", lambda msg: self.log_event("console", {
                    "type": msg.type,
                    "text": msg.text,
                    "url": page.url
                }))
                
                # Log network requests (only important ones - filter out images, fonts, etc.)
                def log_request(request):
                    # Only log document/xhr/fetch - skip images, fonts, stylesheets, etc.
                    if request.resource_type in ["document", "xhr", "fetch"]:
                        self.log_event("network_request", {
                            "url": request.url,
                            "method": request.method,
                            "resource_type": request.resource_type,
                            "page_url": page.url
                        })
                
                def log_response(response):
                    # Only log document/xhr/fetch responses
                    request = response.request
                    if request.resource_type in ["document", "xhr", "fetch"]:
                        self.log_event("network_response", {
                            "url": response.url,
                            "status": response.status,
                            "page_url": page.url
                        })
                
                page.on("request", log_request)
                page.on("response", log_response)
            
            print(f"Browser instrumentation started ({self.browser_type}" + 
                (f" via {self.browser_channel}" if self.browser_channel else "") + ")")
            return page
            
        except Exception as e:
            print(f"Error in browser setup: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def start_event_writer(self):
        """Start event writer thread"""
        def write_events():
            with open(self.events_path, 'w') as f:
                while self.is_recording:
                    try:
                        event = self.event_queue.get(timeout=0.1)
                        f.write(json.dumps(event) + '\n')
                        f.flush()
                    except queue.Empty:
                        continue
                
                # Drain remaining events
                while not self.event_queue.empty():
                    try:
                        event = self.event_queue.get_nowait()
                        f.write(json.dumps(event) + '\n')
                    except queue.Empty:
                        break
                f.flush()
        
        writer_thread = threading.Thread(target=write_events, daemon=True)
        writer_thread.start()
        return writer_thread
    
    async def start_recording(self, enable_browser: bool = True) -> Optional[Page]:
        """Start all recording components"""
        print(f"\n=== Starting Activity Recording ===")
        print(f"Session ID: {self.session_id}")
        print(f"Output directory: {self.session_dir}")
        
        self.is_recording = True
        
        # Start event writer
        self.event_writer = self.start_event_writer()
        
        # Start screen recording
        self.start_screen_recording()
        
        # Start input tracking
        self.start_input_tracking()
        
        # Setup browser (optional)
        page = None
        if enable_browser:
            try:
                page = await self.setup_browser_instrumentation()
            except Exception as e:
                print(f"Warning: Browser setup failed: {e}")
                print("Continuing without browser instrumentation...")
        
        self.log_event("recording_started", {
            "session_id": self.session_id,
            "platform": platform.system(),
            "browser_enabled": enable_browser and page is not None
        })
        
        print("\n✓ All recording components started")
        print("  - Screen recording")
        print("  - Input tracking (mouse/keyboard)")
        if page:
            print("  - Browser instrumentation")
        print("\nPress Ctrl+C or call stop_recording() to stop\n")
        
        return page
    
    async def stop_recording(self):
        """Stop all recording components"""
        print("\n=== Stopping Activity Recording ===")
        
        self.is_recording = False
        
        # Stop input listeners
        if self.mouse_listener:
            self.mouse_listener.stop()
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        
        # Save browser trace and close browser
        if self.browser_context:
            try:
                await self.browser_context.tracing.stop(path=str(self.browser_trace_path))
                await self.browser_context.close()
            except Exception as e:
                print(f"Warning: Error closing browser context: {e}")
        
        if self.browser:
            try:
                await self.browser.close()
            except Exception as e:
                print(f"Warning: Error closing browser: {e}")
        
        if self.playwright:
            try:
                await self.playwright.stop()
            except Exception as e:
                print(f"Warning: Error stopping playwright: {e}")
        
        # Wait for screen recorder to finish
        if self.screen_recorder:
            self.screen_recorder.join(timeout=2)
        
        # Wait for event writer
        if hasattr(self, 'event_writer'):
            self.event_writer.join(timeout=2)
        
        self.log_event("recording_stopped", {
            "session_id": self.session_id
        })
        
        print(f"\n✓ Recording saved to: {self.session_dir}")
        print(f"  - Video: {self.video_path.name}")
        print(f"  - Events: {self.events_path.name}")
        print(f"  - Browser trace: {self.browser_trace_path.name}")


async def main():
    """Manual recording with unique start/stop markers"""
    # Unique markers that can be filtered out
    START_MARKER = "###WORKFLOW_RECORDING_START###"
    STOP_MARKER = "###WORKFLOW_RECORDING_STOP###"
    
    recorder = ActivityRecorder()
    
    print("\n" + "="*60)
    print("ACTIVITY RECORDER - MANUAL MODE")
    print("="*60)
    print("\nTo START recording, type:")
    print(f"  {START_MARKER}")
    print("\nTo STOP recording, type:")
    print(f"  {STOP_MARKER}")
    print("\nThese markers will be logged and can be filtered out later.")
    print("="*60 + "\n")
    
    # Wait for start command
    while True:
        user_input = input("Waiting for start command: ").strip()
        if user_input == START_MARKER:
            print("\n✓ Start command received!\n")
            break
        else:
            print(f"Invalid command. Please type: {START_MARKER}")
    
    try:
        # Start recording and get browser page
        page = await recorder.start_recording(enable_browser=True)
        
        # Log the start marker
        recorder.log_event("workflow_control", {
            "control_type": "START",
            "marker": START_MARKER,
            "description": "Recording started by user command"
        })
        
        if page:
            print("✓ Browser page available for automation")
            print("  You can now interact with the browser and your system")
        else:
            print("✓ Recording screen and input only")
        
        print(f"\nWhen done, type: {STOP_MARKER}\n")
        
        # Wait for stop command in a non-blocking way
        stop_event = asyncio.Event()
        
        def check_stop_command():
            while not stop_event.is_set():
                try:
                    user_input = input().strip()
                    if user_input == STOP_MARKER:
                        print("\n✓ Stop command received!\n")
                        stop_event.set()
                        break
                    else:
                        print(f"Invalid command. To stop, type: {STOP_MARKER}")
                except EOFError:
                    break
        
        # Run input checker in thread
        input_thread = threading.Thread(target=check_stop_command, daemon=True)
        input_thread.start()
        
        # Wait for stop event
        while not stop_event.is_set():
            await asyncio.sleep(0.1)
        
        # Log the stop marker
        recorder.log_event("workflow_control", {
            "control_type": "STOP",
            "marker": STOP_MARKER,
            "description": "Recording stopped by user command"
        })
        
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by Ctrl+C")
        recorder.log_event("workflow_control", {
            "control_type": "INTERRUPT",
            "description": "Recording interrupted by user (Ctrl+C)"
        })
    except Exception as e:
        print(f"\n❌ Error during recording: {e}")
        import traceback
        traceback.print_exc()
        recorder.log_event("workflow_control", {
            "control_type": "ERROR",
            "description": f"Recording stopped due to error: {str(e)}"
        })
    finally:
        await recorder.stop_recording()
        
        # Print filtering instructions
        print("\n" + "="*60)
        print("WORKFLOW FILTERING")
        print("="*60)
        print("\nTo filter out control commands from events.jsonl:")
        print(f'  grep -v "workflow_control" {recorder.events_path}')
        print("\nOr programmatically filter events where:")
        print(f'  event["event_type"] != "workflow_control"')
        print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
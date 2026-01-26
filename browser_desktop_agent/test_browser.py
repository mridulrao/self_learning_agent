"""
macOS Permission Checker and Browser Workaround
"""
import subprocess
import sys
import platform


def check_macos_permissions():
    """Check if running on macOS and show permission instructions"""
    if platform.system() != "Darwin":
        print("Not running on macOS, skipping permission check")
        return True
    
    print("=" * 70)
    print("macOS SCREEN RECORDING PERMISSION REQUIRED")
    print("=" * 70)
    print()
    print("Playwright/Chromium requires Screen Recording permission on macOS.")
    print("Without it, the browser will crash immediately after launch.")
    print()
    print("TO FIX THIS:")
    print("1. Open System Preferences/Settings")
    print("2. Go to 'Privacy & Security' (or 'Security & Privacy')")
    print("3. Click 'Screen Recording' in the left sidebar")
    print("4. Click the lock icon to make changes (enter password)")
    print("5. Add and enable these apps:")
    print("   - Terminal (if running from terminal)")
    print("   - iTerm (if using iTerm)")
    print("   - Python (the interpreter)")
    print("   - Your IDE (VSCode, PyCharm, etc.)")
    print()
    print("6. After adding, you may need to:")
    print("   - Restart Terminal/IDE")
    print("   - Re-run this script")
    print()
    print("=" * 70)
    print()
    
    response = input("Have you granted Screen Recording permission? (y/n): ").lower()
    return response == 'y'


def try_launch_with_workarounds():
    """Try different browser launch strategies"""
    import asyncio
    from playwright.async_api import async_playwright
    
    strategies = [
        {
            "name": "Strategy 1: Channel-based launch",
            "launch_args": {
                "channel": "chrome",  # Use installed Chrome instead
                "headless": False
            }
        },
        {
            "name": "Strategy 2: Chromium with slow-mo",
            "launch_args": {
                "headless": False,
                "slow_mo": 100,  # Slow down operations
                "args": [
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage'
                ]
            }
        },
        {
            "name": "Strategy 3: Firefox",
            "browser": "firefox",
            "launch_args": {
                "headless": False
            }
        },
        {
            "name": "Strategy 4: WebKit",
            "browser": "webkit",
            "launch_args": {
                "headless": False
            }
        }
    ]
    
    async def test_strategy(strategy):
        playwright = None
        browser = None
        try:
            print(f"\nTesting {strategy['name']}...")
            print("-" * 50)
            
            playwright = await async_playwright().start()
            
            browser_type = getattr(playwright, strategy.get('browser', 'chromium'))
            browser = await browser_type.launch(**strategy['launch_args'])
            
            context = await browser.new_context()
            page = await context.new_page()
            
            await page.goto("https://example.com", timeout=10000)
            title = await page.title()
            
            print(f"✓ SUCCESS! Page title: {title}")
            print(f"✓ {strategy['name']} works!")
            
            await asyncio.sleep(2)
            await context.close()
            await browser.close()
            await playwright.stop()
            
            return True, strategy
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            if browser:
                try:
                    await browser.close()
                except:
                    pass
            if playwright:
                try:
                    await playwright.stop()
                except:
                    pass
            return False, None
    
    async def test_all():
        for strategy in strategies:
            success, working_strategy = await test_strategy(strategy)
            if success:
                print("\n" + "=" * 70)
                print(f"FOUND WORKING SOLUTION: {strategy['name']}")
                print("=" * 70)
                return working_strategy
        
        print("\n" + "=" * 70)
        print("NO WORKING BROWSER FOUND")
        print("=" * 70)
        return None
    
    return asyncio.run(test_all())


if __name__ == "__main__":
    print("macOS Playwright Browser Troubleshooter")
    print()
    
    # Check permissions
    if not check_macos_permissions():
        print("\nPlease grant permissions and try again.")
        print("Exiting...")
        sys.exit(1)
    
    print("\nTrying different browser launch strategies...")
    print("(This may take a minute...)")
    print()
    
    working_strategy = try_launch_with_workarounds()
    
    if working_strategy:
        print("\nRECOMMENDED CONFIGURATION:")
        print("-" * 70)
        print(f"Browser: {working_strategy.get('browser', 'chromium')}")
        print(f"Launch args: {working_strategy['launch_args']}")
        print()
        print("Update your activity_recorder.py with these settings!")
    else:
        print("\nTROUBLESHOOTING STEPS:")
        print("1. Ensure Screen Recording permission is granted")
        print("2. Try installing Google Chrome: brew install --cask google-chrome")
        print("3. Restart your terminal/IDE")
        print("4. Try: playwright install chromium")
        print("5. Consider using the 'no browser' option for now")
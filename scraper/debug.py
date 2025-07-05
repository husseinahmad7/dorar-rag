import asyncio
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import traceback

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.microsoft import EdgeChromiumDriverManager

def check_system_dependencies():
    """
    Check system dependencies for WebDriver
    
    Returns:
        dict: Diagnostic information about system setup
    """
    system_info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": sys.version,
        "edge_installed": False,
        "webdriver_available": False,
        "edge_path": None,
        "edge_version": None
    }
    
    # Check Edge installation
    edge_paths = {
        "Windows": [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"
        ],
        "Darwin": ["/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"],
        "Linux": ["/usr/bin/microsoft-edge", "/usr/bin/edge"]
    }
    
    for path in edge_paths.get(platform.system(), []):
        if os.path.exists(path):
            system_info["edge_installed"] = True
            system_info["edge_path"] = path
            
            # Try to get Edge version
            try:
                if platform.system() == "Windows":
                    version_output = subprocess.check_output([path, "--version"]).decode().strip()
                else:
                    version_output = subprocess.check_output([path, "--version"]).decode().strip()
                system_info["edge_version"] = version_output
            except Exception as e:
                system_info["edge_version_error"] = str(e)
    
    # Check WebDriver availability
    try:
        driver_path = EdgeChromiumDriverManager().install()
        system_info["webdriver_available"] = os.path.exists(driver_path)
        system_info["webdriver_path"] = driver_path
    except Exception as e:
        system_info["webdriver_error"] = str(e)
    
    return system_info

def debug_url_access(url, max_retries=3):
    """
    Comprehensive debugging function to diagnose URL and Selenium access issues
    
    Args:
        url (str): The URL to debug
        max_retries (int): Number of times to retry WebDriver initialization
    
    Returns:
        dict: Detailed debug information
    """
    # First, check system dependencies
    system_info = check_system_dependencies()
    
    debug_info = {
        "url": url,
        "success": False,
        "system_info": system_info,
        "error": None,
        "page_source": None,
        "driver_logs": []
    }
    
    driver = None
    for attempt in range(max_retries):
        try:
            # Reinstall WebDriver on each attempt
            service = Service(EdgeChromiumDriverManager().install())
            options = Options()
            
            # Add more robust WebDriver options
            options.use_chromium = True
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-gpu')
            options.add_experimental_option('excludeSwitches', ['enable-logging'])
            
            driver = webdriver.Edge(service=service, options=options)
            
            # Attempt to navigate to the URL
            print(f"Attempt {attempt + 1}: Accessing URL: {url}")
            driver.get(url)
            
            # Wait for page to load with more flexible condition
            WebDriverWait(driver, 30).until(
                lambda d: d.execute_script('return document.readyState') == 'complete'
            )
            
            # Capture page source
            page_source = driver.page_source
            debug_info.update({
                "success": True,
                "page_source": page_source[:5000],  # Limit page source to first 5000 chars
                "current_url": driver.current_url,
                "title": driver.title
            })
            
            break  # Success, exit retry loop
        
        except Exception as e:
            # Capture detailed error information
            debug_info.update({
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            print(f"Debug Error (Attempt {attempt + 1}): {e}")
        
        finally:
            # Always ensure driver is closed
            if driver:
                driver.quit()
    
    return debug_info

def main():
    # URLs to debug
    urls = [
        "https://hadith.inoor.ir/",
        "https://hadith.inoor.ir/ar/hadith/444"
    ]
    
    # Comprehensive debug results
    comprehensive_debug = {}
    
    for url in urls:
        print(f"\nDebugging URL: {url}")
        debug_result = debug_url_access(url)
        comprehensive_debug[url] = debug_result
    
    # Save debug results to a file
    with open('debug_results.json', 'w', encoding='utf-8') as f:
        json.dump(comprehensive_debug, f, ensure_ascii=False, indent=2)
    
    # Print key information
    print("\n--- Debug Results Summary ---")
    for url, debug_result in comprehensive_debug.items():
        print(f"\nURL: {url}")
        print(f"Success: {debug_result.get('success', False)}")
        if not debug_result.get('success', False):
            print("Error Details:")
            print(debug_result.get('error', 'Unknown error'))
            print("\nTraceback:")
            print(debug_result.get('traceback', 'No traceback available'))

if __name__ == "__main__":
    main()
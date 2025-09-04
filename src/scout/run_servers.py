#!/usr/bin/env python3
"""
Script to run both FastAPI and Gradio servers.
"""
import subprocess
import time
import signal
import sys
import os

def run_fastapi():
    """Start the FastAPI server."""
    return subprocess.Popen([
        "python", "api.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def run_gradio():
    """Start the Gradio server."""
    return subprocess.Popen([
        "python", "chat.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def main():
    print("🚀 Starting Scout with FastAPI backend...")
    
    # Start FastAPI server
    print("📡 Starting FastAPI server on port 8000...")
    fastapi_proc = run_fastapi()
    
    # Wait a moment for FastAPI to start
    time.sleep(3)
    
    # Start Gradio server
    print("🎯 Starting Gradio interface on port 7860...")
    gradio_proc = run_gradio()
    
    print("✅ Both servers started!")
    print("   FastAPI: http://localhost:8000")
    print("   Gradio UI: http://localhost:7860")
    print("\n🛑 Press Ctrl+C to stop both servers")
    
    try:
        # Wait for user interrupt
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
        
        # Terminate processes
        fastapi_proc.terminate()
        gradio_proc.terminate()
        
        # Wait for processes to finish
        fastapi_proc.wait()
        gradio_proc.wait()
        
        print("✅ Servers stopped")

if __name__ == "__main__":
    main()
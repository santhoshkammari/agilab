# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "sse-starlette",
#     "pydantic",
#     "claude-code-sdk",
#     "fastapi[standard]",
# ]
# ///
import asyncio
import json
import uuid
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from logger import get_logger
from ssa import claude_code

# Set up logger
logger = get_logger("api", level="DEBUG")

# Global storage for task states and cancellation tokens
task_store: Dict[str, Dict[str, Any]] = {}
task_cancellation: Dict[str, bool] = {}

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    mode: str = "Scout"
    cwd: Optional[str] = None
    append_system_prompt: str = ""

class TaskResponse(BaseModel):
    task_id: str
    status: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    events: list
    error: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("FastAPI server starting...")
    yield
    # Shutdown
    logger.info("FastAPI server shutting down...")

app = FastAPI(
    title="Scout Chat API",
    description="Background chat processing API with task management",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def process_chat_task(task_id: str, chat_request: ChatRequest):
    """Process chat request and store events in task store."""
    try:
        task_store[task_id]["status"] = "running"
        task_store[task_id]["events"] = []
        task_cancellation[task_id] = False
        
        async for event in claude_code(
            prompt=chat_request.message,
            session_id=chat_request.session_id,
            mode=chat_request.mode,
            cwd=chat_request.cwd if chat_request.cwd else None,
            append_system_prompt=chat_request.append_system_prompt
        ):
            # Check for cancellation
            if task_cancellation.get(task_id, False):
                task_store[task_id]["status"] = "cancelled"
                logger.info(f"Task {task_id} was cancelled")
                return
            
            task_store[task_id]["events"].append(event)
            logger.debug(f"Task {task_id}: Added event {event.get('type', 'unknown')}")
        
        task_store[task_id]["status"] = "completed"
        logger.info(f"Task {task_id} completed successfully")
        
    except Exception as e:
        error_msg = str(e)
        task_store[task_id]["status"] = "failed"
        task_store[task_id]["error"] = error_msg
        logger.error(f"Task {task_id} failed: {error_msg}")
    finally:
        # Clean up cancellation token
        task_cancellation.pop(task_id, None)

@app.post("/chat", response_model=TaskResponse)
async def start_chat(chat_request: ChatRequest, background_tasks: BackgroundTasks):
    """Start a new chat task in the background."""
    task_id = str(uuid.uuid4())
    
    # Initialize task state
    task_store[task_id] = {
        "status": "pending",
        "events": [],
        "request": chat_request.dict(),
        "error": None
    }
    
    # Start background task
    background_tasks.add_task(process_chat_task, task_id, chat_request)
    
    logger.info(f"Started chat task {task_id}")
    return TaskResponse(task_id=task_id, status="pending")

@app.get("/chat/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get the status and events of a chat task."""
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_data = task_store[task_id]
    return TaskStatusResponse(
        task_id=task_id,
        status=task_data["status"],
        events=task_data["events"],
        error=task_data.get("error")
    )

@app.get("/chat/{task_id}/stream")
async def stream_chat_events(task_id: str):
    """Stream chat events using Server-Sent Events."""
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found")
    
    async def event_generator():
        last_event_count = 0
        
        while True:
            task_data = task_store[task_id]
            events = task_data["events"]
            status = task_data["status"]
            
            # Send new events since last check
            if len(events) > last_event_count:
                new_events = events[last_event_count:]
                for event in new_events:
                    yield {
                        "event": "chat_event",
                        "data": json.dumps(event)
                    }
                last_event_count = len(events)
            
            # Send status updates
            yield {
                "event": "status",
                "data": json.dumps({"status": status, "task_id": task_id})
            }
            
            # Stop streaming if task is completed or failed
            if status in ["completed", "failed"]:
                if status == "failed":
                    yield {
                        "event": "error",
                        "data": json.dumps({"error": task_data.get("error", "Unknown error")})
                    }
                yield {
                    "event": "done",
                    "data": json.dumps({"task_id": task_id, "status": status})
                }
                break
            
            # Wait before checking again
            await asyncio.sleep(0.5)
    
    return EventSourceResponse(event_generator())

@app.get("/tasks")
async def list_tasks():
    """List all tasks and their statuses."""
    return {
        task_id: {
            "status": data["status"],
            "error": data.get("error"),
            "event_count": len(data["events"])
        }
        for task_id, data in task_store.items()
    }

@app.post("/chat/{task_id}/stop")
async def stop_task(task_id: str):
    """Stop a running task."""
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_data = task_store[task_id]
    if task_data["status"] not in ["running", "pending"]:
        raise HTTPException(status_code=400, detail=f"Task is {task_data['status']}, cannot stop")
    
    # Set cancellation flag
    task_cancellation[task_id] = True
    logger.info(f"Stop requested for task {task_id}")
    
    return {"message": f"Stop requested for task {task_id}"}

@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a task and free up memory."""
    if task_id not in task_store:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Stop task first if it's still running
    if task_store[task_id]["status"] in ["running", "pending"]:
        task_cancellation[task_id] = True
    
    del task_store[task_id]
    task_cancellation.pop(task_id, None)
    return {"message": f"Task {task_id} deleted"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_tasks": len([t for t in task_store.values() if t["status"] == "running"]),
        "total_tasks": len(task_store)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
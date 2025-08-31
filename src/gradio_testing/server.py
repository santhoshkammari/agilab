from fastapi import FastAPI, WebSocket
import asyncio
import json

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        for i in range(5):
            # Send keepalive messages every 1 second for 20 seconds
            for j in range(20):
                keepalive_msg = {"type": "keepalive", "content": f"keepalive {i}-{j}"}
                await websocket.send_text(json.dumps(keepalive_msg))
                await asyncio.sleep(1)
            
            # Send assistant message after 20 seconds
            assistant_msg = {"type": "assistant", "content": f"Assistant message {i}: Hello from the server!"}
            await websocket.send_text(json.dumps(assistant_msg))
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
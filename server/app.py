import uvicorn
import os
import sys
from fastapi import FastAPI, Request
from collections import OrderedDict

# Fix path so server/app.py can find datacenter_env.py in the root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datacenter_env import DataCenterEnv, DataCenterAction

app = FastAPI()

# Use OrderedDict for LRU Cache behavior
_envs = OrderedDict()
MAX_SESSIONS = 50 

@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "Hyperscale AI Data Center Thermal Controller API is running.",
        "endpoints": ["/reset", "/step", "/state"]
    }

@app.post("/reset")
async def reset(request: Request):
    try:
        body = await request.json()
        task_name = body.get("task_name", "easy") if isinstance(body, dict) else "easy"
        session_id = body.get("session_id", "default") if isinstance(body, dict) else "default"
    except Exception:
        task_name, session_id = "easy", "default"
        
    # Bulletproof Memory Management: Pop the oldest session if we hit the limit
    if len(_envs) >= MAX_SESSIONS:
        _envs.popitem(last=False)
        
    _envs[session_id] = DataCenterEnv()
    
    # Mark this session as the most recently used
    _envs.move_to_end(session_id)
    
    result = _envs[session_id].reset(task_name=task_name)
    return result.model_dump()

@app.post("/step")
async def step(action: DataCenterAction, session_id: str = "default"):
    # Fallback initialization if a step hits an unknown session
    if session_id not in _envs:
        # If we need to create a new one, check memory first
        if len(_envs) >= MAX_SESSIONS:
            _envs.popitem(last=False)
            
        _envs[session_id] = DataCenterEnv()
        _envs[session_id].reset()
        
    # Mark this session as the most recently used
    _envs.move_to_end(session_id)
        
    result = _envs[session_id].step(action)
    return result.model_dump()

@app.get("/state")
async def state(session_id: str = "default"):
    if session_id not in _envs:
        if len(_envs) >= MAX_SESSIONS:
            _envs.popitem(last=False)
            
        _envs[session_id] = DataCenterEnv()
        _envs[session_id].reset()
        
    _envs.move_to_end(session_id)
    return _envs[session_id].state().model_dump()

def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
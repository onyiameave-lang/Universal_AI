import sys
import os
import threading
import json
import io
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

# Add workspace to path for local imports
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(WORKSPACE_DIR)

try:
    from main import (
        step_learn_knowledge,
        step_load_data,
        step_optimize_strategies,
        step_train,
        step_evaluate,
        step_save_results
    )
    from experts.mt5_expert import shutdown_mt5
except ImportError as e:
    print(f"Error importing pipeline steps: {e}")
    # Stubs for safety
    def step_learn_knowledge(*args, **kwargs): pass
    def step_load_data(*args, **kwargs): return {}
    def step_optimize_strategies(*args, **kwargs): pass
    def step_train(*args, **kwargs): return None, None
    def step_evaluate(*args, **kwargs): return {}
    def step_save_results(*args, **kwargs): pass
    def shutdown_mt5(): pass

app = FastAPI(title="MarketOracle Training API")

@app.on_event("shutdown")
def on_shutdown():
    shutdown_mt5()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainingStatus:
    def __init__(self):
        self.lock = threading.Lock()
        self.is_running = False
        self.current_step = "Standby"
        self.progress = 0
        self.logs = []
        self.results = {}
        self.start_time = None
        self.end_time = None

    def update(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def add_log(self, message):
        with self.lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.logs.append(f"[{timestamp}] {message}")
            if len(self.logs) > 1000:
                self.logs.pop(0)

status_state = TrainingStatus()

class PipelineConfig(BaseModel):
    skip_learning: bool = False
    skip_optimization: bool = False
    skip_gemini: bool = False
    timesteps: int = 500000
    episodes: int = 10
    train_ratio: float = 0.7
    mt5: bool = False
    topic: str = "trading_strategy"
    force_refresh: bool = False

class AppLogger(io.StringIO):
    def write(self, s):
        clean_s = s.strip()
        if clean_s:
            status_state.add_log(clean_s)
        return super().write(s)

def run_pipeline_task(config: PipelineConfig):
    status_state.update(
        is_running=True,
        start_time=datetime.now().isoformat(),
        logs=[],
        progress=0,
        current_step="Initializing"
    )
    
    original_stdout = sys.stdout
    sys.stdout = AppLogger()
    
    try:
        print(f"PIPELINE START: {config.topic}")
        
        # Step 1: Learn
        status_state.update(current_step="Knowledge Learning", progress=10)
        if not config.skip_learning:
            step_learn_knowledge(topic=config.topic, force_refresh=config.force_refresh)
        else:
            print("Skipped Step 1: Knowledge Learning")

        # Step 2: Load Data
        status_state.update(current_step="Loading Data", progress=30)
        data_bundle = step_load_data(train_ratio=config.train_ratio, use_mt5=config.mt5)

        # Step 3: Optimize
        status_state.update(current_step="Strategy Optimization", progress=50)
        if not config.skip_optimization:
            step_optimize_strategies(data_bundle, config.skip_gemini)
        else:
            print("Skipped Step 3: Strategy Optimization")

        # Step 4: Train
        status_state.update(current_step="RL Agent Training", progress=70)
        trainer, model = step_train(data_bundle=data_bundle, timesteps=config.timesteps)

        # Step 5: Evaluate
        status_state.update(current_step="Evaluation", progress=90)
        results = step_evaluate(trainer=trainer, model=model, data_bundle=data_bundle, episodes=config.episodes)

        # Step 6: Save
        status_state.update(current_step="Saving Results", progress=95)
        step_save_results(results)
        
        status_state.update(results=results, progress=100, current_step="Finished")
        print("PIPELINE COMPLETE")

    except Exception as e:
        status_state.add_log(f"CRITICAL ERROR: {str(e)}")
        status_state.update(current_step=f"Error: {str(e)}", progress=0)
    finally:
        sys.stdout = original_stdout
        status_state.update(is_running=False, end_time=datetime.now().isoformat())

@app.get("/")
async def read_index():
    return FileResponse(os.path.join(WORKSPACE_DIR, "index.html"))

@app.post("/run")
async def run_pipeline(config: PipelineConfig, background_tasks: BackgroundTasks):
    if status_state.is_running:
        raise HTTPException(status_code=400, detail="Pipeline already running")
    
    background_tasks.add_task(run_pipeline_task, config)
    return {"status": "started"}

@app.get("/status")
async def get_status():
    with status_state.lock:
        return {
            "is_running": status_state.is_running,
            "current_step": status_state.current_step,
            "progress": status_state.progress,
            "start_time": status_state.start_time,
            "end_time": status_state.end_time,
            "results": status_state.results
        }

@app.get("/logs")
async def get_logs(after: int = 0):
    with status_state.lock:
        return {"logs": status_state.logs[after:]}

@app.get("/data")
async def get_data_info():
    data_dir = os.path.join(WORKSPACE_DIR, "data")
    if not os.path.exists(data_dir):
        return {"symbols": []}
    
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    symbols = set()
    for f in files:
        symbols.add(f.split("_")[0])
    
    return {"symbols": list(symbols), "files": files}

@app.post("/reset")
async def reset_system():
    """Manually resets the pipeline state if it gets stuck."""
    with status_state.lock:
        status_state.is_running = False
        status_state.current_step = "Standby"
        status_state.progress = 0
        status_state.results = {}
        status_state.add_log("SYSTEM RESET: Manual intervention triggered.")
    return {"status": "system_reset"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

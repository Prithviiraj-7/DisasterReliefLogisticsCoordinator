from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from server.engine import DisasterReliefEngine


app = FastAPI(title="Disaster Relief Logistics Coordinator", version="1.0.0")
engine = DisasterReliefEngine()
app.mount("/static", StaticFiles(directory="server/static"), name="static")


class ResetRequest(BaseModel):
    task_id: str = Field(default="task_easy_local")


class ActionPayload(BaseModel):
    command: str


class StepRequest(BaseModel):
    action: ActionPayload


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/")
def home() -> FileResponse:
    return FileResponse("server/static/index.html")


@app.post("/reset")
def reset_environment(request: ResetRequest) -> Dict[str, Any]:
    observation = engine.reset(request.task_id)
    return {"observation": observation, "done": False}


@app.post("/step")
def step_environment(request: StepRequest) -> Dict[str, Any]:
    observation, reward, done = engine.step(request.action.command)
    return {"observation": observation, "reward": reward, "done": done}

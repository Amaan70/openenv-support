from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
import uuid
import os
import uvicorn

app = FastAPI(title="Customer Support Ticket Triage")

# ============ Root Endpoint ============
@app.get("/")
async def root():
    return {
        "message": "Customer Support Ticket Triage Environment",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "tasks": "GET /tasks",
            "reset": "GET/POST /reset",
            "step": "POST /step",
            "state": "GET /state"
        }
    }

# ============ Models ============
class Observation(BaseModel):
    ticket_summary: str
    current_ticket: Optional[Dict[str, Any]]
    metrics: Dict[str, float]
    timestamp: str

class Action(BaseModel):
    action_type: Literal["prioritize", "respond", "escalate", "resolve", "reassign"]
    priority: Optional[int] = Field(None, ge=1, le=5)
    response: Optional[str] = None
    assign_to: Optional[str] = None

class Ticket:
    def __init__(self, title, description, customer_type, urgency, complexity):
        self.id = str(uuid.uuid4())
        self.title = title
        self.description = description
        self.customer_type = customer_type
        self.urgency = urgency
        self.complexity = complexity
        self.status = "open"
        self.correct_action = self._determine_action()
        
    def _determine_action(self):
        if self.urgency >= 4 and self.customer_type == "premium":
            return "prioritize"
        elif self.urgency >= 4:
            return "escalate"
        elif self.complexity >= 4:
            return "reassign"
        elif self.urgency <= 2:
            return "resolve"
        else:
            return "respond"

class SupportTicketEnv:
    def __init__(self):
        self.reset()
    
    def reset(self, task_id: str = "support_easy"):
        self.task_id = task_id
        self.current_step = 0
        self.max_steps = 15
        self.total_reward = 0.0
        self.processed_tickets = []
        self.tickets = self._generate_tickets(task_id)
        self.current_ticket = self.tickets.pop(0) if self.tickets else None
        self.score = 0.0
        self.done = False
        return self._get_observation()
    
    def _generate_tickets(self, task_id):
        if task_id == "support_easy":
            return [
                Ticket("Password reset", "User cannot login", "regular", 5, 1),
                Ticket("Billing question", "Invoice discrepancy", "premium", 3, 2),
                Ticket("Feature request", "Add dark mode", "regular", 2, 3),
            ]
        elif task_id == "support_medium":
            return [
                Ticket("Account suspended", "Critical issue", "premium", 5, 4),
                Ticket("Data export", "Need data by EOD", "regular", 4, 3),
                Ticket("Integration error", "API failing", "enterprise", 5, 5),
            ]
        else:
            return [
                Ticket("Security breach", "Potential data leak", "premium", 5, 5),
                Ticket("Legal complaint", "GDPR violation", "enterprise", 5, 5),
                Ticket("System outage", "Production down", "enterprise", 5, 5),
            ]
    
    def _get_observation(self):
        if not self.current_ticket:
            return Observation(
                ticket_summary="No tickets pending",
                current_ticket=None,
                metrics={"score": self.score, "processed": len(self.processed_tickets)},
                timestamp=datetime.now().isoformat()
            )
        return Observation(
            ticket_summary=f"{len(self.tickets)} tickets remaining",
            current_ticket={
                "title": self.current_ticket.title,
                "description": self.current_ticket.description[:200],
                "customer_type": self.current_ticket.customer_type,
                "urgency": self.current_ticket.urgency,
            },
            metrics={"score": self.score, "processed": len(self.processed_tickets)},
            timestamp=datetime.now().isoformat()
        )
    
    def step(self, action: Action):
        if self.done:
            return self._get_observation(), 0.0, True, {"error": "Episode done"}
        
        reward = self._evaluate_action(action)
        self.total_reward += reward
        self.current_step += 1
        
        if self.current_ticket:
            self.processed_tickets.append(self.current_ticket)
        
        if self.tickets:
            self.current_ticket = self.tickets.pop(0)
        else:
            self.current_ticket = None
        
        if self.current_step >= self.max_steps or not self.current_ticket:
            self.done = True
            self.score = self.total_reward / self.max_steps
        
        return self._get_observation(), reward, self.done, {"feedback": "Action processed"}
    
    def _evaluate_action(self, action: Action):
        if not self.current_ticket:
            return 0.0
        
        ticket = self.current_ticket
        reward = 0.0
        
        if action.action_type == ticket.correct_action:
            reward += 0.5
        else:
            reward -= 0.2
        
        reward += min(0.1, len(self.processed_tickets) / self.max_steps)
        
        return max(0.0, min(1.0, reward))
    
    def state(self):
        return {
            "task": self.task_id,
            "step": self.current_step,
            "total_reward": self.total_reward,
            "score": self.score,
            "done": self.done,
            "processed_count": len(self.processed_tickets)
        }

env = SupportTicketEnv()

# ============ API Endpoints ============
@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/reset")
@app.post("/reset")
async def reset(task: str = "support_easy"):
    obs = env.reset(task)
    return {"observation": obs.dict(), "state": env.state()}

@app.post("/step")
async def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info,
        "state": env.state()
    }

@app.get("/state")
async def get_state():
    return env.state()

@app.get("/tasks")
async def list_tasks():
    return {
        "tasks": [
            {"task_id": "support_easy", "difficulty": "easy", "description": "Simple tickets"},
            {"task_id": "support_medium", "difficulty": "medium", "description": "Mixed urgency"},
            {"task_id": "support_hard", "difficulty": "hard", "description": "Critical issues"}
        ]
    }

# ============ Main Entry Point ============
def main():
    """Main entry point for the application."""
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
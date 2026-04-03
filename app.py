from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
import uuid
import os
import json

app = FastAPI(title="Customer Support Ticket Triage")

# ============ Pydantic Models (OpenEnv Spec) ============

class Observation(BaseModel):
    ticket_summary: str = Field(description="Summary of current ticket")
    current_ticket: Optional[Dict[str, Any]] = Field(description="Current ticket details")
    metrics: Dict[str, float] = Field(description="Performance metrics")
    timestamp: str = Field(description="Current time")

class Action(BaseModel):
    action_type: Literal["prioritize", "respond", "escalate", "resolve", "reassign"] = Field(
        description="Action to take on ticket"
    )
    priority: Optional[int] = Field(None, ge=1, le=5, description="Priority 1=Lowest, 5=Highest")
    response: Optional[str] = Field(None, max_length=500)
    assign_to: Optional[str] = Field(None, description="Team to reassign to")

class Ticket:
    def __init__(self, title, description, customer_type, urgency, complexity):
        self.id = str(uuid.uuid4())
        self.title = title
        self.description = description
        self.customer_type = customer_type  # premium, regular, new
        self.urgency = urgency  # 1-5
        self.complexity = complexity  # 1-5
        self.status = "open"
        self.correct_action = self._determine_action()
        self.correct_priority = self.urgency
        
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
                Ticket("Bug report", "Button not working", "premium", 4, 2),
            ]
        elif task_id == "support_medium":
            return [
                Ticket("Account suspended", "Critical issue", "premium", 5, 4),
                Ticket("Data export", "Need data by EOD", "regular", 4, 3),
                Ticket("Integration error", "API failing", "enterprise", 5, 5),
                Ticket("Training request", "New user onboarding", "new", 3, 2),
                Ticket("Billing dispute", "Chargeback threat", "premium", 5, 3),
            ]
        else:  # hard
            return [
                Ticket("Security breach", "Potential data leak", "premium", 5, 5),
                Ticket("Legal complaint", "GDPR violation", "enterprise", 5, 5),
                Ticket("System outage", "Production down", "enterprise", 5, 5),
                Ticket("Contract renewal", "Complex negotiation", "premium", 4, 5),
                Ticket("Executive complaint", "CEO escalation", "premium", 5, 4),
                Ticket("Compliance audit", "Regulatory deadline", "enterprise", 5, 5),
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
            ticket_summary=f"{len(self.tickets)} tickets remaining. Processed: {len(self.processed_tickets)}",
            current_ticket={
                "title": self.current_ticket.title,
                "description": self.current_ticket.description[:200],
                "customer_type": self.current_ticket.customer_type,
                "urgency": self.current_ticket.urgency,
                "complexity": self.current_ticket.complexity
            },
            metrics={
                "score": self.score,
                "processed": len(self.processed_tickets),
                "accuracy": self._calculate_accuracy()
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _calculate_accuracy(self):
        if not self.processed_tickets:
            return 0.0
        correct = sum(1 for t in self.processed_tickets if t.get("correct", False))
        return correct / len(self.processed_tickets)
    
    def step(self, action: Action):
        if self.done:
            return self._get_observation(), 0.0, True, {"error": "Episode done"}
        
        reward, feedback = self._evaluate_action(action)
        self.total_reward += reward
        self.current_step += 1
        
        # Record result
        if self.current_ticket:
            is_correct = reward >= 0.7
            self.processed_tickets.append({
                "ticket_id": self.current_ticket.id,
                "correct": is_correct,
                "reward": reward
            })
        
        # Move to next ticket
        if self.tickets:
            self.current_ticket = self.tickets.pop(0)
        else:
            self.current_ticket = None
        
        # Check episode end
        if self.current_step >= self.max_steps or not self.current_ticket:
            self.done = True
            self.score = self.total_reward / self.max_steps
        
        return self._get_observation(), reward, self.done, {"feedback": feedback}
    
    def _evaluate_action(self, action: Action):
        if not self.current_ticket:
            return 0.0, "No ticket"
        
        ticket = self.current_ticket
        reward = 0.0
        
        # Correct action (40% weight)
        if action.action_type == ticket.correct_action:
            reward += 0.4
        else:
            reward -= 0.1
        
        # Priority accuracy (30% weight)
        if action.action_type == "prioritize" and action.priority:
            diff = abs(action.priority - ticket.correct_priority)
            reward += max(0, 0.3 * (1 - diff/4))
        
        # Response quality (20% weight)
        if action.action_type == "respond" and action.response:
            quality = min(0.2, len(action.response) / 200)
            reward += quality
        
        # Efficiency (10% weight)
        reward += min(0.1, len(self.processed_tickets) / self.max_steps)
        
        # Penalty for wrong escalation
        if action.action_type == "escalate" and ticket.urgency < 4:
            reward -= 0.15
        
        final_reward = max(0.0, min(1.0, reward))
        feedback = f"Action: {action.action_type}, Expected: {ticket.correct_action}, Reward: {final_reward:.2f}"
        
        return final_reward, feedback
    
    def state(self):
        return {
            "task": self.task_id,
            "step": self.current_step,
            "total_reward": self.total_reward,
            "score": self.score,
            "done": self.done,
            "accuracy": self._calculate_accuracy(),
            "processed_count": len(self.processed_tickets)
        }

# Initialize environment
env = SupportTicketEnv()

# ============ FastAPI Endpoints ============

@app.get("/")
async def root():
    return {
        "message": "Customer Support Ticket Triage Environment",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "tasks": "GET /tasks",
            "reset": "GET/POST /reset?task=support_easy",
            "step": "POST /step",
            "state": "GET /state",
            "grader": "GET /grader/{task_id}"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/reset")
@app.post("/reset")
async def reset(task: str = "support_easy"):
    if task not in ["support_easy", "support_medium", "support_hard"]:
        task = "support_easy"
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
            {
                "task_id": "support_easy",
                "description": "Simple support tickets with clear priorities",
                "difficulty": "easy",
                "success_criteria": ">0.8 accuracy",
                "expected_score": 0.85
            },
            {
                "task_id": "support_medium",
                "description": "Mixed urgency with complex scenarios",
                "difficulty": "medium",
                "success_criteria": ">0.7 accuracy",
                "expected_score": 0.75
            },
            {
                "task_id": "support_hard",
                "description": "Critical escalations and complex decisions",
                "difficulty": "hard",
                "success_criteria": ">0.6 accuracy",
                "expected_score": 0.65
            }
        ]
    }

@app.get("/grader/{task_id}")
async def grader(task_id: str):
    if task_id not in ["support_easy", "support_medium", "support_hard"]:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Run evaluation
    env.reset(task_id)
    total_reward = 0
    steps = 0
    
    while not env.done and steps < 15:
        # Simple baseline action
        action = Action(action_type="respond", response="Acknowledged")
        _, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1
    
    score = total_reward / 15 if steps > 0 else 0
    
    return {
        "task_id": task_id,
        "score": round(score, 3),
        "max_score": 1.0,
        "passed": score >= (0.8 if "easy" in task_id else 0.7 if "medium" in task_id else 0.6)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
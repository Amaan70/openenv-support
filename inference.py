#!/usr/bin/env python3
import os
import json
import textwrap
from openai import OpenAI
import requests
from typing import List, Optional

API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8080")
MAX_STEPS = 15

SYSTEM_PROMPT = textwrap.dedent("""
You are a customer support agent. For each ticket, choose the best action:
- prioritize: For urgent issues from premium customers (priority 1-5)
- respond: For regular tickets needing reply
- escalate: For critical issues needing management
- resolve: For simple issues you can close
- reassign: For complex issues needing specialist

Reply with JSON: {"action_type": "...", "priority": 1-5, "response": "...", "assign_to": "..."}
""").strip()

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_str = error if error else "null"
    done_str = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def get_model_action(client: OpenAI, observation: dict, step: int) -> dict:
    prompt = f"""
Step {step}
Ticket: {json.dumps(observation.get('current_ticket', {}), indent=2)}
Choose action as JSON.
"""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150
        )
        text = completion.choices[0].message.content.strip()
        return json.loads(text)
    except Exception as e:
        print(f"Model error: {e}", flush=True)
        return {"action_type": "respond", "response": "Acknowledged"}

def run_task(task: str) -> tuple:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    log_start(task, "support_triage", MODEL_NAME)
    
    # Reset environment
    resp = requests.get(f"{ENV_URL}/reset", params={"task": task})
    data = resp.json()
    obs = data.get("observation", {})
    
    rewards = []
    step = 0
    done = False
    
    while step < MAX_STEPS and not done:
        action = get_model_action(client, obs, step + 1)
        
        resp = requests.post(f"{ENV_URL}/step", json=action)
        step_data = resp.json()
        
        reward = step_data.get("reward", 0.0)
        done = step_data.get("done", False)
        rewards.append(reward)
        
        log_step(step + 1, json.dumps(action), reward, done, None)
        
        obs = step_data.get("observation", {})
        step += 1
    
    score = step_data.get("state", {}).get("score", sum(rewards) / MAX_STEPS)
    success = score >= (0.7 if "easy" in task else 0.6)
    log_end(success, step, score, rewards)
    
    return score, rewards

def main():
    tasks = ["support_easy", "support_medium", "support_hard"]
    results = {}
    
    for task in tasks:
        score, rewards = run_task(task)
        results[task] = {"score": score, "rewards": rewards}
    
    print("\n" + "="*50)
    print("BASELINE RESULTS")
    print("="*50)
    for task, data in results.items():
        print(f"{task}: {data['score']:.3f}")
    
    avg = sum(d['score'] for d in results.values()) / len(results)
    print(f"\nAverage Score: {avg:.3f}")

if __name__ == "__main__":
    main()
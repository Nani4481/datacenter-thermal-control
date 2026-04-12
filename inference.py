import os
import textwrap
import json
import sys
from typing import List, Optional
from openai import OpenAI

try:
    from datacenter_env import DataCenterEnv, DataCenterAction
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from datacenter_env import DataCenterEnv, DataCenterAction

# --- OFFICIAL SUBMISSION CONFIG ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
BENCHMARK = "datacenter-thermal-control"

# --- REFINED PROMPT: Removed hardcoded answers, properly instructs the agent ---
SYSTEM_PROMPT = textwrap.dedent("""
    You are a datacenter AI. You MUST output EXACTLY a valid JSON object and NOTHING else.
    CRITICAL: Do NOT wrap the JSON in markdown blocks. NO comments inside the JSON.

    PRIORITY 1 - HVAC FAILURE & EMERGENCIES:
    Check the 'cooling_status' of every rack. If it says "failed":
    - IMMEDIATELY shift as much workload OUT of the failed racks to safe racks as possible.
    - If the safe racks do not have enough capacity, you MUST use the "throttles" action on the failing racks to reduce their workload and save the hardware.

    PRIORITY 2 - LOAD BALANCING (If Priority 1 is safe):
    - Calculate the average workload of all OPERATIONAL racks.
    - Shift workload from racks with high % to racks with low %.
    
    PRIORITY 3 - PUE REDUCTION:
    - Optimize HVAC cooling_output_percent (e.g. 80.0) to save energy if temps are safe.
    - If an HVAC is 'failed', do NOT attempt to adjust its cooling_output_percent.
    
    JSON SCHEMA EXAMPLE:
    {
      "hvac_adjustments": {"H1": 100.0},
      "workload_shifts": [{"from_rack": "R0", "to_rack": "R2", "amount_percent": 50.0}],
      "throttles": {"R0": 30.0}
    }
""").strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Ensure reward is exactly 2 decimal places 
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # CRITICAL FIX: score must be exactly 2 decimal places to match Regex requirements
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

def summarize_action(action_obj: DataCenterAction) -> str:
    parts = []
    if action_obj.hvac_adjustments:
        parts.append(f"HVAC:{action_obj.hvac_adjustments}")
    if action_obj.workload_shifts:
        shifts = [f"{s.from_rack}->{s.to_rack}:{s.amount_percent}%" 
                  for s in action_obj.workload_shifts]
        parts.append(f"Shifts:[{', '.join(shifts)}]")
    if action_obj.throttles:
        parts.append(f"Throttles:{action_obj.throttles}")
    return " | ".join(parts) if parts else "NoOp"

def get_model_action(client: OpenAI, obs_dict: dict, history: list) -> tuple[DataCenterAction, str]:
    history_str = "\n".join(history[-3:]) if history else "None"
    user_msg = f"Previous Actions:\n{history_str}\n\nCurrent State: {json.dumps(obs_dict)}"
    
    fallback = DataCenterAction(hvac_adjustments={"H0": 80.0, "H1": 80.0}, workload_shifts=[], throttles={})
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.1,
            max_tokens=800
        )
        text = completion.choices[0].message.content.strip()
    except Exception as exc:
        sys.stderr.write(f"\n[DEBUG API] {exc}\n")
        return fallback, str(exc)
    
    text = text.replace("```json", "").replace("```", "").strip()
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1:
        clean_text = text[start:end+1]
    else:
        clean_text = text

    try:
        data = json.loads(clean_text)
        if isinstance(data.get("workload_shifts"), dict):
            data["workload_shifts"] = []
        if not isinstance(data.get("throttles"), dict):
            data["throttles"] = {}
            
        return DataCenterAction(**data), None
    except Exception as exc:
        sys.stderr.write(f"\n[DEBUG JSON PARSE FAILED] Error: {exc}\nModel Output: {clean_text}\n")
        return fallback, None

def run_task(client: OpenAI, task_name: str):
    env = DataCenterEnv() 
    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False
    
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
    
    try:
        result = env.reset(task_name=task_name)
        max_steps = env.max_steps
        
        for step in range(1, max_steps + 1):
            if result.done:
                break
                
            action_obj, err_str = get_model_action(client, result.observation.model_dump(), history)
            result = env.step(action_obj)
            
            reward = result.reward
            done = result.done
            clean_action_str = summarize_action(action_obj)
            
            log_step(step=step, action=clean_action_str, reward=reward, done=done, error=err_str)
            
            rewards.append(reward)
            steps_taken = step
            history.append(f"S{step}: {clean_action_str}")
            
            if done or step == max_steps:
                score = result.info.get("final_grade", 0.0)
                break
                
        success = score >= 0.4
        
    except Exception as e:
        sys.stderr.write(f"\n[DEBUG EXECUTION] {e}\n")
        success = False
        
    if not rewards:
        rewards = [0.0]
        
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

def main():
    if not API_KEY:
        print("WARNING: No HF_TOKEN or API_KEY found in environment. API calls will fail.", flush=True)
        
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task in ["easy", "medium", "hard"]:
        run_task(client, task)

if __name__ == "__main__":
    main()
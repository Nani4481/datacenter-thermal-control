import math
from typing import List, Dict, Literal, Optional
from pydantic import BaseModel, Field

# --- 1. OPENENV INTERFACES (STRICT TYPING) ---

class RackState(BaseModel):
    rack_id: str
    temperature_c: float
    workload_percent: float = Field(..., ge=0.0, le=100.0)
    connected_hvac: str
    cooling_status: Literal["operational", "failed"]

class HVACState(BaseModel):
    hvac_id: str
    cooling_output_percent: float = Field(..., ge=0.0, le=100.0)
    status: Literal["operational", "failed"]

class DataCenterObservation(BaseModel):
    step: int
    racks: List[RackState]
    hvacs: List[HVACState]
    current_pue: float
    total_compute_kw: float
    total_cooling_kw: float
    load_imbalance: float 
    thermal_warnings: int

class WorkloadShift(BaseModel):
    from_rack: str
    to_rack: str
    amount_percent: float = Field(..., ge=0.0, le=100.0)

class DataCenterAction(BaseModel):
    hvac_adjustments: Dict[str, float] = Field(
        default_factory=dict, 
        description="Map of hvac_id to new cooling_output_percent (0-100)"
    )
    workload_shifts: List[WorkloadShift] = Field(
        default_factory=list,
        description="Shift compute load between racks"
    )
    throttles: Dict[str, float] = Field(
        default_factory=dict,
        description="Map of rack_id to throttle amount (reduce workload by X percent)"
    )

class StepResult(BaseModel):
    observation: DataCenterObservation
    reward: float
    done: bool
    info: Dict

# --- 2. ENVIRONMENT & CORE PHYSICS ENGINE ---

class DataCenterEnv:
    def __init__(self):
        self.max_steps = 10
        self.current_step = 0
        self.task_name = "easy"
        self.racks: Dict[str, RackState] = {}
        self.hvacs: Dict[str, HVACState] = {}
        self.episode_pues: List[float] = []
        self.thermal_violations = 0
        self.current_warnings = 0
        self.ambient_temp = 22.0
        self.neighbors = {
            "R0": ["R1", "R2"],
            "R1": ["R0", "R3"],
            "R2": ["R0", "R3"],
            "R3": ["R1", "R2"]
        }

    def reset(self, task_name: str = "easy") -> StepResult:
        self.task_name = task_name
        self.current_step = 0
        self.episode_pues = []
        self.thermal_violations = 0
        self.current_warnings = 0
        self.max_steps = 10
        
        self.hvacs = {
            "H0": HVACState(hvac_id="H0", cooling_output_percent=100.0, status="operational"),
            "H1": HVACState(hvac_id="H1", cooling_output_percent=100.0, status="operational"),
        }

        self.racks = {
            f"R{i}": RackState(
                rack_id=f"R{i}", 
                temperature_c=40.0, 
                workload_percent=50.0,
                connected_hvac="H0" if i in [0, 1] else "H1",
                cooling_status="operational"
            ) 
            for i in range(4)
        }

        if task_name == "medium":
            self.max_steps = 15
            self.racks["R0"].workload_percent = 100.0
            self.racks["R0"].temperature_c = 75.0
            for rid in ["R1", "R2", "R3"]:
                self.racks[rid].workload_percent = 0.0
            self.hvacs["H0"].cooling_output_percent = 50.0
            
        elif task_name == "hard":
            self.hvacs["H0"].status = "failed"
            self.hvacs["H0"].cooling_output_percent = 0.0
            self.racks["R0"].cooling_status = "failed"
            self.racks["R1"].cooling_status = "failed"
            self.racks["R0"].workload_percent = 80.0
            self.racks["R1"].workload_percent = 80.0
            self.racks["R0"].temperature_c = 70.0
            self.racks["R1"].temperature_c = 70.0
            # Removed the artificial R2/R3 cheat. The agent must now figure out how to 
            # shift load into partially full racks, and throttle the rest to survive.

        return StepResult(
            observation=self._get_obs(),
            reward=0.0,
            done=False,
            info={"status": "initialized"}
        )

    def state(self) -> DataCenterObservation:
        return self._get_obs()

    def step(self, action: DataCenterAction) -> StepResult:
        self.current_step += 1
        
        for hid, val in action.hvac_adjustments.items():
            if hid in self.hvacs and self.hvacs[hid].status == "operational":
                self.hvacs[hid].cooling_output_percent = round(max(0.0, min(100.0, val)), 2)
                
        for shift in action.workload_shifts:
            if shift.from_rack in self.racks and shift.to_rack in self.racks:
                moveable = min(self.racks[shift.from_rack].workload_percent, shift.amount_percent)
                capacity_left = 100.0 - self.racks[shift.to_rack].workload_percent
                actual_move = min(moveable, capacity_left)
                
                self.racks[shift.from_rack].workload_percent = round(max(0.0, self.racks[shift.from_rack].workload_percent - actual_move), 2)
                self.racks[shift.to_rack].workload_percent = round(min(100.0, self.racks[shift.to_rack].workload_percent + actual_move), 2)

        for rid, amount in action.throttles.items():
            if rid in self.racks:
                self.racks[rid].workload_percent = round(max(0.0, self.racks[rid].workload_percent - amount), 2)

        warnings = 0
        step_reward = 0.5 # Start at baseline positive reward 
        new_temps = {}

        for rid, rack in self.racks.items():
            heat_gen = rack.workload_percent * 0.4
            cooling = self.hvacs[rack.connected_hvac].cooling_output_percent * 0.5
            
            bleed_effect = 0.0
            for neighbor_id in self.neighbors[rid]:
                temp_diff = self.racks[neighbor_id].temperature_c - rack.temperature_c
                bleed_effect += temp_diff * 0.05
            
            ambient_drift = (rack.temperature_c - self.ambient_temp) * 0.12
            
            new_temp = rack.temperature_c + heat_gen - cooling + bleed_effect - ambient_drift
            new_temps[rid] = round(max(self.ambient_temp, new_temp), 2)

        for rid, temp in new_temps.items():
            self.racks[rid].temperature_c = temp
            if temp >= 90.0:
                self.thermal_violations += 1
                step_reward -= 0.5 # Penalize, but bounds checking handles floor
            elif temp >= 80.0:
                warnings += 1
                step_reward -= 0.1

        self.current_warnings = warnings

        obs = self._get_obs()
        self.episode_pues.append(obs.current_pue)

        if self.task_name == "hard":
            failed_rack_load = (self.racks["R0"].workload_percent + self.racks["R1"].workload_percent)
            evacuation_bonus = max(0.0, (160.0 - failed_rack_load) / 160.0) * 0.5
            step_reward += evacuation_bonus
        else:
            pue_bonus = max(0.0, (1.8 - obs.current_pue) * 0.4) 
            balance_penalty = (obs.load_imbalance / 100.0) * 0.6 
            step_reward += (pue_bonus - balance_penalty)

        done = self.current_step >= self.max_steps
        info = {}
        
        if done:
            info["final_grade"] = self._grade_task(obs)

        # STRICTLY enforce (0.0, 1.0) bounds per hackathon rules
        step_reward = round(max(0.01, min(0.99, step_reward)), 2)

        return StepResult(observation=obs, reward=step_reward, done=done, info=info)

    def _get_obs(self) -> DataCenterObservation:
        total_compute = sum((r.workload_percent / 10.0) for r in self.racks.values()) + 1.0 
        total_cooling = sum(((h.cooling_output_percent / 100.0) ** 3) * 5.0 for h in self.hvacs.values())
        pue = (total_compute + total_cooling) / total_compute if total_compute > 0 else 1.0
        
        active_loads = [r.workload_percent for r in self.racks.values() if r.cooling_status == "operational"]
        imbalance = round(max(active_loads) - min(active_loads), 2) if active_loads else 0.0
        
        return DataCenterObservation(
            step=self.current_step,
            racks=list(self.racks.values()),
            hvacs=list(self.hvacs.values()),
            current_pue=round(pue, 3),
            total_compute_kw=round(total_compute, 2),
            total_cooling_kw=round(total_cooling, 2),
            load_imbalance=imbalance,
            thermal_warnings=self.current_warnings
        )

    def _grade_task(self, obs: DataCenterObservation) -> float:
        avg_pue = sum(self.episode_pues) / len(self.episode_pues) if self.episode_pues else obs.current_pue
        
        if self.thermal_violations > 0:
            return 0.01
            
        if self.task_name == "easy":
            if avg_pue <= 1.25: return 0.99
            if avg_pue <= 1.45: return 0.50
            return 0.10
            
        elif self.task_name == "medium":
            if obs.load_imbalance < 15.0 and avg_pue < 1.65: return 0.99
            if obs.load_imbalance < 40.0: return 0.40
            return 0.10
            
        elif self.task_name == "hard":
            top_load = self.racks["R0"].workload_percent + self.racks["R1"].workload_percent
            top_temp_ok = (self.racks["R0"].temperature_c < 75.0 and self.racks["R1"].temperature_c < 75.0)
            
            if top_load < 5.0 and top_temp_ok: return 0.99
            if top_load < 10.0: return 0.70
            if top_load < 25.0: return 0.50
            return 0.10

        return 0.01
---
title: Hyperscale AI Data Center Thermal Controller
emoji: 🌡️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---

# 🌡️ Hyperscale AI Data Center Thermal & Power Controller
### *Solving the Physical Bottleneck of AI Progress*

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-LRU_Cached-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-2026_Challenge-FF4500)](https://openenv.ai)

---

## 📖 Overview

In the race to train the next generation of Frontier Models — like **Llama-4** or **Gemini-3** — the absolute biggest physical bottleneck isn't silicon. **It's power and cooling.**

Massive GPU clusters consume megawatts of electricity. If GPUs overheat, they undergo **thermal throttling**, killing training performance — or worse, triggering hardware failure worth millions of dollars.

This project provides a **high-fidelity OpenEnv environment** that simulates a 2D grid of server racks. It challenges AI agents to act as *Thermal & Power Controllers*, balancing computational throughput with energy-efficient cooling while navigating hardware failures — a real problem at the frontier of AI infrastructure.

---

## 🏗️ System Architecture

The environment models a **complex, stateful thermodynamic system** with realistic physics. To ensure stability during concurrent automated evaluations, the API backend implements a robust **Least Recently Used (LRU) Cache** for session memory management.

| Physics Layer | Description |
|---|---|
| **Heat Generation** | Proportional to GPU workload (`0.4°C` per `%` load) |
| **Cooling Physics** | Non-linear HVAC efficiency — cooling power scales **cubically** with output |
| **Thermal Bleed** | Heat diffuses between adjacent racks via neighbor-state differential |
| **Ambient Drift** | Natural heating/cooling based on data center ambient temperature |

```text
┌─────────────────────────────────────────────────────────┐
│              DATA CENTER SIMULATION GRID                │
│                                                         │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐│
│   │  RACK R0 │  │  RACK R1 │  │  RACK R2 │  │  RACK R3 ││
│   │ 🔥 85°C  │──│ 🌡️ 72°C  │──│ ✅ 60°C  │──│ ✅ 58°C  ││
│   │  Load:90%│  │  Load:65%│  │  Load:40%│  │  Load:35%││
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘│
│        │             │             │             │      │
│   ┌────▼─────────────▼─────────────▼─────────────▼──────┐ │
│   │          HVAC ZONE H0         HVAC ZONE H1          │ │
│   │          Output: 78%              Output: 45%       │ │
│   └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ OpenEnv Specification

### 1. Observation Space — `DataCenterObservation`

The agent receives a complete telemetry snapshot of the data center at every timestep, strictly typed via Pydantic:

```python
class DataCenterObservation(BaseModel):
    step: int
    racks: List[RackState]          # Per-rack telemetry (temp, load, status)
    hvacs: List[HVACState]          # HVAC zone status (output %, operational status)
    current_pue: float              # Power Usage Effectiveness
    total_compute_kw: float
    total_cooling_kw: float
    load_imbalance: float           # Max-rack load − Min-rack load (%)
    thermal_warnings: int           # Racks exceeding 80°C
```

### 2. Action Space — `DataCenterAction`

The agent controls the physical layer through three primary vectors:

```python
class DataCenterAction(BaseModel):
    hvac_adjustments: Dict[str, float]   # Map of hvac_id to new cooling_output_percent (0-100)
    workload_shifts: List[WorkloadShift] # Shift compute load: {from_rack, to_rack, amount_percent}
    throttles: Dict[str, float]          # Map of rack_id to throttle amount (reduce workload)
```

> ⚠️ `throttles` is a **destructive action**. Reducing workload saves hardware but sacrifices compute SLA. The agent must mathematically deduce when safe racks lack the capacity to absorb failing workloads and trigger throttles to survive.

### 3. Reward Function (Strictly Bounded)

The reward signal is shaped to incentivize efficiency while penalizing risk. Per hackathon rules, all rewards and final grades are strictly clamped within the bounds of `[0.01, 0.99]`.

$$R = \text{Clamp}_{0.01}^{0.99} \left( f(\text{PUE}) - 0.1 \times |\text{Warnings}| - 0.5 \times |\text{Violations}| \right)$$

| Signal | Condition | Impact |
|---|---|---|
| **PUE Bonus** | PUE ≤ 1.25 | Positive scaling |
| **Warning Penalty** | Rack temp 80°C – 89°C | -0.1 per rack |
| **Violation Penalty** | Rack temp ≥ 90°C | -0.5 per rack |

---

## 🎯 Tasks & Difficulty

| # | Task Name | Objective | Core Challenge |
|---|---|---|---|
| 🟢 Easy | Steady State | Achieve PUE ≤ 1.25 | Balance cooling vs. power in a static environment without triggering thermal warnings. |
| 🟡 Medium | Load Surge | Handle 100% load spike on R0 | Rapid workload migration (shifts) to idle racks before thermal violation occurs. |
| 🔴 Hard | HVAC Failure | H0 fails; evacuate R0/R1 | **Capacity Crisis:** Safe racks lack full capacity to absorb the failing workload. The agent MUST logically deduce the need to use destructive throttles alongside shifts to prevent a total meltdown. |

---

## 🚀 Getting Started

### Prerequisites

- Docker installed and running
- Python 3.12+
- **Hugging Face Token** (`HF_TOKEN`): Required to authenticate via the Hugging Face inference router.

### 🐳 Local Development

**Step 1** — Build the container:

```bash
docker build -t datacenter-env .
```

**Step 2** — Run the FastAPI simulation server:

```bash
docker run -p 7860:7860 datacenter-env
```

> The server implements an LRU Cache to safely handle up to **50 concurrent validation sessions** without memory leaks.

**Step 3** — Run the inference agent *(in a new terminal)*:

```bash
export HF_TOKEN="hf_your_token_here"
python inference.py
```

---

## 📊 Baseline Results

Evaluated using Frontier-class reasoning models (Qwen-2.5-72B-Instruct / Llama-3.3-70B):

| Task | Difficulty | Result | Final Score |
|---|---|---|---|
| Steady State | 🟢 Easy | ✅ Success | 0.99 |
| Load Surge | 🟡 Medium | ✅ Success | 0.99 |
| HVAC Failure | 🔴 Hard | ✅ Success | 0.99 |

> **Note:** High-reasoning models successfully utilize the `throttles` fallback action in the Hard task, proving the environment is legitimately solvable by LLMs reacting to state telemetry.

---

## 📁 Project Structure

```plaintext
datacenter-env/
├── Dockerfile               # Container definition & healthchecks
├── openenv.yaml             # Standardized OpenEnv task configuration
├── datacenter_env.py        # Core thermodynamics engine and strict Pydantic schemas
├── inference.py             # Agent entry point with structured prompt engineering
└── server/
    └── app.py               # FastAPI server with memory-safe LRU caching
```

---

## 📜 License

This project is licensed under the [Apache License 2.0](https://opensource.org/licenses/Apache-2.0).

---

<div align="center">

Created for the **Meta x Scaler OpenEnv 2026 Challenge** 🏆

*Pushing the physical limits of AI infrastructure, one thermal cycle at a time.*

</div>
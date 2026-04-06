---
title: Hyperscale AI Data Center Thermal Controller
emoji: 🌡️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
---# 🌡️ Hyperscale AI Data Center Thermal & Power Controller
### *Solving the Physical Bottleneck of AI Progress*

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Spaces-FFD21E)](https://huggingface.co/spaces)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-2026_Challenge-FF4500)](https://openenv.ai)

---

## 📖 Overview

In the race to train the next generation of Frontier Models — like **Llama-4** or **Gemini-3** — the absolute biggest physical bottleneck isn't silicon. **It's power and cooling.**

Massive GPU clusters consume megawatts of electricity. If GPUs overheat, they undergo **thermal throttling**, killing training performance — or worse, triggering hardware failure worth millions of dollars.

This project provides a **high-fidelity OpenEnv environment** that simulates a 2D grid of server racks. It challenges AI agents to act as *Thermal & Power Controllers*, balancing computational throughput with energy-efficient cooling — a real problem at the frontier of AI infrastructure.

---

## 🏗️ System Architecture

The environment models a **complex, stateful thermodynamic system** with realistic physics:

| Physics Layer | Description |
|---|---|
| **Heat Generation** | Proportional to GPU workload (`0.4°C` per `%` load) |
| **Cooling Physics** | Non-linear HVAC efficiency — cooling power scales **cubically** with output |
| **Thermal Bleed** | Heat diffuses between adjacent racks via neighbor-state differential |
| **Ambient Drift** | Natural heating/cooling based on data center ambient temperature |

```
┌─────────────────────────────────────────────────────────┐
│              DATA CENTER SIMULATION GRID                │
│                                                         │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐│
│   │  RACK R0 │  │  RACK R1 │  │  RACK R2 │  │  RACK R3 ││
│   │ 🔥 85°C  │──│ 🌡️ 72°C  │──│ ✅ 60°C  │──│ ✅ 58°C  ││
│   │  Load:90%│  │  Load:65%│  │  Load:40%│  │  Load:35%││
│   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘│
│        │              │              │              │      │
│   ┌────▼─────────────▼──────────────▼──────────────▼────┐ │
│   │          HVAC ZONE H0         HVAC ZONE H1          │ │
│   │         Output: 78%              Output: 45%         │ │
│   └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ OpenEnv Specification

### 1. Observation Space — `DataCenterObservation`

The agent receives a **complete telemetry snapshot** of the data center at every timestep:

```python
@dataclass
class DataCenterObservation:
    racks: List[RackState]          # Per-rack telemetry
    hvacs: List[HVACState]          # HVAC zone status
    pue: float                      # Power Usage Effectiveness
    load_imbalance: float           # Max-rack load − Min-rack load (%)
    thermal_warnings: int           # Racks exceeding 80°C
```

| Field | Type | Description |
|---|---|---|
| `RackState.temperature` | `float` | Current rack temperature (°C) |
| `RackState.workload` | `float` | GPU utilization (0–100%) |
| `RackState.hvac_zone` | `str` | Connected HVAC zone ID |
| `RackState.health` | `str` | `OK` / `WARNING` / `CRITICAL` |
| `HVACState.output` | `float` | Cooling output (0–100%) |
| `HVACState.operational` | `bool` | Failure flag |
| `pue` | `float` | Total Power / Compute Power |

---

### 2. Action Space — `DataCenterAction`

The agent controls the physical layer through **three primary vectors**:

```python
@dataclass
class DataCenterAction:
    hvac_adjustments: Dict[str, float]   # zone_id → output % (0–100)
    workload_shifts:  Dict[str, str]     # source_rack → target_rack
    throttles:        Dict[str, float]   # rack_id → new workload % (destructive)
```

> ⚠️ **`throttles` is a destructive action.** Reducing workload saves hardware but sacrifices compute SLA. Use only in extreme thermal events.

---

### 3. Reward Function

The reward signal is shaped to incentivize **efficiency** while penalizing **risk**:

$$R = \underbrace{f(\text{PUE})}_{\text{Efficiency Bonus}} - \underbrace{0.1 \times |\text{Warning Racks}|}_{\text{Warning Penalty}} - \underbrace{1.0 \times |\text{Critical Racks}|}_{\text{Failure Penalty}}$$

| Signal | Condition | Value |
|---|---|---|
| **PUE Bonus** | PUE ≤ 1.25 | Positive (scales with efficiency) |
| **Warning Penalty** | Rack temp 80°C – 89°C | `−0.1` per rack |
| **Critical Failure** | Rack temp ≥ 90°C | `−1.0` per rack |

---

## 🎯 Tasks & Difficulty

| # | Task Name | Objective | Core Challenge |
|---|---|---|---|
| 🟢 **Easy** | Steady State | Achieve PUE ≤ 1.25 | Balance cooling vs. power in a static environment |
| 🟡 **Medium** | Load Surge — Eliminate Hot Spots | Handle 100% load spike on R0 | Rapid workload migration to idle racks before thermal violation |
| 🔴 **Hard** | HVAC Failure — Emergency Evacuation | H0 fails; evacuate R0/R1 before shutdown | Real-time re-routing with reduced cooling capacity |

---

## 🚀 Getting Started

### Prerequisites

- [Docker](https://www.docker.com/get-started) installed and running
- Python **3.12+**
- OpenAI API Key **or** a local LLM via [Ollama](https://ollama.com/)

---

### 🐳 Local Development (Ollama / Mock Mode)

**Step 1 — Build the container:**
```bash
docker build -t datacenter-env .
```

**Step 2 — Run the simulation server:**
```bash
docker run -p 7860:7860 datacenter-env
```

**Step 3 — Run the inference agent:**
```bash
python inference.py
```

The server will be available at `http://localhost:7860`.

---

### ☁️ Deployment on Hugging Face Spaces

This environment is designed to be hosted on **Hugging Face Spaces** using the Docker SDK.

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Docker** → **Blank** template
3. Push this repository to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/Nani4481/datacenter-thermal-env
   git push space main
   ```
4. The Space will automatically build and deploy at:
   ```
   https://<user>-<space>.hf.space
   ```

---

## 📊 Baseline Results

Evaluated using **Qwen-2.5-72B-Instruct** (Zero-Shot Chain-of-Thought):

| Task | Difficulty | Result | Score |
|---|---|---|---|
| Steady State | 🟢 Easy | ✅ Success | `1.00` |
| Load Surge | 🟡 Medium | ✅ Success | `1.00` |
| HVAC Failure | 🔴 Hard | ✅ Success | `1.00` |

> Results represent zero-shot performance with no fine-tuning or task-specific prompting. Stronger reasoning models are expected to score higher on multi-step emergency scenarios.

---

## 📁 Project Structure

```
datacenter-env/
├── Dockerfile               # Container definition
├── inference.py             # Agent entry point
├── environment/
│   ├── datacenter.py        # Core OpenEnv simulation
│   ├── physics.py           # Thermodynamic models
│   ├── observation.py       # DataCenterObservation schema
│   └── action.py            # DataCenterAction schema
├── agents/
│   └── llm_agent.py         # LLM-based controller
├── tasks/
│   ├── easy.yaml            # Steady State config
│   ├── medium.yaml          # Load Surge config
│   └── hard.yaml            # HVAC Failure config
└── README.md
```

---

## 📜 License

This project is licensed under the **Apache License 2.0**. See [`LICENSE`](./LICENSE) for details.

---

<div align="center">

Created for the **OpenEnv 2026 Challenge** 🏆

*Pushing the physical limits of AI infrastructure, one thermal cycle at a time.*

</div>

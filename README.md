D-HASH Experiment Framework
Redis-based Key Routing Evaluation Suite

Overview
This repository provides a reproducible experimental framework for evaluating key-routing algorithms in a Redis-based sharded environment.
The framework implements and compares several routing strategies under skewed workloads, including:

Consistent Hashing (CH)

Weighted Consistent Hashing (WCH; largest remainder method)

Rendezvous / Highest Random Weight (HRW)

D-HASH (Dynamic Hot-key Aware Scalable Hashing)

All experiments follow a structured multi-stage evaluation methodology commonly used in systems research, covering pipeline selection, microbenchmarking, threshold ablation, skewed workload evaluation, and redistribution behavior under membership changes.

Features

Five-node Redis 7.4.2 cluster deployed via Docker

Deterministic setup with fixed seeds for reproducibility

Zipf workload generation (α = 1.1, 1.3, 1.5)

Pipeline-sweep stage (A1) for selecting optimal pipeline size

Microbench stage (A2) for isolating routing overhead (ns/op)

Threshold ablation (B) for analyzing sensitivity to hot-key detection parameters

Main workload evaluation (C) under skewed distributions

Redistribution report for 5→6 and 6→5 membership transitions

All results written to CSV under the results/ directory

Environment metadata logging for reproducibility

Directory Structure
.
├── docker-compose.yml
├── Dockerfile.runner
├── requirements.txt
├── src/
│ └── dhash_experiments/
│ ├── cli.py
│ ├── stages.py
│ ├── algorithms.py
│ ├── workloads.py
│ ├── bench.py
│ ├── config.py
│ └── utils.py
├── data/
│ ├── nasa_http_logs.log
│ └── ebay_auction_logs.csv
└── results/ (generated after running experiments)

System Requirements

Docker and Docker Compose

Python 3.11 (used inside the runner container)

At least 8 GB RAM recommended for long-running workloads

Host OS tested on Windows 11 + WSL2, should also run on Linux/Mac

Build and Execution

5.1 Build runner image
docker compose build runner

5.2 Start Redis cluster and runner container
docker compose up -d

5.3 Run all experiments
docker compose run --rm runner

5.4 Custom execution
docker compose run --rm runner python -m dhash_experiments.cli --mode <stage> --dataset <dataset>

Experiment Stages

A1. Pipeline Sweep
Evaluates multiple pipeline sizes (B) to determine B* where throughput and tail latency balance.
Each algorithm is tested with identical Zipf-distributed workloads.

A2. Microbench
Measures pure routing overhead by calling get_node() repeatedly without Redis I/O.
Reports cold vs hot phases for D-HASH.

B. Ablation (Threshold T)
Evaluates sensitivity to hot-key threshold values.
Replica count R=2 and window size W fixed.

C. Zipf Main Evaluation
Runs algorithms under α ∈ {1.1, 1.3, 1.5}.
Reports throughput, avg latency, p95, p99, and load standard deviation.

Redistribution
Analyzes key movement ratio (%) when cluster size changes from 5→6 and 6→5 nodes for CH, WCH, HRW.

Output Files
All CSV outputs are placed under results/.
Each stage generates:

{dataset}_pipeline_sweep.csv

{dataset}_microbench_ns.csv

{dataset}_ablation_results.csv

{dataset}_zipf_results.csv

{dataset}_redistribution.csv

{dataset}_*_env_meta.csv (environment metadata)

Reproducibility Notes

Seeds: SEED=1337, PYTHONHASHSEED=123

Redis configuration: appendonly no, save ""

Single-host deployment using internal bridge network

All nodes configured with equal resource limits

Measurement excludes warmup batches

All metrics aggregated as mean ± standard deviation over repeated runs

License
MIT License

Contact
For questions or suggestions, please open an Issue in this repository.
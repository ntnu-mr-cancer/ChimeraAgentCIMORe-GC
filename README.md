# ChimeraAgentCIMORe

CIMORe's submission for the CHIMERA-Agent Challenge (MICCAI 2026).

This repository is based on the official DIAGNijmegen CHIMERA Agent Baseline.

---

## Getting Started

1. Download this repository

```bash
git clone https://github.com/ntnu-mr-cancer/ChimeraAgentCIMORe.git
cd ChimeraAgentCIMORe

```

2. Create a Python Environment and Install Dependencies

```bash
uv venv
source .venv/bin/activate
make install
```


3. Setting up environment variables and huggingface

```bash
cp .env.example .env
```

Inside the .env edit these lines and save:

```bash
HF_TOKEN=ADD YOUR HF TOKEN HERE
HF_HOME=shared/location/for/hf_models
```

And activate the token via this command:

```bash
source .env 
hf auth login --token $HF_TOKEN
```

4. Accept terms and conditions and download models 

Make sure to log into your account and then accept the terms of conditions
[Gemma 4](https://huggingface.co/google/gemma-4-E2B-it) and
[embeddinggemma](https://huggingface.co/google/embeddinggemma-300m):


```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-4-E2B-it', local_dir='model/llm/')"
make fetch-embedding-model    # downloads embeddinggemma-300m (~1.2 GB) into resources/embedding_model/
```

5. Build docke dev container:

```bash
make docker-dev-build
```


Once setup has been completed, all development and inference can be performed through Docker


## Running Baseline Inference


```bash
make docker-dev-run # This will run the configs/config.yaml run for all tasks.
```

Run secific tasks:

```bash
make docker-dev-run RUN_ARGS="agent.tasks=[1,2]"
```

Run on multiple GPUs:

```bash
make docker-dev-run-multi-gpu \
    CUDA_DEVICES=0,1 \
    RUN_ARGS="agent.tasks=[1,2,3]"
```


Predictions are written to:

```text
test/output/task<N>/<case_id>/
```


---

## Running Evaluation

Evaluate a single task:

```bash
make eval TASKS="2" EXPERIMENT_NAME="My_Experiment_Name" USE_RATIONALE_JUDGE=1
```

Evaluate multiple tasks:

```bash
make eval TASKS="1 2 3" EXPERIMENT_NAME="My_Experiment_Name" USE_RATIONALE_JUDGE=1
```

Results are written to:

```text
eval_results/My_Experiment_Name/
```

---

## Useful Docker Commands

Build the development image:

```bash
make docker-dev-build
```

Open an interactive shell:

```bash
make docker-dev-shell
```

Build the Grand Challenge image:

```bash
make gc-build
```

Run the Grand Challenge test workflow:

```bash
make gc-test
```

Export Grand Challenge artifacts:

```bash
make gc-save
```

## Original Documentation

For additional details on the baseline architecture and challenge framework, see:

```text
README_ORIGINAL.md
docs/user-manual.md
docs/architecture.md
docs/models.md
docs/chimera.md
```

---

## Acknowledgements

This work builds upon the official CHIMERA Agent Baseline developed by DIAGNijmegen.

The repository has been adapted and extended by CIMORe (MR Cancer Research Group, NTNU) for participation in the CHIMERA-Agent Challenge (MICCAI 2026).


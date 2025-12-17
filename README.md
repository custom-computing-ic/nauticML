# nauticML

An orchestration framework for co-optimizing neural network models on hardware platforms.

## Installation

### Prerequisites

- Docker
- Conda (or miniconda)
- GPU support (optional, but recommended)

### Setup Steps

1. **Clone the repository**
   ```bash
   % git clone <repository-url>
   % cd nauticML
   ```

2. **Create Conda Development Environment**
   
   To install (just once):
   ```bash
    % conda env create -f environment.yml
    % bash conda/post-install.sh nauticML 
   ```
   
   Once it is created:
   ```bash
   conda activate nauticml
   ```   

3. **Start Prefect and PostgreSQL services**
   
   Use Docker Compose to launch the Prefect server and PostgreSQL database:
   ```bash
   docker compose up -d
   ```

   This will start:
   - **PostgreSQL Database** (port not exposed, internal use only)
   - **Prefect Server** (accessible at http://localhost:4200)



4. **Configure your experiment**
   
   Edit `config.yaml` to set up your experiment parameters:
   - Dataset configuration
   - Model architecture and hyperparameters
   - Bayesian optimization search space
   - Training and evaluation settings

5. **Run the optimization pipeline**
   ```bash
   python run.py
   ```

   This will:
   - Initialize the experiment
   - Load the dataset
   - Run Bayesian optimization iterations
   - Build and train Bayesian models
   - Evaluate model performance
   - Track results in Prefect

## Configuration

The `config.yaml` file controls all experiment parameters:

```yaml
experiment:
  seed: 42                    # Random seed for reproducibility
  save_dir: ./results         # Directory for saving results
  gpus: [0]                   # GPU device IDs to use

dataset:
  name: mnist                 # Dataset name

model:
  name: lenet                 # Model architecture
  is_quant: false            # Enable quantization
  dropout_rate: 0.4          # Dropout rate
  num_bayes_layer: 3         # Number of Bayesian layers

bayes_opt:
  num_iter: 3                # Number of optimization iterations
  tunable:                   # Hyperparameters to optimize
    dropout_rate:
      space: [0.1, 0.2, 0.3, 0.4]
    p_rate:
      space: [0.0, 0.1, 0.2]
    num_bayes_layer:
      space: [1, 2, 3]
```

## Running Experiments

### Local Execution

```bash
python run.py
```

### Monitor with Prefect UI

After starting Docker Compose, access the Prefect UI at:
```
http://localhost:4200
```

The UI provides:
- Real-time flow execution monitoring
- Task logs and performance metrics
- Experiment history and results

### Results

Results are saved to the configured `save_dir` (default: `./results`):
- Model checkpoints
- Training logs
- Evaluation metrics
- Optimization summaries

## Project Structure

```
nauticML/
├── nautic/                 # Core framework
│   ├── engine.py          # Main orchestration engine
│   ├── flowx.py           # Prefect flow decorators
│   ├── taskx.py           # Task definitions
│   └── context.py         # Experiment context
├── logic/                  # Model and data handling
│   ├── datasets.py        # Dataset loaders
│   ├── pipeline.py        # Data pipelines
│   ├── models/            # Model definitions
│   └── models2/           # Quantized and Bayesian models
├── config.yaml            # Experiment configuration
├── run.py                 # Main entry point
├── docker-compose.yaml    # Service definitions
└── requirements.txt       # Python dependencies
```

## GPU Support

To use GPU acceleration:

1. Ensure CUDA-compatible GPU is available
2. Configure GPU devices in `config.yaml`:
   ```yaml
   experiment:
     gpus: [0, 1]  # Use GPU 0 and 1
   ```

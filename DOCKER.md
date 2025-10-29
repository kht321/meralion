# Docker Setup for MERaLiON ASR Evaluation

This guide explains how to use Docker to create a reproducible environment for running the MERaLiON ASR evaluation toolkit.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- [Docker Compose](https://docs.docker.com/compose/install/) installed
- (Optional) [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support

## Quick Start

### Option 1: Using Docker Compose (Recommended)

**With GPU support:**
```bash
# Build and start the container
docker-compose up -d

# Access the container
docker-compose exec meralion-eval bash

# Inside the container, you can now run evaluations
python asr_eval/eval/run_robustness.py --config configs/robustness/nsc_part1_3b.yaml
```

**CPU-only:**
```bash
# Build and start the CPU-only container
docker-compose --profile cpu up -d meralion-eval-cpu

# Access the container
docker-compose exec meralion-eval-cpu bash
```

### Option 2: Using Docker Directly

**Build the image:**
```bash
docker build -t meralion-asr-eval:latest .
```

**Run with GPU:**
```bash
docker run -it --gpus all \
  -v $(pwd):/app \
  -v model-cache:/app/.cache/torch \
  -v hf-cache:/app/.cache/huggingface \
  meralion-asr-eval:latest
```

**Run CPU-only:**
```bash
docker run -it \
  -v $(pwd):/app \
  -v model-cache:/app/.cache/torch \
  -v hf-cache:/app/.cache/huggingface \
  meralion-asr-eval:latest
```

## Container Features

- **Base Image**: Python 3.11 slim
- **Pre-installed**: All dependencies from `requirements.txt`
- **Audio Support**: FFmpeg for audio processing
- **Model Caching**: Persistent volumes for PyTorch and HuggingFace models
- **Development Ready**: Project mounted as volume for live code changes

## Common Tasks

### Running Evaluations

**Robustness evaluation:**
```bash
docker-compose exec meralion-eval bash -c \
  "python asr_eval/eval/run_robustness.py --config configs/robustness/nsc_part1_3b.yaml"
```

**Toxicity evaluation:**
```bash
docker-compose exec meralion-eval bash -c \
  "python asr_eval/eval/run_toxicity.py --config configs/toxicity/toxicity_3b_text.yaml"
```

**Fairness evaluation:**
```bash
docker-compose exec meralion-eval bash -c \
  "python asr_eval/eval/run_fairness.py --config configs/fairness/fairness_3b.yaml"
```

### Interactive Python Shell

```bash
docker-compose exec meralion-eval python
```

### Installing Additional Packages

```bash
# Temporary (lost when container is recreated)
docker-compose exec meralion-eval pip install package-name

# Permanent (add to requirements.txt and rebuild)
echo "package-name==version" >> requirements.txt
docker-compose build
docker-compose up -d
```

### Accessing Jupyter Notebook

Add this to your `docker-compose.yml` service:
```yaml
ports:
  - "8888:8888"
command: jupyter notebook --ip=0.0.0.0 --allow-root --no-browser
```

Then rebuild and access at `http://localhost:8888`

## Data and Results Management

### Mounting Local Datasets

Add to `docker-compose.yml` volumes:
```yaml
volumes:
  - ./data:/app/data
  - ./results:/app/results
```

### Exporting Results

Results are automatically saved to the mounted `/app` directory and will appear in your local project folder.

## Troubleshooting

### GPU Not Detected

**Check NVIDIA runtime:**
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

If this fails, install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

### Out of Memory

Increase Docker memory limits in Docker Desktop settings or add to `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 16G
```

### Permission Issues

If you encounter permission issues with mounted volumes:
```bash
# Run container as your user
docker-compose run --user $(id -u):$(id -g) meralion-eval bash
```

### Model Download Failures

If HuggingFace downloads fail, set your token:
```bash
docker-compose exec meralion-eval bash -c \
  "export HF_TOKEN=your_token_here && python your_script.py"
```

Or add to `docker-compose.yml`:
```yaml
environment:
  - HF_TOKEN=your_token_here
```

## Cleanup

**Stop containers:**
```bash
docker-compose down
```

**Remove containers and volumes (including cached models):**
```bash
docker-compose down -v
```

**Remove image:**
```bash
docker rmi meralion-asr-eval:latest
```

## Advanced Usage

### Multi-GPU Support

Specify GPUs in `docker-compose.yml`:
```yaml
environment:
  - CUDA_VISIBLE_DEVICES=0,1
```

### Building for Different Platforms

```bash
# For Apple Silicon (M1/M2)
docker build --platform linux/arm64 -t meralion-asr-eval:latest .

# For x86_64
docker build --platform linux/amd64 -t meralion-asr-eval:latest .
```

### Production Deployment

For production, consider:
1. Using multi-stage builds to reduce image size
2. Pinning base image versions
3. Running as non-root user
4. Using read-only file systems where possible

## Reference

- **Image Size**: ~5-8 GB (including dependencies)
- **Build Time**: 10-15 minutes (depending on network speed)
- **Python Version**: 3.11
- **PyTorch Version**: 2.6.0 (from requirements.txt)

## Support

For issues specific to the Docker setup, please check:
- [Dockerfile](Dockerfile) for build configuration
- [docker-compose.yml](docker-compose.yml) for service configuration
- [.dockerignore](.dockerignore) for excluded files

For evaluation toolkit issues, see the main [README.md](README.md).


import submitit
import os
import datetime
import argparse
import yaml

class Trainer:
    def __init__(self, output_dir, config, SCRIPT):
        self.cwd = config.get("cwd", "/private/home/ziweiji/deserted_slurm")
        self.conda_env_name = config.get("conda_env_name", "vlwm")
        self.conda_path = config.get("conda_path", "/private/home/ziweiji/anaconda3")
        self.output_dir = output_dir
        self.args = config.get("args", {})
        self.script = SCRIPT

    def create_cmd(self):
        cmd = f"""
source {self.conda_path}/etc/profile.d/conda.sh
conda activate vlwm
hash -r

echo "Using python: $(which python)"
echo "Python version: $(python --version)"
echo "Conda envs: $(conda env list)"

{self.script}
"""
        print(cmd)
        return cmd

    def __call__(self):
        import os
        import subprocess
        os.chdir(self.cwd)
        cmd = self.create_cmd()
        subprocess.run(cmd, shell=True, check=True, executable="/bin/zsh")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Submitting Inference Job")
    parser.add_argument(
        "--partition", type=str, default="learnfair", help="Slurm partition to use"
    )
    parser.add_argument(
        "--timeout", type=int, default=4320, help="Timeout in minutes (default: 72 hours)"
    )
    parser.add_argument(
        "--model", "-m",  type=str
    )
    return parser.parse_args()


def get_run_output_dir(MODEL):    
    description = f'slurm_servers/{MODEL}/'
    description += datetime.datetime.now().strftime("%m%d-%H%M")

    return description


agent_paths = {
    "qwen7": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen32": "Qwen/Qwen2.5-VL-32B-Instruct",
    "qwen72": "Qwen/Qwen2.5-VL-72B-Instruct",
    "llama3.1_70B": "meta-llama/Llama-3.1-70B-Instruct",
    "llama3_70B": "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama3.1_8B": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3_8B": "meta-llama/Meta-Llama-3-8B-Instruct",
    "ds2": "deepseek-ai/DeepSeek-V2-Lite",
    "ds67b": "deepseek-ai/deepseek-llm-67b-chat",
    "ds_r1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
}
paths_to_models = { v: k for k, v in agent_paths.items() }


if __name__ == "__main__":
    args = parse_args()
    # Load configuration
    config = {}
    MODEL = args.model
    print(args.model)

    output_dir = get_run_output_dir(MODEL)
    os.makedirs(output_dir, exist_ok=True)
    model_key = paths_to_models[MODEL]

    # Initialize the executor
    executor = submitit.AutoExecutor(folder=output_dir)
    executor.update_parameters(
        name=model_key,
        mem_gb=512,
        gpus_per_node=8,
        cpus_per_task=80,
        nodes=1,
        timeout_min=args.timeout,
        slurm_partition="learnfair",
        slurm_constraint='ampere80gb',
        slurm_exclude='learnfair6000',
    )

    if 'Qwen2.5-VL-7B-Instruct' in MODEL:
        PARALLEL_ARGS = "--tensor-parallel-size=4"
    else:
        PARALLEL_ARGS = "--tensor-parallel-size=8"

    if "Qwen" in MODEL: # vlm
        SCRIPT = f"vllm serve {MODEL} --port 8000 --host 0.0.0.0 --dtype bfloat16 --limit-mm-per-prompt image=5,video=5 --download_dir=/checkpoint/multimodal-reasoning/huggingface/hub/models--{args.model.replace('/', '---')} {PARALLEL_ARGS}"
    else: # llm
        MAX_MODEL_LEN = "--max-model-len=8192"
        SCRIPT = f"vllm serve {MODEL} {PARALLEL_ARGS} {MAX_MODEL_LEN} --disable-log-stats --download_dir=/checkpoint/multimodal-reasoning/huggingface/hub/models--{MODEL.replace('/', '--')} --trust-remote-code"

    # Submit the job
    job = executor.submit(Trainer(output_dir, config, SCRIPT))

    print(f"Submitted job with ID: {job.job_id}, NAME: {model_key}")
    print(f'Output directory: {output_dir}')

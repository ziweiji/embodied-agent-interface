import os
import json
import argparse
root_path = os.path.dirname(os.path.abspath(__file__))
import openai
import subprocess
from tqdm.contrib.concurrent import thread_map
from tqdm import tqdm
import sys
def get_available_servers():
    agent_paths = {
        "llama3.1_70B": "meta-llama/Llama-3.1-70B-Instruct",
        "llama3_70B": "meta-llama/Meta-Llama-3-70B-Instruct",
        "llama3.1_8B": "meta-llama/Llama-3.1-8B-Instruct",
        "llama3_8B": "meta-llama/Meta-Llama-3-8B-Instruct",
        "ds2": "deepseek-ai/DeepSeek-V2-Lite",
        "ds_r1": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "qwen32": "Qwen/Qwen2.5-VL-32B-Instruct",
    }

    # Run squeue and capture output
    result = subprocess.run(['squeue', '--me', '-o', '"%j, %N, %T, %i"'], capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')

    # initialize server dict
    server_dict = {}
    
    # Iterate over each line, skipping the header
    for line in lines[1:]:
        line = line.strip('\"')
        # Get job name, nodelist, and status
        job_name, nodelist, status, job_id = line.split(', ')

        assert "[" not in nodelist, "Multi-node servers not currently supported."

        # keep only running jobs
        if status == "RUNNING" and job_name != "bash":
            try: 
                model_path = agent_paths[job_name]
                server_address = f"http://{nodelist}:8000/v1"

    
                if model_path in server_dict:
                    server_dict[model_path]["server_urls"].append(server_address)
                    server_dict[model_path]["job_ids"].append(job_id)
                else:
                    server_dict[model_path] = {"name": job_name, "server_urls": [server_address], "job_ids": [job_id]}
    
            except KeyError:
                continue
    # sort all server_urls and sort job_ids based on server_urls
    for model_path, server_info in server_dict.items():
        server_urls = server_info["server_urls"]
        job_ids = server_info["job_ids"]
        sorted_pairs = sorted(zip(server_urls, job_ids))
        sorted_server_urls, sorted_job_ids = zip(*sorted_pairs)
        sorted_server_urls = list(sorted_server_urls)
        sorted_job_ids = list(sorted_job_ids)
        server_info["server_urls"] = sorted_server_urls
        server_info["job_ids"] = sorted_job_ids
    return server_dict



class VLLM:
    def __init__(self, name, port):
        self.name = name
        self.port = port

    def predict(self, prompt, system_prompt=None):
        client = openai.OpenAI(
            base_url=self.port,
            api_key="NOT A REAL KEY",
        )
        # try:
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt},]
        else:
            messages = []
        messages += [{"role": "user","content": prompt}]
        chat_completion = client.chat.completions.create(
            model=self.name,
            messages=messages,
            # temperature=temperature,
        )
        # except openai.BadRequestError as e:
        #     if "maximum context length" in str(e):
        #         print("Error: The prompt is too long for the model's maximum context length.")
        #         return ''
        #     assert False, f"Error: {e}"
        return chat_completion.choices[0].message.content
    
    def batch_predict(self, batch_prompts, system_prompt=None):
        all_return_values = thread_map(
            lambda p: self.predict(p, system_prompt),
            batch_prompts,
            max_workers=20,
            desc="using vllm")
        
        try:
            assert type(all_return_values[0]) == str
        except:
            print("batch_prompts", batch_prompts)
            print("all_return_values", type(all_return_values), all_return_values)
            assert False
        return all_return_values


class VLLM_VL(VLLM):
    def __init__(self, name, port):
        super().__init__(name, port)
        
    def predict(self, prompt, system_prompt=None):
        client = openai.OpenAI(
            base_url=self.port,
            api_key="NOT A REAL KEY",
        )
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt},]
        else:
            messages = []
        messages += [{"role": "user","content": [{"type": "text", "text": prompt}]}]
        chat_completion = client.chat.completions.create(
            model=self.name,
            messages=messages,
            # temperature=temperature,
        )
        return chat_completion.choices[0].message.content


def main(args):
    model_name = args.model_name
    dataset = args.dataset
    eval_type = args.eval_type

    if "Qwen" in model_name:
        model = VLLM_VL(args.full_model_name, args.port)
    else:
        model = VLLM(args.full_model_name, args.port)

    output_dir = f'{root_path}/{args.llm_response_path}/{dataset}/{eval_type}/'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/{model_name}_outputs.json'
    if os.path.exists(output_file):
        with open(output_file) as f:
            all_results = json.load(f)
    else:
        all_results = []
    history_i = len(all_results)
    print("history", history_i)

    if dataset == "virtualhome":
        input_path = f'{root_path}/output/{dataset}/generate_prompts/{eval_type}/helm_prompt.json'
    else:
        if "action_sequenc" in eval_type:
            input_path = f'{root_path}/output/{dataset}/generate_prompts/{eval_type}/action_sequence_prompts.json'
        else:
            input_path = f'{root_path}/output/{dataset}/generate_prompts/{eval_type}/{eval_type}_prompts.json'
        
    with open(input_path) as f:
        data = json.load(f)
        if "system_prompt" in data[0]:
            system_prompt = data[0]['system_prompt']
        else:
            system_prompt = None
    for i in range(history_i, len(data), args.batch_size):
        batch = data[i:i + args.batch_size]
        batch_prompts = [line['llm_prompt'] for line in batch]
        batch_identifier = [line['identifier'] for line in batch]
        batch_responses = model.batch_predict(batch_prompts, system_prompt)
        assert len(batch_responses) == len(batch_prompts), f"batch_responses: {len(batch_responses)}, batch_prompts: {len(batch_prompts)}"

        with open(output_file, 'w') as fout: # save every batch
            for i, response in enumerate(batch_responses):
                line = {
                    'identifier': batch_identifier[i],
                    'llm_output': response,
                }
                all_results.append(line)
            json.dump(all_results, fout)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=str, default="")
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--batch_size', type=int, default=40)
    parser.add_argument('--dataset', type=str, default="virtualhome")
    parser.add_argument('--eval_type', type=str, default="action_sequencing")
    parser.add_argument('--llm_response_path', type=str, default="responses")
    args = parser.parse_args()

    if "Llama" in args.model_name:
        args.full_model_name = f"meta-llama/{args.model_name}"
    elif "DeepSeek" in args.model_name:
        args.full_model_name = f"deepseek-ai/{args.model_name}"
    elif 'Qwen' in args.model_name:
        args.full_model_name = f"Qwen/{args.model_name}"

    if "http" not in args.port and len(args.port) < 3:
        server_dict = get_available_servers()[args.full_model_name]
        server_urls = server_dict["server_urls"]
        args.port = server_urls[int(args.port)]

    main(args)
    """
    python generate.py \
    --model_name Llama-3.1-70B-Instruct \
    --port 0 \
    --batch_size 40 \
    --dataset virtualhome \
    --eval_type action_sequencing &
    
    """
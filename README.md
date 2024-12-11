# Reproduction Steps for ToolSandbox

Note that the original README is at `README_original.md`.

## Setup

- Make a new Conda environment with Python 3.11, i.e.,
```
conda create -n tool_sandbox python=3.11
conda activate tool_sandbox
```
- Install `pip` via `conda install pip`.
- Change directory to the `ToolSandbox` directory.
- Run `pip install ".[dev]"`.

## Benchmarking

- To benchmark Athene V2 Chat, run `RAPID_API_KEY=<rapid_api_key> OPENAI_API_KEY=<openai_api_key> ATHENE_BASE_URL=<endpoint> ATHENE_API_KEY=<api_key> tool_sandbox --user GPT_4_o_2024_05_13 --agent AtheneV2Chat`
- To benchmark Athene V2 Agent, run `RAPID_API_KEY=<rapid_api_key> OPENAI_API_KEY=<openai_api_key> ATHENE_BASE_URL=<endpoint> ATHENE_API_KEY=<api_key> tool_sandbox --user GPT_4_o_2024_05_13 --agent AtheneV2Agent`
- To benchmark OpenAI GPT-4o-0513, run `RAPID_API_KEY=<rapid_api_key> OPENAI_API_KEY=<openai_api_key> tool_sandbox --user GPT_4_o_2024_05_13 --agent GPT_4_o_2024_05_13`

The `rapid_api_key` is any valid RapidAPI key. The `endpoint` is the endpoint provided to you by Nexusflow. It should be of the form `<url>:<port>/v1`, e.g., "http://abc.xyz:10101/v1". The `api_key` is the key given to you by Nexusflow. The `openai_api_key` is any valid OpenAI API key.

## Generating results 

Note that each benchmark run generates a new result cache in `data`, say `data/<run_id>`. To obtain the evaluation result, open the file `data/<run_id>/result_summary.json`. Near the end of the file, there is an entry called `ALL_CATEGORIES`: the corresponding value is the evaluation metric.

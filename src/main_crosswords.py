import argparse
import logging
import os
import json
import re
from datetime import datetime
from cat.methods.cat import cat_cross
from cat.tasks.crosswords import MiniCrosswordsEnv
from cat.models import gpt_usage, gpt
from cat.prompts.crosswords import propose_prompt

# Helper functions from notebook
def prompt_wrap(obs):
    return propose_prompt.format(input=obs)

def parse_line(input_str):
    # regular expression pattern to match the input string format
    pattern = r'^([hv][1-5])\. ([a-zA-Z]{5,5}) \((certain|high|medium|low)\).*$'

    # use regex to extract the parts of the input string
    match = re.match(pattern, input_str)

    if match:
        # extract the matched groups
        parts = [match.group(1), match.group(2), match.group(3)]
        return parts
    else:
        return None

confidence_to_value = {'certain': 1, 'high': 0.5, 'medium': 0.2, 'low': 0.1}  # TODO: ad hoc

def parse_response(response):
    # split the response into lines
    lines = response.split('\n')

    # parse each line
    parsed_lines = [parse_line(line) for line in lines]

    # filter out the lines that didn't match the format
    parsed_lines = [(line[0].lower() + '. ' + line[1].lower(), confidence_to_value.get(line[2], 0)) for line in parsed_lines if line is not None]

    return parsed_lines if len(parsed_lines) >= 1 else None

def get_candidates_to_scores(env):
    obs = env.render()
    if obs in env.cache: 
        print('cache hit')
        return env.cache[obs]
    print('call gpt')
    responses = gpt(prompt_wrap(obs), model='gpt-4', n=8)
    candidates_to_scores = {}
    for response in responses:
        parsed_response = parse_response(response)
        if parsed_response:
            for candidate, score in parsed_response:
                candidates_to_scores[candidate] = candidates_to_scores.get(candidate, 0) + score
    env.cache[obs] = candidates_to_scores
    return candidates_to_scores

# Configuration for distributed/stochastic solve
args = argparse.Namespace(
    backend='gpt-4', 
    temperature=0.7,
    softmax_temperature=1.0,  # Temperature for softmax in dfs_softmax
    max_per_state=3,  # Maximum number of actions to try per state
    time_limit=100  # Maximum number of infos to collect
)

# Set up logging to file
log_dir = '/workspaces/caterpillar-of-thoughts/logs/crosswords/new'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'distsolve_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# Configure logging to write to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),  # Append mode
        logging.StreamHandler()
    ]
)

# Set up logging to file
log_dir = '/workspaces/caterpillar-of-thoughts/logs/crosswords/new'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'dfs_softmax_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# Configure logging to write to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),  # Append mode
        logging.StreamHandler()
    ]
)

# Initialize environment
env = MiniCrosswordsEnv()

logging.info("="*60)
logging.info("SOFTMAX REWINDING - CROSSWORDS")
logging.info("="*60)
logging.info(f"Softmax temperature: {args.softmax_temperature}")
logging.info(f"Max per state: {args.max_per_state}")
logging.info(f"Time limit: {args.time_limit}")
logging.info(f"Log file: {log_file}")
logging.info("-" * 60)

# Run DFS with softmax on a range of crossword puzzles
infoss = []

# Process each puzzle with separate logging and token tracking
for i in range(0, 100, 5):
    # Create separate log file for this puzzle
    puzzle_log_file = os.path.join(log_dir, f'puzzle_{i:03d}_detailed.log')
    
    logging.info("="*60)
    logging.info(f"STARTING PUZZLE INDEX: {i}")
    logging.info("="*60)
    logging.info(f"Puzzle-specific log: {puzzle_log_file}")
    
    # Get token usage before this puzzle
    usage_before = gpt_usage(backend=args.backend)
    tokens_before = usage_before['prompt_tokens'] + usage_before['completion_tokens']
    
    # Reset environment and solve
    env.reset(i)
    infos = []
    actions = []
    
    print(f"\n{'='*60}")
    print(f"Processing Puzzle {i}")
    print(f"{'='*60}")
    
    cat_cross(env, actions, infos, args.time_limit, args.max_per_state, 
                args.softmax_temperature, get_candidates_to_scores, log_file=puzzle_log_file)
    infoss.append(infos)
    
    # Get token usage after this puzzle
    usage_after = gpt_usage(backend=args.backend)
    tokens_after = usage_after['prompt_tokens'] + usage_after['completion_tokens']
    puzzle_tokens = tokens_after - tokens_before
    puzzle_cost = usage_after['cost'] - usage_before['cost']
    
    # Save results for this specific puzzle
    puzzle_output_file = os.path.join(log_dir, f'puzzle_{i:03d}_results.json')
    with open(puzzle_output_file, 'w') as fout:
        json.dump(infos, fout, indent=2)
    
    # Save cumulative results
    cumulative_output_file = os.path.join(log_dir, 'infoss_cat_softmax.json')
    with open(cumulative_output_file, 'w') as fout:
        json.dump(infoss, fout)
    
    # Save token usage for this puzzle
    token_log_file = os.path.join(log_dir, f'puzzle_{i:03d}_tokens.txt')
    with open(token_log_file, 'w') as fout:
        fout.write(f"Puzzle {i} Token Usage\n")
        fout.write(f"{'='*50}\n")
        fout.write(f"Prompt tokens: {usage_after['prompt_tokens'] - usage_before['prompt_tokens']:,}\n")
        fout.write(f"Completion tokens: {usage_after['completion_tokens'] - usage_before['completion_tokens']:,}\n")
        fout.write(f"Total tokens: {puzzle_tokens:,}\n")
        fout.write(f"Estimated cost: ${puzzle_cost:.4f}\n")
    
    # Log and print detailed results
    logging.info(f"Puzzle {i} completed. Total steps: {len(infos)}")
    print(f"Puzzle {i} completed. Total steps: {len(infos)}")
    
    if infos:
        best = max(infos, key=lambda x: x['info']['r_word'])
        logging.info(f"Best solution for puzzle {i}: r_word={best['info']['r_word']}, steps={best['env_step']}")
    logging.info("-" * 60)

    # remove this for full run
    break

logging.info("="*60)
logging.info("ALL PUZZLES COMPLETED")
logging.info("="*60)
logging.info(f"Total puzzles processed: {len(infoss)}")


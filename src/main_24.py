import argparse
import logging
import os
from datetime import datetime
from cat.methods.cat import cat_24
from cat.tasks.game24 import Game24Task
from cat.models import gpt_usage

# Configuration for distributed/stochastic solve
# dist_selection options: 'uniform', 'weighted', or 'softmax'
#   - 'uniform': Equal probability for all states
#   - 'weighted': Linear value-based weighting
#   - 'softmax': Softmax probability based on values (with temperature control)
# dist_temperature: Temperature for softmax (higher = more uniform, lower = more greedy)
# max_iterations sets the maximum number of steps (default 100)
args = argparse.Namespace(
    backend='gpt-4', 
    temperature=0.7, 
    task='game24', 
    naive_run=False, 
    prompt_sample=None, 
    method_generate='propose', 
    method_evaluate='value', 
    method_select='greedy',  # Not used in dist_solve
    n_generate_sample=1, 
    n_evaluate_sample=3, 
    n_select_sample=5,  # Not used in dist_solve
    dist_selection='softmax',  # Options: 'uniform', 'weighted', 'softmax'
    dist_temperature=1.0,  # Temperature for softmax (only used when dist_selection='softmax')
    max_iterations=15  # Maximum iterations before stopping
)

# Set up logging to file
log_dir = '/workspaces/caterpillar-of-thought-llm/logs/game24'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'.log')

# Configure logging to write to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),  # Append mode
        logging.StreamHandler()
    ]
)

task = Game24Task()
logging.info("="*60)
logging.info("DISTRIBUTED/STOCHASTIC SOLVE - RANDOM EXPLORATION")
logging.info("="*60)
logging.info(f"Selection mode: {args.dist_selection}")
if args.dist_selection == 'softmax':
    logging.info(f"Softmax temperature: {args.dist_temperature}")
logging.info(f"Max iterations: {args.max_iterations}")
logging.info(f"Log file: {log_file}")
logging.info("-" * 60)

indices = range(900, 1000)

for idx in [900]:
    logging.info(f"Processing index: {idx}")
    ys, infos = cat_24(args, task, idx, to_print=True)

    # Get token usage information
    token_usage = gpt_usage(backend=args.backend)
    infos['token_count'] = token_usage

    logging.info("="*60)
    logging.info(f"FINAL RESULTS FOR INDEX {idx}")
    logging.info("="*60)
    logging.info(f"Token Usage: {token_usage}")
    logging.info(f"Explored States Count: {len(infos.get('explored_states', {}))}")
    # Get top 5 for display purposes
    if 'explored_states' in infos:
        explored = infos['explored_states']
        sorted_states = sorted(explored.items(), key=lambda x: x[1], reverse=True)
        top_10 = sorted_states[:10]
        logging.info(f"Top 10 States by Value: {[y for y, v in top_10]}")
        logging.info(f"Top 10 Values: {[v for y, v in top_10]}")
    logging.info(f"Results: {infos}")

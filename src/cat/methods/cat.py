import itertools
import numpy as np
import re
from functools import partial
from cat.models import gpt

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

def get_proposals(task, x, y): 
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n')
    return [y + _ + '\n' for _ in proposals]

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]



def cat_24(args, task, idx, to_print=True):
   
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    x = task.get_input(idx)
    
    # Track all explored states with their values
    explored_states = {'': 0}  # start with empty state
    last_step_ys = {'': 0}  # start with empty state

    current_state = ''
    infos = []
    
    max_iterations = getattr(args, 'max_iterations', 15)  # default max 100 iterations
    step = 0
    found_answer = False
    
    while step < max_iterations:
        # Generation: generate children from current state
        if args.method_generate == 'sample':
            stop_idx = min(step, len(task.stops) - 1) if hasattr(task, 'stops') and task.stops else step
            stop = task.stops[stop_idx] if hasattr(task, 'stops') and task.stops else None
            new_ys = get_samples(task, x, current_state, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=stop)
        elif args.method_generate == 'propose':
            new_ys = get_proposals(task, x, current_state)
        else:
            raise ValueError(f'Unknown method_generate {args.method_generate}')
        
        # Normalize candidate formatting
        new_ys = [ny if ny.endswith('\n') else ny + '\n' for ny in new_ys]

        # check if new_ys contains a valid answer
        for ny in new_ys:
            if task.is_valid_answer(ny, x):
                if to_print:
                    print(f"Found valid answer at step {step}: {ny}")
                found_answer = True
                infos.append({
                    'step': step,
                    'x': x,
                    'current_state': current_state,
                    'current_value': current_value,
                    # 'last_step_ys_state': last_step_ys_state,
                    # 'last_step_ys_value': last_step_ys_value,
                    'new_ys': new_ys,
                    'values': values,
                    'explored_count': len(explored_states)
                })
                break
        if found_answer:
            break
        
        # Evaluation: evaluate all new children
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)
        else:
            values = get_values(task, x, new_ys, args.n_evaluate_sample)
        
        # Display sorted candidates like solve() does
        if to_print and values:
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'\n-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n')
        
        # Filter and add new states to explored states
        for state, value in zip(new_ys, values):
            if state not in explored_states:
                explored_states[state] = value
             

    
        # Filter valid states (exclude discarded ones)
        valid_states = list(explored_states.keys())
        valid_values = list(explored_states.values())
        
        # Check if we have any valid states left
        if not valid_states:
            if to_print:
                print("No valid states remaining!")
            break
        
        # Random selection: pick one state at random from all valid explored states
        selection_mode = getattr(args, 'dist_selection', 'uniform')  # uniform, weighted, or softmax
        
        if selection_mode == 'softmax':
            print("Using softmax-based selection")
            # Softmax-based selection
            temperature = getattr(args, 'dist_temperature', 1.0)
            weights = np.array(valid_values, dtype=float)
            # Apply softmax: exp(value/T) / sum(exp(value/T))
            weights = np.exp(weights / temperature)
            weights = weights / weights.sum()
            selected_idx = np.random.choice(len(valid_states), p=weights)
        elif selection_mode == 'weighted':
            # Simple value-weighted selection (linear normalization)
            weights = np.array(valid_values, dtype=float)
            weights = weights - weights.min() + 1e-10  # shift to positive
            weights = weights / weights.sum()
            selected_idx = np.random.choice(len(valid_states), p=weights)
        else:  # uniform
            # Uniform random selection
            selected_idx = np.random.choice(len(valid_states))
        
        current_state = valid_states[selected_idx]
        current_value = valid_values[selected_idx]

        # Log
        infos.append({
            'step': step,
            'x': x,
            'current_state': current_state,
            'current_value': current_value,
            'new_ys': new_ys,
            'values': values,
            'explored_count': len(explored_states)
        })
        
        if to_print:
            print(f"step={step} explored={len(explored_states)} current_value={current_value:.3f} new_states={new_ys}")
        
        step += 1

 
    
    # Return all explored states
    all_explored_ys = list(explored_states.keys())
    all_explored_values = list(explored_states.values())
    
    return all_explored_ys, {'steps': infos, 'explored_states': explored_states}
 


def cat_cross(env, actions, infos, time_limit, max_per_state, temperature=1.0, get_candidates_to_scores=None, log_file=None):


    if get_candidates_to_scores is None:
        raise ValueError("get_candidates_to_scores function must be provided")
    
    # Open log file if provided
    log_f = open(log_file, 'w') if log_file else None
    
    def log_print(msg):
        """Print to console and optionally to log file"""
        print(msg)
        if log_f:
            log_f.write(msg + '\n')
            log_f.flush()
    
    # S: Set of non-complete tables (each table is a tuple of actions)
    # Start with the empty table
    S = [tuple()]
    
    # Track values/scores for each table in S
    # Give empty table a high score so it gets sampled more often (allows rewinding to start)
    state_values = {tuple(): 1.0}  # Empty table has higher value to encourage exploration from scratch
    
    # Track cumulative confidence scores (sum of action scores) for each table
    confidence_scores = {tuple(): 0.0}  # Empty table has 0 cumulative confidence
    
    # Save the original initial state
    initial_board = env.board.copy()
    initial_status = env.status.copy()
    
    iteration = 0
    max_iterations = 20
    
    
    while iteration < max_iterations and len(infos) < time_limit:
        log_print(f"\n{'='*60}")
        log_print(f"Iteration {iteration}: |S| = {len(S)} non-complete tables")
        log_print(f"{'='*60}")
        
        if len(S) == 0:
            log_print("No tables in S, ending search")
            break
        
        # Step 1: Sample a table from S based on softmax over scores
        # Pr[table] ∝ exp(-score/T) -- use negative to encourage exploring lower-scored tables
        table_list = list(S)
        scores = np.array([state_values.get(t, 0.0) for t in table_list])
        
        # Apply softmax: exp(-score/T) to sample tables (prefer lower scores for exploration)
        exp_scores = np.exp(scores / temperature)
        table_probs = exp_scores / exp_scores.sum()
        
        # Sample one table
        table_idx = np.random.choice(len(table_list), p=table_probs)
        selected_table = table_list[table_idx]
        selected_prob = table_probs[table_idx]
        selected_score = scores[table_idx]
        
        log_print(f"\n*** Sampled table #{table_idx} from S ***")
        log_print(f"  Length: {len(selected_table)} actions")
        log_print(f"  Score: {selected_score:.4f}")
        log_print(f"  Probability: {selected_prob:.4f}")
        if len(selected_table) > 0:
            log_print(f"  Actions: {list(selected_table)}")
        else:
            log_print(f"  (Empty table)")
        
        # Step 2: Replay the selected table to get to its state
        env.reset(env.idx, board=initial_board.copy(), status=initial_status.copy(), steps=0)
        
        valid_table = True
        for action in selected_table:
            obs, r, done, info = env.step(action)
            if any(_ == 2 for _ in env.status):
                valid_table = False
                break
        
       
        
        if env.steps >= 10:
            log_print(f"Table is complete (10 steps), removing from S")
            S.remove(selected_table)
            continue
        
        # Step 3: Generate next step candidates for this table
        candidates_to_scores = get_candidates_to_scores(env)
        
        if len(candidates_to_scores) == 0:
            log_print(f"No candidates available, removing table from S")
            S.remove(selected_table)
            if selected_table in state_values:
                del state_values[selected_table]
            continue
        
        log_print(f"Generated {len(candidates_to_scores)} candidate actions for next step")
        
        # Step 4: Create tables from top 3 candidates
        candidate_list = list(candidates_to_scores.keys())
        candidate_scores = np.array([candidates_to_scores[c] for c in candidate_list])
        
        # Take only top 3 candidates by score
        num_candidates_to_try = min(3, len(candidate_list))
        top_indices = np.argsort(candidate_scores)[-num_candidates_to_try:][::-1]
        selected_candidates = [candidate_list[i] for i in top_indices]
        selected_scores = [candidate_scores[i] for i in top_indices]
        
        log_print(f"Processing top {num_candidates_to_try} candidates out of {len(candidate_list)}...")
        
        # Save the current state (after replaying selected table) to restore for each candidate
        saved_board = env.board.copy()
        saved_status = env.status.copy()
        saved_steps = env.steps
        
        # Try each candidate action to create new tables
        tables_added = 0
        
        for i, sampled_action in enumerate(selected_candidates):
            action_score = selected_scores[i]
            
            # Reset to the saved state (after replaying selected table)
            env.reset(env.idx, board=saved_board.copy(), status=saved_status.copy(), steps=saved_steps)
            
            # Take the sampled action to create new table
            obs, r, done, info = env.step(sampled_action)
            count = env.prompt_status()
            
            # Create new table: old table + new action
            new_table = selected_table + (sampled_action,)
            
            # Check if new table is valid
            is_valid = env.steps < 10 and not any(_ == 2 for _ in env.status)
            
            if is_valid:
                # Add new table to S if not already present and not complete
                if new_table not in S:
                    # Only add if table is not complete
                    if env.steps < 10:
                        S.append(new_table)
                        # Compute score for new table (use cumulative confidence score)
                        cumulative_confidence = confidence_scores.get(selected_table, 0.0) + action_score
                        confidence_scores[new_table] = cumulative_confidence
                        state_values[new_table] = cumulative_confidence
                        new_score = cumulative_confidence
                        tables_added += 1
                        
                        log_print(f"  [{i+1}/{num_candidates_to_try}] Added table from action: {sampled_action} (confidence={new_score:.4f}, r_word={info['r_word']:.4f})")
                    
                    # Log this step (only for first few to avoid cluttering)
                    if len(infos) < time_limit:
                        step_info = {
                            'total_step': len(infos),
                            'env_step': env.steps,
                            'actions': list(new_table),
                            'info': info,
                            'count': count,
                            'table_score': cumulative_confidence,
                            'parent_table_score': selected_score,
                            'parent_prob': float(selected_prob),
                            'action_score': float(action_score),
                            'iteration': iteration,
                            'pool_size': len(S),
                            'candidate_index': i
                        }
                        infos.append(step_info)
                else:
                    log_print(f"  [{i+1}/{num_candidates_to_try}] Table from {sampled_action} already in S")
            else:
                log_print(f"  [{i+1}/{num_candidates_to_try}] Invalid table from {sampled_action} (steps={env.steps}, impossible={any(_ == 2 for _ in env.status)})")
        
        log_print(f"Added {tables_added} new tables to S (now |S|={len(S)})")
        
        if infos:
            best = max(infos, key=lambda x: x['info']['r_word'])
            log_print(f"Best so far: r_word={best['info']['r_word']}")
        
       
  
        iteration += 1
    
    log_print(f"\nFinished after {iteration} iterations with {len(infos)} steps logged")
    log_print(f"Final |S| = {len(S)} tables")
    log_print(f"Total unique tables explored: {len(state_values)}")
    
    # Reset environment to original state
    env.reset(env.idx, board=initial_board.copy(), status=initial_status.copy(), steps=0)
    
    # Close log file if it was opened
    if log_f:
        log_f.close()



 
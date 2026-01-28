import re
import os
import sympy
import pandas as pd
from cat.tasks.base import Task, DATA_PATH
from cat.prompts.game24 import * 


def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]


class Game24Task(Task):
    """
    Input (x)   : a string of 4 numbers
    Output (y)  : a trajectory of 3 steps to reach 24
    Reward (r)  : 0 or 1, depending on whether the trajectory is correct
    Input Example: 
        1 2 3 4
    Output Example: 
        1 + 2 = 3 (left: 3 3 4)
        3 + 3 = 6 (left: 4 6)
        6 * 4 = 24 (left: 24)
        (1 + 2 + 3) * 4 = 24
    """
    def __init__(self, file='24.csv'):
        """
        file: a csv file (fixed)
        """
        super().__init__()
        path = os.path.join(DATA_PATH, '24', file)
        self.data = list(pd.read_csv(path)['Puzzles'])
        self.value_cache = {}
        self.steps = 4
        self.stops = ['\n'] * 4

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]

    def test_output(self, idx: int, output: str):
        expression = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
        numbers = re.findall(r'\d+', expression)
        problem_numbers = re.findall(r'\d+', self.data[idx])
        if sorted(numbers) != sorted(problem_numbers):
            return {'r': 0}
        try:
            # print(sympy.simplify(expression))
            return {'r': int(sympy.simplify(expression) == 24)}
        except Exception as e:
            # print(e)
            return {'r': 0}
            
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        return cot_prompt.format(input=x) + y
    
    @staticmethod
    def propose_prompt_wrap(x: str, y: str='') -> str:
        current_numbers = get_current_numbers(y if y else x)
        if current_numbers == '24':
            prompt = cot_prompt.format(input=x) + 'Steps:' + y
            # print([prompt])
        else:
            prompt = propose_prompt.format(input=current_numbers)
        return prompt
    
    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        last_line = y.strip().split('\n')[-1]
        if 'left: ' not in last_line:  # last step
            ans = last_line.lower().replace('answer: ', '')
            # print([value_last_step_prompt.format(input=x, answer=ans)])
            return value_last_step_prompt.format(input=x, answer=ans)
        current_numbers = get_current_numbers(y)
        return value_prompt.format(input=current_numbers)
    
    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
            return 0
        value_names = [_.split('\n')[-1] for _ in value_outputs]
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20,  'very sure':60}  # TODO: ad hoc
        value = sum(value * value_names.count(name) for name, value in value_map.items())
        return value
    
    @staticmethod
    def cot_judge_value_prompt_wrap(x: str, y: str='') -> str:
    
        last_line = y.strip().split('\n')[-1] if y else ''
        # if last line has no 'left:' it's probably the final Answer line -> judge the answer
        if last_line and 'left:' not in last_line:
            ans = last_line.lower().replace('answer: ', '')
            return value_judge_prompt.format(input=x, answer=ans)
        # otherwise continue chain-of-thought from the current numbers
        current_numbers = get_current_numbers(y if y else x)
        return cot_prompt.format(input=current_numbers) + y

    @staticmethod
    def cot_judge_value_outputs_unwrap(x: str, y: str, outputs: list) -> float:
        import re
        judges = []
        token_re = re.compile(r"\b(sure|likely|impossible)\b")
        for out in outputs:
            text = out.strip()
            low = text.lower()
            m = token_re.search(low)
            if m:
                judge = m.group(1)
            elif 'judge:' in low:
                after = low.split('judge:')[-1].strip()
                m2 = token_re.search(after)
                judge = m2.group(1) if m2 else after.split('\n')[0].split()[0] if after.split() else ''
            else:
                # fallback: take last non-empty line and search there
                parts = [l.strip() for l in text.split('\n') if l.strip()]
                last = parts[-1].lower() if parts else ''
                m3 = token_re.search(last)
                judge = m3.group(1) if m3 else last
            judges.append(judge)
        value_map = {'impossible': 0.000, 'likely': 1, 'sure': 20}
        value = sum(value_map.get(j, 0) for j in judges)
        return value

    def is_valid_answer(self, y: str, x: str) -> bool:
        answer_line = y.strip().split('\n')[-1]
        if 'answer:' not in answer_line.lower():
            return False
        expression = answer_line.lower().replace('answer: ', '').split('=')[0]
        #check if expression is valid
        try:
            result = sympy.simplify(expression)
            if result != 24:
                return False
        except:
            return False
        numbers = re.findall(r'\d+', expression)
        problem_numbers = re.findall(r'\d+', x)
        return sorted(numbers) == sorted(problem_numbers)   
    
    
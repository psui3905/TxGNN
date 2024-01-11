from openai import OpenAI
from multiprocessing import Lock
import openai, sys, yaml, os
import subprocess

lock = Lock()
os.environ['model'] = ''
os.environ['prompt'] = ''
os.environ['api_key'] = ''

question = '''Which signature profile is the best as disease embedding for the query disease? (give a short reasoning)'''

final_question ='''Provide your answer in the format (nothing else should be included in your answer): 
Most informative signature: [Choose from (1), (2), (3)]
Confidential score: [Assign a score from 0.0 to 1.0 for each signature, ensuring the total score across all signatures sums to 1.0]'''

refinement = '''Which signature profile is most informative for the query disease and can be used as disease embedding? Solve them in a step-by-step fashion, starting by summarizing the available information. 
Output a single option from the three signature options as the final answer. We provide several possible reasonings for the question. 
Some of them maybe correct and some incorrect. Beware of wrong reasoning and do not repeat wrong reasoning.'''



class LLM_Enhancement:
    def __init__(self) -> None:
        self.llm_args = yaml.load(open('/root/TxGNN/txgnn/llm_config.yaml', 'r'), Loader=yaml.FullLoader)
        # Path to your Python 3.9 interpreter '/path/to/python3.9/bin/python'
        self.python39_interpreter = '/opt/conda/envs/py39/bin/python'
        # Path to the Python 3.9 script '/path/to/python39_script.py'
        self.python39_script = '/root/TxGNN/txgnn/llmplus.py'
        # subprocess for Gemini-pro & PaLM2
        self.command = [self.python39_interpreter, self.python39_script]
        
    def sig_suggestion(self, disease, llm, ps_sig, at_sig, ds_sig):
        info = f'''Here is the query disease and available signature profiles:
        Disease info: {disease}
        (1): {ps_sig}
        (2): {at_sig}
        (3): {ds_sig}'''
        
        return self.ensemble_refine(info, model=llm, ensemble=3)
        
    def ensemble_refine(self, info, model='palm2', ensemble=3):
        explore_prompt = self.llm_args['instructions'] + '\n' + info + '\n' + question
        reasons = []
        for i in range(ensemble):
            if model == 'gemini':
                reasons.append(self.gemini(prompt=explore_prompt))
            elif model == 'palm2':
                reasons.append(self.palm2(prompt=explore_prompt))
            else:
                reasons.append(self.gpt(prompt=explore_prompt, model=model))
        reason_prompt = ['Reasoning: ' + reason for i, reason in enumerate(reasons)]
        reason_prompt = '\n'.join(reason_prompt)
        refine_prompt = self.llm_args['instructions'] + '\n' + info + '\n' + refinement + '\n' + reason_prompt + '\n' + final_question
        print(refine_prompt)
        if model == 'gemini':
            return self.gemini(prompt=refine_prompt)
        elif model == 'palm2':
            return self.palm2(prompt=refine_prompt)
        return self.gpt(prompt=refine_prompt, model=model)
    
    def gpt(self, prompt, model='gpt-4'):
        prompt = {
            "role": "user",
            "content": prompt
        }
        while True:
            try:
                with lock:
                    client = OpenAI(api_key=self.llm_args['gpt']['api_key'])
                    response = client.chat.completions.create(
                        model=model,
                        messages=[prompt],
                        temperature=self.llm_args['gpt'][model]['temperature'],
                        max_tokens=self.llm_args['max_output_tokens'],
                        top_p=self.llm_args['gpt'][model]['top_p']
                    )
                return response.choices[0].message.content
            except openai.APIConnectionError as e:
                print('Failed to connect... trying again')
                pass
    
    def gemini(self, prompt):
        os.environ['prompt'] = prompt
        os.environ['model'] = 'gemini'
        os.environ['api_key'] = self.llm_args['gemini']['api_key']
        result = subprocess.run(self.command, capture_output=True, text=True)
        # print(result.stderr)
        return result.stdout
        
    def palm2(self, prompt):
        os.environ['prompt'] = prompt
        os.environ['model'] = 'gemini'
        os.environ['api_key'] = self.llm_args['palm2']['api_key']
        result = subprocess.run(self.command, capture_output=True, text=True)
        # print(result.stderr)
        return result.stdout
    
    def str2idx_sig(self, content):
        try:
            # print(content)
            choosen_sig = content.split('\n')[0].split(': ')[1].split()[0].lower()
            # score = float(content.split('\n')[1].split(': ')[1])
            print('Best: ' + choosen_sig + ' ' + content.split('\n')[1])
        except:
            # print('LLM+ failed, using default signature: at')
            choosen_sig = 'at'
            score = 1.0
        return (choosen_sig, 1.0)

if __name__ == "__main__":
    disease = 'asthma'
    model_name = 'palm2'
    ps_sig = ['RAD50', 'TNIP1', 'ADCY2', 'KIF3A', 'PTGDR2', 'ADCYAP1R1', 'SCGB3A2', 'PARP1', 'ADRB2', 'DNAH5', 'DNMT1', 'EDN1', 'ALDH2', 'CDHR3', 'IKZF3', 'IFNL3', 'CTNNA3', 'CXCL1', 'TBX21', 'HLA-DPB1']
    at_sig = ['RAD50', 'TNIP1', 'ADCY2', 'KIF3A', 'PTGDR2', 'ADCYAP1R1', 'SCGB3A2', 'PARP1', 'ADRB2', 'DNAH5', 'DNMT1', 'EDN1', 'ALDH2', 'CDHR3', 'IKZF3', 'IFNL3', 'CTNNA3', 'CXCL1', 'TBX21', 'HLA-DPB1', 'bronchial disease', 'cough variant asthma', 'occupational asthma', 'intrinsic asthma', 'allergic asthma', ]
    ds_sig = ['DNM1L', 'MIR499B', 'HUWE1', 'ATP9A', 'RAD50', 'G3BP1', 'CDK6', 'HNRNPR', 'CDKN1A', 'TNIP1', 'TRDN', 'RACK1', 'TUBGCP3', 'SEMA4F', 'MYBBP1A', 'OLFM4', 'CCT4', 'PDLIM5', 'CTCF', 'NFAT5']
    llm = LLM_Enhancement()
    result = llm.sig_suggestion(disease, model_name, ps_sig, at_sig, ds_sig)
    choosen_sig, score = llm.str2idx_sig(result)
    print(choosen_sig, score)

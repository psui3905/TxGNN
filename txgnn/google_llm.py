import subprocess
import sys
import os

os.environ['model'] = ''
os.environ['prompt'] = ''
os.environ['api_key'] = ''

# Path to your Python 3.9 interpreter '/path/to/python3.9/bin/python'
python39_interpreter = '/opt/conda/envs/py39/bin/python'

# Path to the Python 3.9 script '/path/to/python39_script.py'
python39_script = '~/TxGNN/txgnn/llmplus.py'

class Gemini:
    def query(self, disease, ps_sig, at_sig, ds_sig):
        prompt = f'''
            Suppose you are an expert in the interdisciplinary field of pharmaceutical science and AI.
            Given several signature profiles of the query disease, your job is to determine the best signature that
            can be used as an auxiliary disease embedding for drug repurposing.

            Here is the query disease and available signature profiles:
            Disease info: {disease}
            PS  (Protein Signature): {ps_sig}
            AT  (All-node-types Signature): {at_sig}
            DS  (Diffusion Signature): {ds_sig}

            Your answer should only answer:
            Most informative signature: <the best signature name>
            Confidential score: <scale from 0.0 to 1.0, the score of all given signatures should sum to 1.0>'''

        # add subprocess here
        # Command to execute the script with the interpreter
        os.environ['prompt'] = prompt
        os.environ['model'] = 'gemini'
        command = [python39_interpreter, python39_script]

        # Run the command and capture the output
        result = subprocess.run(command, capture_output=True, text=True)
        return result.stdout
    
    def str2idx_sig(self, content):
        try:
            choosen_sig = content.split('\n')[0].split(': ')[1].split()[0].lower()
            score = float(content.split('\n')[1].split(': ')[1])
        except:
            # print(content)
            # print(response.choices[0].message.content)
            choosen_sig = 'at'
            score = 1.0

        return (choosen_sig, score)
    
class PaLM2:
    def query(self, disease, ps_sig, at_sig, ds_sig):
        prompt = f'''
            Suppose you are an expert in the interdisciplinary field of pharmaceutical science and AI.
            Given several signature profiles of the query disease, your job is to determine the best signature that
            can be used as an auxiliary disease embedding for drug repurposing.

            Here is the query disease and available signature profiles:
            Disease info: {disease}
            PS  (Protein Signature): {ps_sig}
            AT  (All-node-types Signature): {at_sig}
            DS  (Diffusion Signature): {ds_sig}

            Your answer should only answer:
            Most informative signature: <the best signature name>
            Confidential score: <scale from 0.0 to 1.0, the score of all given signatures should sum to 1.0>'''
        
        # add subprocess here
        # Command to execute the script with the interpreter

        os.environ['prompt'] = prompt
        os.environ['model'] = 'palm2'
        command = [python39_interpreter, python39_script]

        # Run the command and capture the output
        result = subprocess.run(command, capture_output=True, text=True)

        return result.stdout
    
    def str2idx_sig(self, content):
        try:
            choosen_sig = content.split('\n')[0].split(': ')[1].split()[0].lower()
            score = float(content.split('\n')[1].split(': ')[1])
        except:
            # print(content)
            # print(response.choices[0].message.content)
            choosen_sig = 'at'
            score = 1.0

        return (choosen_sig, score)
    
if __name__ == "__main__":
    disease = 'asthma'
    ps_sig = ['RAD50', 'TNIP1', 'ADCY2', 'KIF3A', 'PTGDR2', 'ADCYAP1R1', 'SCGB3A2', 'PARP1', 'ADRB2', 'DNAH5', 'DNMT1', 'EDN1', 'ALDH2', 'CDHR3', 'IKZF3', 'IFNL3', 'CTNNA3', 'CXCL1', 'TBX21', 'HLA-DPB1']
    at_sig = ['bronchial disease', 'cough variant asthma', 'occupational asthma', 'intrinsic asthma', 'allergic asthma', 'RAD50', 'TNIP1', 'ADCY2', 'KIF3A', 'PTGDR2', 'ADCYAP1R1', 'SCGB3A2', 'PARP1', 'ADRB2', 'DNAH5', 'DNMT1', 'EDN1', 'ALDH2', 'CDHR3', 'IKZF3', 'IFNL3', 'CTNNA3', 'CXCL1', 'TBX21', 'HLA-DPB1']
    ds_sig = ['DNM1L', 'MIR499B', 'HUWE1', 'ATP9A', 'RAD50', 'G3BP1', 'CDK6', 'HNRNPR', 'CDKN1A', 'TNIP1', 'TRDN', 'RACK1', 'TUBGCP3', 'SEMA4F', 'MYBBP1A', 'OLFM4', 'CCT4', 'PDLIM5', 'CTCF', 'NFAT5']
    palm2 = Gemini()
    result = palm2.query(disease, ps_sig, at_sig, ds_sig)
    choosen_sig, score = palm2.str2idx_sig(result)
    print(choosen_sig, score)

    palm2 = PaLM2()
    result = palm2.query(disease, ps_sig, at_sig, ds_sig)
    choosen_sig, score = palm2.str2idx_sig(result)
    print(choosen_sig, score)

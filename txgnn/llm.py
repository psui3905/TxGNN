from openai import OpenAI

openai_api_key = ''


class Gpt4:
    def __init__(self, api_key=openai_api_key):
        self.client = OpenAI(api_key=api_key)

    def query(self, disease, ps_sig, at_sig, ds_sig):
        prompt = {
            "role": "user",
            "content": f'''
            Suppose you are an expert in the interdisciplinary field of pharmaceutical sciecen and AI.
            Given several signature profile of the query disease, your job is to determine the best signature that
            can use as a auxiliary disease embedding for drug repurposing.
            
            Your answer should follow the following format:
            Most informative signature: <the best signature name>
            Confidential score: <scale from 0.0 to 1.0, the score of all given signatures should sum to 1.0>
            
            Here is the query disease and avalaible signature profiles:
            Disease info: {disease}
            PS (Protein Signature): {ps_sig}
            AT (All-node-types Signature): {at_sig}
            DS (Diffusion Signature): {ds_sig}'''
        }
        print(prompt['content'])
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[prompt],
            temperature=0.8,
            max_tokens=1000,
            top_p=1
        )
        return response.choices[0].message.content
    
    def str2idx_sig(self, content):
        choosen_sig = content.split('\n')[0].split(': ')[1].split()[0].lower()
        score = float(content.split('\n')[1].split(': ')[1])
        return (choosen_sig, score)

if __name__ == "__main__":
    disease = 'asthma'
    ps_sig = ['RAD50', 'TNIP1', 'ADCY2', 'KIF3A', 'PTGDR2', 'ADCYAP1R1', 'SCGB3A2', 'PARP1', 'ADRB2', 'DNAH5', 'DNMT1', 'EDN1', 'ALDH2', 'CDHR3', 'IKZF3', 'IFNL3', 'CTNNA3', 'CXCL1', 'TBX21', 'HLA-DPB1']
    at_sig = ['bronchial disease', 'cough variant asthma', 'occupational asthma', 'intrinsic asthma', 'allergic asthma', 'RAD50', 'TNIP1', 'ADCY2', 'KIF3A', 'PTGDR2', 'ADCYAP1R1', 'SCGB3A2', 'PARP1', 'ADRB2', 'DNAH5', 'DNMT1', 'EDN1', 'ALDH2', 'CDHR3', 'IKZF3', 'IFNL3', 'CTNNA3', 'CXCL1', 'TBX21', 'HLA-DPB1']
    ds_sig = ['DNM1L', 'MIR499B', 'HUWE1', 'ATP9A', 'RAD50', 'G3BP1', 'CDK6', 'HNRNPR', 'CDKN1A', 'TNIP1', 'TRDN', 'RACK1', 'TUBGCP3', 'SEMA4F', 'MYBBP1A', 'OLFM4', 'CCT4', 'PDLIM5', 'CTCF', 'NFAT5']
    gpt4 = GPT4()
    result = gpt4.query(disease, ps_sig, at_sig, ds_sig)
    choosen_sig, score = gpt4.str2idx_sig(result)
    print(choosen_sig, score)

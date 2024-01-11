# python3.9 -m pip install requests
# python3.9 -m pip install google-generativeai==0.3.1

import sys
import os
import yaml

prompt = os.getenv('prompt')

model = os.getenv('model')

API_KEY = os.getenv('api_key')

args = yaml.load(open('/root/TxGNN/txgnn/llm_config.yaml', 'r'), Loader=yaml.FullLoader)

def PaLM2(input_prompt):
    import google.generativeai as generativeai

    generativeai.configure(api_key=API_KEY)

    palm_models = [m for m in generativeai.list_models() if 'generateText' in m.supported_generation_methods]
    palm_model = palm_models[0].name

    completion = generativeai.generate_text(
        model=args['palm2']['model'],
        prompt=input_prompt,
        temperature=args['palm2']['temperature'],
        # The maximum length of the response
        max_output_tokens=args['max_output_tokens'],
    )    
    
    return completion.result

def Gemini(input_prompt):
    import google.generativeai as generativeai

    generativeai.configure(api_key=API_KEY)

    gemini_model = generativeai.GenerativeModel(model_name=args['gemini']['model'])

    completion = gemini_model.generate_content(
        input_prompt,
        generation_config={
            'temperature': args['gemini']['temperature'],
            'max_output_tokens': args['max_output_tokens'],
        }
    )
    return completion.text


if __name__ == "__main__":
    if model == 'palm2':
        result = PaLM2(prompt)

    elif model == 'gemini':
        result = Gemini(prompt)

    # must not delete this line
    print(result)


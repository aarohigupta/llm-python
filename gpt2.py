from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# fixing issue with SSL certificate
os.environ["CURL_CA_BUNDLE"] = ""

# Load the environment variables - HF API key
load_dotenv()

# Load the model
model = HuggingFaceHub(
    repo_id="gpt2",
    model_kwargs={'temperature': 0.7, 'max_length':100}
)

# Create the prompt
prompt = PromptTemplate(
    input_variables=["job"],
    template="YOU HAD ONE JOB! You're the CEO of the company and you couldn't even {job}!",
)

# Create the chain
chain = LLMChain(prompt=prompt, llm=model, verbose=True)

# Run the chain
print(chain.run(job="hire a competent receptionist"))
print(chain.run(job="convince the investors to give you more money"))
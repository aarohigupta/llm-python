from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load the environment variables - HF API key
load_dotenv()

# Load the model
model = HuggingFaceHub(repo_id="mrm8488/t5-base-finetuned-wikiSQL")

# Create the prompt
prompt = PromptTemplate(
    input_variables=["question"],
    template="translate English to SQL: {question} </s>",
)

# Create the chain
chain = LLMChain(prompt=prompt, llm=model, verbose=True)

# Run the chain
print(chain.run(question="What is the average age of respondants using a mobile device?"))
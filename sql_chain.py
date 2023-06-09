from dotenv import load_dotenv
from langchain import HuggingFaceHub, SQLDatabase, SQLDatabaseChain, HuggingFacePipeline
# from transformers import AutoTokenizer, GPT2TokenizerFast, pipeline, AutoModelForSeq2SeqLM, AutoModelForCausalLM

load_dotenv()

# load database : chinook.db is a sample database from https://www.sqlitetutorial.net/sqlite-sample-database/
db_uri = "sqlite:///data/chinook.db"

# make custom table info
custom_table_info = {
    "Track": """CREATE TABLE Track (
	"TrackId" INTEGER NOT NULL, 
	"Name" NVARCHAR(200) NOT NULL,
	"Composer" NVARCHAR(220),
	PRIMARY KEY ("TrackId")
)
/*
3 rows from Track table:
TrackId	Name	Composer
1	For Those About To Rock (We Salute You)	Angus Young, Malcolm Young, Brian Johnson
2	Balls to the Wall	None
3	My favorite song ever	The coolest composer of all time
*/"""
}


# db = SQLDatabase.from_uri(
#         db_uri,
#         include_tables=["tracks"], # only include the Track table for this example
#         sample_rows_in_table_info=2,
#         custom_table_info=custom_table_info
#     )

db = SQLDatabase.from_uri(
        db_uri,
        # include_tables=["albums", "artists"], # only include these tables
        include_tables=["customers"],
        # sample_rows_in_table_info=2,
        # custom_table_info=custom_table_info
    )

print("Database tables:")
print(db.table_info)

# print a divider
print("-"*50)

# Load the model
print("Loading model...")

# model = HuggingFaceHub(
#     repo_id="gpt2",
#     model_kwargs={'temperature': 0.1, 'max_length':200}
# )
# Load sql model
# model = HuggingFaceHub(
#     repo_id="mrm8488/t5-base-finetuned-wikiSQL", 
#     model_kwargs={'temperature': 0, 'max_length':1024}
# )
# model = HuggingFaceHub(repo_id="dbernsohn/t5_wikisql_en2SQL", model_kwargs={'temperature': 0.1, 'max_length':100})

# trying google t5 flan
# model = HuggingFaceHub(
#     repo_id="google/flan-ul2", 
#     model_kwargs={'temperature': 0}
# )

# Trying a different method (downside: more GPU intensive)
# model_id = "dbernsohn/t5_wikisql_en2SQL"
# model = AutoModelForSeq2SeqLM.from_pretrained(model_id, temperature=0)
# device_id = -1 # -1 for CPU, 0 for GPU
# tokenizer = AutoTokenizer.from_pretrained(model_id)
# pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=1024, device=device_id)
# model = HuggingFacePipeline(pipeline)


model = HuggingFaceHub(repo_id="bigcode/starcoderplus", model_kwargs={'temperature': 0.1, 'max_length':100})

# FINAL MODEL
# model = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={'temperature': 0.1, 'max_length':100})

print("-"*50)

# Create the chain
print("Creating chain...")
chain = SQLDatabaseChain.from_llm(model, db, verbose=True, use_query_checker=True)
print("-"*50)

# Run the chain
print("Running chain...")
print(chain.run("How many customers are there?"))
# print(chain.run("How many albums are there by Aerosmith?"))
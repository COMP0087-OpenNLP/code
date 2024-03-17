import argparse
import importlib
import datasetlist
importlib.reload(datasetlist) # Reload instead of using cached version
from datasetlist import cache_embeddings
import time
from typing import List, Optional

def run_gist():
    from models.gist import MyModelGist
    model = MyModelGist()
    cache_embeddings(model.model_name, model, batch_size=64)

def run_voyage():
    from models.voyage import MyModelVoyage
    model = MyModelVoyage()
    cache_embeddings(model.model_name, model, batch_size=128)

def run_llmrails():
    from models.llmrails import MyModellmrails
    model = MyModellmrails()
    cache_embeddings(model.model_name, model, batch_size=64)

def run_flag_embedding():
    from models.flagembedding import MyModelFlagEmbedding
    model = MyModelFlagEmbedding()
    cache_embeddings(model.model_name, model, batch_size=64)

def run_openai():
    from models.myopenai import MyModelOpenAI
    model = MyModelOpenAI()
    cache_embeddings(model.model_name, model, batch_size=2048)
    
def run_angle():
    from models.angle import MyModelAngle
    model = MyModelAngle()
    cache_embeddings(model.model_name, model, batch_size=64)

def run_cohere():
    from models.mycohere import MyModelCohere
    model = MyModelCohere()
    cache_embeddings(model.model_name, model, batch_size=4000) # 4000 arbitrary, but don't want to go too crazy either ngl

RUN_MODELS = {
    "gist": run_gist,
    "voyage": run_voyage,
    "llmrails": run_llmrails,
    "flag-embedding": run_flag_embedding,
    "openai": run_openai,
    "angle": run_angle,
    "cohere": run_cohere
}

def run_for_model(model_name):
    if model_name not in RUN_MODELS:
        raise ValueError(f"Model {model_name} not found")
    
    retries = 50
    while retries > 0:
        try:
            RUN_MODELS[model_name]()
        except Exception as e:
            print(e)
            print("Error occurred, retrying in 60 seconds...")
            retries -= 1
            time.sleep(60)
        else:
            break

def main():
    parser = argparse.ArgumentParser(description="Run the code overnight (hopefully should still run after log out cs computers)")
    parser.add_argument('--model', type=str, help='The name of the model we want to run')
    args = parser.parse_args()
    run_for_model(args.model)

if __name__ == "__main__":
    main()
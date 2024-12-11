from utils import *

hyperparams.model_name = "/tmp/gemma-7b-hf/" # <== Add model weight location here e.g. "/path/to/downloaded/gemma/weights"
hyperparams.qkv_format = "thd"

model = init_te_gemma_model(hyperparams)

print_sample_of_generated_texts(model)
# benchmark_generation(model)
# Import necessary packages and methods
from utils import *


# Default hyperparams, also defined in `utils.py` in class `Hyperparameters`
## !!! `model_name` attr must point to the location of the model weights !!!
hyperparams.model_name = "/bei/datasets/llama-7bf-hf" # <== Add model weight location here
hyperparams.mixed_precision = "bf16"
hyperparams.batch_size = 64
hyperparams.max_seq_length = 512

## Init the model and accelerator wrapper
model = init_te_llama_model(hyperparams)
accelerator, model, optimizer, train_dataloader, lr_scheduler = wrap_with_accelerator(model, hyperparams)



model.eval()
total_loss = 0
optimizer.zero_grad()
train_dataloader = enumerate(train_dataloader)

time_vals = []

with torch.no_grad():
    for _ in range(hyperparams.num_training_steps):
        step, batch = next(train_dataloader)
        start_time = time.time()
        with accelerator.accumulate(model):
            outputs = model(**batch)

        torch.cuda.synchronize()
#         loss = outputs.loss
#         total_loss += loss.detach().float()
#         accelerator.backward(loss)
#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()

        end_time = time.time()
        print(f"batch size: {batch['input_ids'].shape}, total time: {end_time - start_time}, peak gpu mem: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
#     total_time = end_time - start_time
#     time_vals.append(total_time)

# accelerator.end_training()

# # ignore the first couple of time vals
# time_vals = time_vals[2:]
# print(f"{hyperparams.num_training_steps} finetuning steps complete!\nAverage time taken per step: {(sum(time_vals)/len(time_vals)) * 1000:.0f} milliseconds")
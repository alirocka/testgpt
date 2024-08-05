import torch
from transformers import GPT2LMHeadModel,EvalPrediction
from transformers import PreTrainedTokenizerFast
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from scipy.special import softmax
from sklearn.metrics import log_loss
wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizer_BPE3.json",
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>",
    pad_token = "<|endoftext|>",
    
)
tokenizer=wrapped_tokenizer

def compute_metrics(p: EvalPrediction):
    logits = p.predictions
    labels = p.label_ids
    probabilities = softmax(logits, axis=-1)
    loss = log_loss(labels.flatten(), probabilities.reshape(-1, probabilities.shape[-1]), labels=[i for i in range(logits.shape[-1])])
    perplexity = torch.exp(loss)
    
    return {"perplexity": perplexity}
def load_dataset(filepath,tokenizer,blocksize=128):
    dataset=TextDataset(tokenizer=tokenizer,
                        file_path=filepath,
                        block_size=blocksize)
    return dataset



def load_data_collator(tokenizer,mlm=False):
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=mlm)
    return data_collator


def train(train_file_path,
          val_file_path,
          model_name,
          output_dir,
          overwrite_output_dir,
          num_train_epochs,
          per_device_train_batch_size,
          save_steps):
    train_dataset=load_dataset(train_file_path,tokenizer)
    eval_dataset=load_dataset(val_file_path,tokenizer)
    data_collator=load_data_collator(tokenizer)
    model=GPT2LMHeadModel.from_pretrained(model_name)
    model.save_pretrained(output_dir)


    training_args=TrainingArguments(output_dir=output_dir,
                                    overwrite_output_dir=overwrite_output_dir,
                                    per_device_train_batch_size=per_device_train_batch_size,
                                    num_train_epochs=num_train_epochs,
                                    save_steps=save_steps,
                                    evaluation_strategy='steps',
                                    save_strategy="steps",  
                                    load_best_model_at_end=True,  
                                    metric_for_best_model='eval_loss',
    )
    trainer=Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,

    )
    trainer.train()
    trainer.save_model()
 
train(
    train_file_path="train.txt",
    val_file_path="validata.txt",
    model_name="gpt2",
    output_dir="gpt2_finetuned_10ktest",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    num_train_epochs=400,
    save_steps=14000
)

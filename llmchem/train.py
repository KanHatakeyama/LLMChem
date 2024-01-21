import transformers
from datetime import datetime
from .dataset import tokenize_dataset, gen_train_text
import json


def train_model(model, tokenizer, dataset,
                project_dir="",
                epochs=3,
                lr=10**-5,
                per_device_train_batch_size=1,
                gradient_checkpointing=False,
                generation=0,
                ):
    tokenized_dataset = tokenize_dataset(gen_train_text(dataset), tokenizer)
    if epochs == 0:
        return None
    # train
    train_args = transformers.TrainingArguments(
        per_device_train_batch_size=per_device_train_batch_size,
        # gradient_accumulation_steps=1,
        warmup_steps=0,
        num_train_epochs=epochs,
        learning_rate=lr,
        fp16=True,
        logging_steps=100,
        save_total_limit=1,
        output_dir=f'outputs/{generation}_' + \
        datetime.now().strftime('%Y%m%d%H%M%S'),
        gradient_checkpointing=gradient_checkpointing,
    )

    # trainer
    # callbacks = [EarlyStoppingCallback()]
    callbacks = []

    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=train_args,
        callbacks=callbacks,
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False)
    )

    train_result = trainer.train()

    if project_dir != "":
        with open(f"{project_dir}/train/{datetime.now().strftime('%Y%m%d%H%M%S')}.json", "w") as f:
            json.dump(train_result, f, indent=4)
    return train_result

from transformers import TrainingArguments

def get_training_args(output_dir, lr=3e-5, epochs=2): # CHANGE BACK TO 20
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        fp16=False,
        bf16=False,
        eval_strategy="epoch", # val is being used
        save_strategy="epoch",
        save_total_limit=3,
        num_train_epochs=epochs,
        logging_strategy="steps",
        logging_steps=25,
        learning_rate=lr,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        label_names=["labels"],
        dataloader_num_workers=4, # added in last commit
        dataloader_pin_memory=True,
        dataloader_persistent_workers=False,
        dataloader_prefetch_factor=1
        #label_smoothing_factor=0.1,
    )

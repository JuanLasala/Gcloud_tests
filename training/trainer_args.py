from transformers import TrainingArguments

def get_training_args(output_dir, lr=3e-4, epochs=25): #3e-5, 25 epochs
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        fp16=True,
        bf16=False,
        eval_strategy="epoch", # val is being used
        save_strategy="epoch",
        save_total_limit=3,
        num_train_epochs=epochs,
        logging_strategy="steps",
        logging_steps=25,
        learning_rate=lr,
        lr_scheduler_type = "cosine",
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="f1", #F1 O ROC_AUC?
        greater_is_better=True,
        label_names=["labels"],
        dataloader_num_workers=4, # added in last commit
        dataloader_pin_memory=True,
        dataloader_persistent_workers=False,
        dataloader_prefetch_factor=1,
        warmup_ratio=0.1,
        weight_decay=0.01,
        #label_smoothing_factor=0.1,
    )

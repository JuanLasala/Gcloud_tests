from transformers import TrainingArguments

def get_training_args(output_dir, lr=3e-6, epochs=5):
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        fp16=True,
        eval_strategy="epoch",
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
        #label_smoothing_factor=0.1,
    )

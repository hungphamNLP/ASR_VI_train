from build_config import model,config,processor
from dataset_process import timit_train,timit_test,DataCollatorCTCWithPadding
from transformers import Trainer,TrainingArguments,Wav2Vec2ForCTC
from datasets import load_metric
import numpy as np

wer_metric = load_metric("wer")
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

if __name__ == '__main__':
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    #model.gradient_checkpointing_enable()
    model.freeze_feature_encoder()
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        group_by_length=True,
        per_device_train_batch_size=config['batch'],
        evaluation_strategy="steps",
        num_train_epochs=config['num_epoch'],
        fp16=True,
        save_steps=config['save_steps'],
        eval_steps=config['eval_steps'],
        logging_steps=config['logging_steps'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        warmup_steps=config['warmup_steps'],
        save_total_limit=config['save_total_limit'],
        report_to=None
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=timit_train['train'],
        eval_dataset=timit_test['train'],
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    

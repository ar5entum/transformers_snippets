import os
import torch
import librosa
import evaluate
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from datasets import load_from_disk
from typing import Dict, List, Optional, Union
from transformers import Wav2Vec2PhonemeCTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer

tqdm.pandas()

torch.cuda.empty_cache()
PRETRAINED="/data/wav_vec_training/malay_hindi_train/wav2vec_phoneme_hindi_09_14/checkpoint-84000"
OUTDIR = "./wav2vec_phoneme_hindi_09_26"
DATASET = '/data/wav_vec_training/malay_hindi_train/iffco.csv'

try:
    tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(PRETRAINED)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(PRETRAINED)
except:
    
    tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(os.path.dirname(PRETRAINED))
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(os.path.dirname(PRETRAINED))
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
tokenizer.save_pretrained(OUTDIR)
feature_extractor.save_pretrained(OUTDIR)

train_dataset = load_from_disk(os.path.join(OUTDIR, 'train_dataset'))
train_dataset = train_dataset.shuffle()
eval_dataset = load_from_disk(os.path.join(OUTDIR, 'eval_dataset'))


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.AutoProcessor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None
    feature_extractor_input_name: Optional[str] = "input_values"

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [
            {self.feature_extractor_input_name: feature[self.feature_extractor_input_name]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        if "attention_mask" in batch:
            batch["attention_mask"] = batch["attention_mask"].to(torch.long)

        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

print("Loading WER metric...")

wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    label_str = ['$' if l.strip() == '' else l for l in label_str]
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


model = Wav2Vec2ForCTC.from_pretrained(
    PRETRAINED,
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    ignore_mismatched_sizes=True
)
model.freeze_feature_encoder()
model.config.ctc_zero_infinity = True

training_args = TrainingArguments(
    output_dir=OUTDIR,
    num_train_epochs=20,
    group_by_length=False,
    per_device_train_batch_size=60,
    per_device_eval_batch_size=60,
    eval_strategy="epoch",
    save_strategy='epoch',
    logging_steps=5,
    learning_rate=2e-5,
    warmup_steps=50,
    max_grad_norm=1.0,
    dataloader_num_workers=32,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    report_to='tensorboard',
    fp16=True,
)

print("Initializing Trainer...")
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.feature_extractor,
)
print("Trainer initialized.")

print("Starting training...")
trainer.train()
print("Training complete.")
model.save_pretrained(OUTDIR)
processor.save_pretrained(OUTDIR)

model.to('cuda')

def parse_transcription(wav_file):
    try:
        audio_input, sample_rate = librosa.load(wav_file, sr=16000)
        input_values = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_values.to('cuda')
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
        return transcription
    except:
        return "-"

with open(os.path.join(OUTDIR, 'generated_output.txt'), 'w') as f:
    f.write('\n'.join(parse_transcription(d['path']) if os.path.exists(d['path']) else "?" for d in tqdm(eval_dataset, desc='Running Prediction')))




import os
import torch
import random
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import Wav2Vec2PhonemeCTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

tqdm.pandas()
torch.cuda.empty_cache()
PRETRAINED=""
OUTDIR = ""
DATASET = ''

try:
    tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(PRETRAINED)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(PRETRAINED)
except:
    
    tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(os.path.dirname(PRETRAINED))
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(os.path.dirname(PRETRAINED))
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
tokenizer.save_pretrained(OUTDIR)
feature_extractor.save_pretrained(OUTDIR)

print("Loading CSV data...")
data = pd.read_csv(DATASET)
data.dropna(subset='sentence', inplace=True)
data = data[data['path'].str.lower().str.contains('.wav')]

def filter_audio_duration(audio_path, min_time=0.1, max_time=20):
    try:
        audio, sr = librosa.load(audio_path, sr=16000)
        if not np.any(audio):
            return False
        duration = librosa.get_duration(y=audio, sr=sr)
        return min_time <= duration <= max_time
    except:
        return False


data['duration'] = data['path'].progress_apply(filter_audio_duration)
data = data[data['duration'] == True]
data = data.drop(columns=['duration'])

def prepare_dataset(batch):
    audio, _ = librosa.load(batch["path"], sr=16000)
    batch["input_values"] = processor(audio, sampling_rate=16000).input_values[0]
    batch["labels"] = processor(text=batch["sentence"]).input_ids
    return batch

def prepare_dataset_aug(batch):
    audio, _ = librosa.load(batch["path"], sr=16000)
    if random.random() < 0.35:
        noise = np.random.normal(0, 0.03, audio.shape)
        audio = audio + noise
    if random.random() < 0.35:
        speed_factor = 1.5
        audio = librosa.effects.time_stretch(audio, rate=speed_factor)
    batch["input_values"] = processor(audio, sampling_rate=16000).input_values[0]
    batch["labels"] = processor(text=batch["sentence"]).input_ids
    return batch

train_data, eval_data = train_test_split(data, test_size=2000, random_state=42, shuffle=True)
print(f"Number of train samples:\t{len(train_data)}")
print(f"Number of test samples:\t{len(eval_data)}")


print("Creating training dataset...")
train_dataset = Dataset.from_pandas(train_data)
print(f"Mapping training dataset... Number of samples: {len(train_dataset)}")
train_dataset = train_dataset.map(prepare_dataset_aug, desc='Preparing augmented training dataset')
print("Training dataset mapping done.")

print("Creating evaluation dataset...")
eval_dataset = Dataset.from_pandas(eval_data)
print(f"Mapping evaluation dataset... Number of samples: {len(eval_dataset)}")
eval_dataset = eval_dataset.map(prepare_dataset, desc='Preparing evaluation dataset')
print("Evaluation dataset mapping done.")

try:
    print("Saving data")
    train_dataset.save_to_disk(os.path.join(OUTDIR, "train_dataset"))
    eval_dataset.save_to_disk(os.path.join(OUTDIR, "eval_dataset"))
except:
    print("Error in saving data")


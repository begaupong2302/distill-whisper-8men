import os
import json
from pathlib import Path
from datasets import Dataset, Audio, DatasetDict
from huggingface_hub import HfApi
import pandas as pd

CONFIG = {
    "hf_token": "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",
    "repo_name": "parkpoongpa/data4whipser",
    "private": False,
    "audio_dir": "data4whipser",
}

def create_dataset_from_csv(lang_dir):
    datasets = {}
    for split in ['train', 'validation', 'test']:
        csv_path = lang_dir / f"{split}.csv"
        if not csv_path.exists():
            continue
            
        df = pd.read_csv(csv_path)
        df['audio'] = df['audio_path'].apply(lambda x: str(Path(CONFIG["audio_dir"]) / x))
        
        datasets[split] = Dataset.from_pandas(df).cast_column("audio", Audio())
    
    return DatasetDict(datasets) if datasets else None

def main():
    audio_dir = Path(CONFIG["audio_dir"])
    all_datasets = {}
    
    for lang_folder in audio_dir.iterdir():
        if lang_folder.is_dir() and (lang_folder / "train.csv").exists():
            lang = lang_folder.name
            print(f"Processing {lang}...")
            all_datasets[lang] = create_dataset_from_csv(lang_folder)
    
    api = HfApi(token=CONFIG["hf_token"])
    api.create_repo(repo_id=CONFIG["repo_name"], repo_type="dataset", 
                    private=CONFIG["private"], exist_ok=True)
    
    for lang, dataset_dict in all_datasets.items():
        print(f"Pushing {lang}...")
        dataset_dict.push_to_hub(
            CONFIG["repo_name"],
            config_name=lang,
            token=CONFIG["hf_token"],
            private=CONFIG["private"],
            max_shard_size="2GB"
        )
    
    print(f"\nDone: https://huggingface.co/datasets/{CONFIG['repo_name']}")

if __name__ == "__main__":
    main()
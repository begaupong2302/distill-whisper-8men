from huggingface_hub import snapshot_download
import os
os.environ["HF_HUB_DISABLE_XET"] = "2"

snapshot_download(
    repo_id="parkpoongpa/data4whipser",
    repo_type="dataset",
    local_dir="../data4whipser",
    allow_patterns=["vi/*", "zh/*", "de/*", "en/*", "fr/*", "ja/*", "ko/*","metadata.json"]
)

#huggingface-cli download parkpoongpa/data4whisper --repo-type dataset --local-dir ./data4whisper

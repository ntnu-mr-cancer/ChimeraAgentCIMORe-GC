from pathlib import Path
from huggingface_hub import snapshot_download

def resolve_model_path(model_id_or_path: str) -> Path:
    target_path = Path(model_id_or_path)
    # 1. If it's already an existing local folder or file path
    if target_path.exists():
        print(f"Model found at: {model_id_or_path}")
        return target_path

    # 2. Otherwise, resolve from local Hugging Face cache
    try:
        cached_path = snapshot_download(
            repo_id=model_id_or_path,
            local_files_only=True
        )
        return Path(cached_path)
    except Exception:
        # Fallback to downloading if not cached locally, or return raw path
        try:
            print(f"Model {model_id_or_path} was not found. Downloading ...")
            cached_path = snapshot_download(repo_id=model_id_or_path)
            print(f"Completed!")
            return Path(cached_path)
        except Exception:
            return target_path
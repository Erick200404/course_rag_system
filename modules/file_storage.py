"""Legacy upload helper, retained for older integrations."""

import os
from modules.workspace_store import validate_filename


def save_uploaded_file(uploaded_file, save_dir: str = "data") -> str:
    validate_filename(uploaded_file.name)
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, uploaded_file.name)
    temp_path = f"{file_path}.uploading"
    try:
        with open(temp_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
        os.replace(temp_path, file_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return file_path

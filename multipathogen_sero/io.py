import os
import json

def save_metadata_json(directory, metadata_dict, filename="metadata.json"):
    """
    Save metadata_dict as metadata.json in the specified directory.

    Parameters:
        directory (str or Path): Directory where metadata.json will be saved.
        metadata_dict (dict): Dictionary to save as JSON.
    """
    os.makedirs(directory, exist_ok=True)
    json_path = os.path.join(directory, filename)
    with open(json_path, "w") as f:
        json.dump(metadata_dict, f, indent=2)


def load_metadata_json(directory, filename="metadata.json"):
    """
    Load metadata.json from the specified directory.

    Parameters:
        directory (str or Path): Directory where metadata.json is located.

    Returns:
        dict: Loaded metadata dictionary.
    """
    json_path = os.path.join(directory, filename)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} does not exist.")
    with open(json_path, "r") as f:
        metadata_dict = json.load(f)
    return metadata_dict
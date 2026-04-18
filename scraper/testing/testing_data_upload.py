import json
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import firebase.firebase_write

# Get the directory of the current script
script_dir = Path(__file__).parent
data_file_path = script_dir.parent / "data" / "test.json"

# Example scraped data (you would replace this with actual scraped data)
with open(data_file_path, 'r') as f:
    policy_data = json.load(f)


firebase.firebase_write.upload_scraped_policy(policy_data)

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import firebase.firebase_write

data_file_path = Path(__file__).parent.parent / "data" / "test.json"
with open(data_file_path, "r") as f:
    policy_data = json.load(f)

firebase.firebase_write.upload_scraped_policy(policy_data)

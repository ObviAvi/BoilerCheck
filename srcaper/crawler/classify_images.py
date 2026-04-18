import json
import os
import base64
import time
import io
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_DIR = os.path.join(BASE_DIR, "crawler", "data", "images")

JSON_FILES = [
    os.path.join(BASE_DIR, "data", "housing.json"),
    os.path.join(BASE_DIR, "data", "zucrow_2024.json"),
]

IMAGE_TYPES = [
    "diagram_or_flowchart",
    "table_as_image",
    "photo",
    "logo_or_icon",
    "screenshot",
    "floor_plan",
    "signature_or_stamp",
    "other",
]

PROMPT = f"""You are classifying images extracted from Purdue University websites and policy PDFs.

Respond with ONLY a JSON object — no markdown, no explanation. Format:
{{
  "image_type": "<one of: {', '.join(IMAGE_TYPES)}>",
  "description": "<2-3 sentence description useful for a policy assistant RAG system. Focus on what the image communicates, not how it looks.>"
}}"""

DELAY_BETWEEN_CALLS = 1.5


# ── Helpers ───────────────────────────────────────────────────────────────────

def image_to_base64(filepath):
    ext = filepath.rsplit(".", 1)[-1].lower()
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                "jp2": "image/jp2", "webp": "image/webp", "svg": "image/svg+xml"}
    mime = mime_map.get(ext, "image/png")
    with open(filepath, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8"), mime


def classify_image(client, filepath):
    if filepath.endswith(".svg"):
        drawing = svg2rlg(filepath)
        if drawing is None:
            raise ValueError(f"Could not parse SVG: {filepath}")
        buf = io.BytesIO()
        renderPM.drawToFile(drawing, buf, fmt="PNG")
        buf.seek(0)
        b64 = base64.standard_b64encode(buf.read()).decode("utf-8")
        mime = "image/png"
    else:
        b64, mime = image_to_base64(filepath)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(data=base64.b64decode(b64), mime_type=mime),
            types.Part.from_text(text=PROMPT),
        ]
    )
    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_all_images(data):
    """Works whether the JSON is a single dict (PDF) or a list of pages (crawler)."""
    if isinstance(data, list):
        images = []
        for page in data:
            images.extend(page.get("images", []))
        return images
    else:
        return data.get("images", [])


def classify_images_for_data(data: list, json_path):
    """
    Called externally by dynamic_crawlerV2.py to classify images
    in an already-loaded data list. json_path can be None when
    called from the crawler since we don't need to save to disk.
    """
    if not GEMINI_API_KEY:
        raise EnvironmentError("Set the GEMINI_API_KEY environment variable first.")

    client = genai.Client(api_key=GEMINI_API_KEY)
    images = get_all_images(data)

    if not images:
        return

    total = len(images)
    for i, record in enumerate(images):
        filepath = os.path.join(IMAGE_DIR, record.get("filename", ""))

        if record.get("description") and record.get("image_type"):
            continue

        if not filepath or not os.path.exists(filepath):
            continue

        print(f"  [{i+1}/{total}] Classifying: {record['filename']} ...", end=" ", flush=True)

        try:
            result = classify_image(client, filepath)
            record["image_type"]  = result.get("image_type", "other")
            record["description"] = result.get("description", "")
            print(record["image_type"])
        except Exception as e:
            print(f"ERROR — {e}")
            record["image_type"]  = "error"
            record["description"] = ""

        time.sleep(DELAY_BETWEEN_CALLS)


# ── Main (standalone usage) ───────────────────────────────────────────────────

def main():
    if not GEMINI_API_KEY:
        raise EnvironmentError("Set the GEMINI_API_KEY environment variable first.")

    client = genai.Client(api_key=GEMINI_API_KEY)

    for json_path in JSON_FILES:
        if not os.path.exists(json_path):
            print(f"Skipping {json_path} — file not found.")
            continue

        print(f"\nProcessing {json_path} ...")
        data = load_json(json_path)
        images = get_all_images(data)

        if not images:
            print("  No images found.")
            continue

        total = len(images)
        print(f"  Found {total} images.")

        for i, record in enumerate(images):
            filepath = os.path.join(IMAGE_DIR, record.get("filename", ""))

            if record.get("description") and record.get("image_type"):
                print(f"  [{i+1}/{total}] Skipping (already classified): {record['filename']}")
                continue

            if not filepath or not os.path.exists(filepath):
                print(f"  [{i+1}/{total}] File not found, skipping: {filepath}")
                continue

            print(f"  [{i+1}/{total}] Classifying: {record['filename']} ...", end=" ", flush=True)

            try:
                result = classify_image(client, filepath)
                record["image_type"]  = result.get("image_type", "other")
                record["description"] = result.get("description", "")
                print(record["image_type"])
            except Exception as e:
                print(f"ERROR — {e}")
                record["image_type"]  = "error"
                record["description"] = ""

            save_json(json_path, data)
            time.sleep(DELAY_BETWEEN_CALLS)

        print(f"  Done with {json_path}.")

    print("\nAll files processed.")


if __name__ == "__main__":
    main()
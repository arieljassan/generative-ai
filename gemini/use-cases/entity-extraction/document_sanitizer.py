# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extract entities from dirty documents."""
import cv2
import dotenv
import fitz
import json
import numpy as np
import os
from urllib.parse import urlparse

from google import genai
from google.cloud import storage
from google.genai import types

import utils

EXTRACT_LOW_QUALITY_PROMPT_TEMPLATE = """
    The document is {document_name}.
    This document is in a very low quality and difficult to read.
    Please read it thoroughly and pay special attention to characters that
    can be confusing in low quality document.
    Pay special attention between the following:
    * 8 and 3
    * 8 and 6
    This document contains important information, therefore read it thoroughly and
    intentfully to extract it correctly.
    Output as JSON.

    Extract the following fields from the document:

    Fields:\n
    {fields}
"""

# Load environment variables.
dotenv.load_dotenv(override=True)
project_id = os.environ.get("GEMINI_PROJECT_ID")
if not project_id:
    raise ValueError("GEMINI_PROJECT_ID environment variable must be set.")
location = os.environ.get("GEMINI_LOCATION", "global")
config_path = os.environ.get("CONFIG_PATH", "config.json")

# Initialize Gemini client.
client = genai.Client(vertexai=True, project=project_id, location=location)
CONFIGS = utils.load_app_config(config_path)

def download_gcs_file(gcs_uri, local_dest_folder="."):
    """
    Downloads a file from a GCS URI (gs://bucket/path) to a local folder.
    Returns the local path of the downloaded file.
    """
    # Parse the URI
    parsed = urlparse(gcs_uri)
    if parsed.scheme != "gs":
        raise (
            ValueError(
                f"Error: {gcs_uri} is not a valid GCS URI. Must start with 'gs://'"
            )
        )

    bucket_name = parsed.netloc
    blob_name = parsed.path.lstrip("/")

    # Setup local path
    filename = os.path.basename(blob_name)
    local_path = os.path.join(local_dest_folder, filename)

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    print(f"Downloading {filename} to {local_path}...")
    blob.download_to_filename(local_path)
    return local_path


def enhance_cv2_image(img_array):
    """
    Applies the CV2 processing logic to a numpy image array.
    """
    # PyMuPDF loads as RGB, OpenCV expects BGR. Convert to grayscale.
    if len(img_array.shape) == 3:
        # If RGB or BGR, convert to Gray
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # 1. Denoising
    denoised_img = cv2.medianBlur(gray, 5)

    # 2. Gamma Correction
    gamma = 1.5
    gamma_corrected = np.array(255 * (denoised_img / 255) ** gamma, dtype='uint8')

    # 3. Adaptive Thresholding
    enhanced_img = cv2.adaptiveThreshold(
        gamma_corrected,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        1
    )

    return enhanced_img

def preprocess_file(document_uri):
    # Download file from GCS if it's a gs:// URI, otherwise use local path.
    if document_uri.startswith("gs://"):
        try:
            local_file_path = download_gcs_file(document_uri)
        except Exception as e:
            print(f"Failed to download file: {e}")
            return []
    else:
        local_file_path = document_uri

    file_name, file_ext = os.path.splitext(local_file_path)
    file_ext = file_ext.lower()
    output_files = []

    if file_ext == '.pdf':
        print(f"Processing PDF (via PyMuPDF): {local_file_path}")

        # Open the PDF
        doc = fitz.open(local_file_path)

        for i, page in enumerate(doc):
            # Render page to image (pixmap)
            pix = page.get_pixmap(dpi=300)

            # Convert the buffer to a numpy array
            img_array = (
                np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            )

            # Handle potential Alpha channel (Transparency) if strictly RGB is needed.
            # If RGBA, drop Alpha.
            if pix.n >= 4:
                img_array = img_array[:,:,:3]

            processed = enhance_cv2_image(img_array)

            # Save individual page.
            out_name = f"{file_name}_page_{i+1}_enhanced.png"
            cv2.imwrite(out_name, processed)
            output_files.append(out_name)

        doc.close()

    else:
        print(f"Processing Image: {local_file_path}")
        img = cv2.imread(local_file_path)
        if img is None:
            print(f"Error: Could not read image {local_file_path}")
            return None

        processed = enhance_cv2_image(img)
        out_name = f"{file_name}_enhanced.png"
        cv2.imwrite(out_name, processed)
        output_files.append(out_name)

    return output_files


def evaluate_document_quality(document_uri):
    TEXT_DOCUMENT_QUALITY = """
        Evaluate the quality of the following image based on whether you can
        read the text in the image clearly. Classify between "good" and "bad".

        Output:
        {
            "quality": "good" or "bad"
        }
    """
    if document_uri.startswith("gs://"):
        doc_part = types.Part.from_uri(
            file_uri=document_uri,
            mime_type="application/pdf",
        )
    else:
        bytes_data, mime_type = get_bytes_from_file(document_uri)
        doc_part = types.Part.from_bytes(
            data=bytes_data,
            mime_type=mime_type,
        )

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[
            doc_part,
            TEXT_DOCUMENT_QUALITY,
        ],
        config={
            "response_mime_type": "application/json",
        },
    )

    response_json = json.loads(response.text)
    return response_json.get("quality")

def get_bytes_from_file(file_path):
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    print(f"Detected {ext}, reading bytes directly...")
    with open(file_path, "rb") as f:
        image_bytes = f.read()

    if ext in ['.jpg', '.jpeg']:
        mime_type = 'image/jpeg'
    elif ext == '.png':
        mime_type = 'image/png'
    elif ext == '.pdf':
        mime_type = 'application/pdf'
    else:
        raise ValueError(
            f"Unsupported format for direct reading: {ext}"
        )
    return image_bytes, mime_type


def send_local_image_to_gemini(image_path, text_prompt, model):
    """Loads a local image file, prepares it, and sends it to Gemini."""

    print(f"Loading local image: {image_path}")

    image_bytes, mime_type = get_bytes_from_file(image_path)

    image_blob = types.Blob(
        data=image_bytes,
        mime_type=mime_type
    )

    contents = [
        types.Part.from_text(text=text_prompt),
        types.Part(inline_data=image_blob),
    ]

    generate_content_config = types.GenerateContentConfig(
        response_mime_type = "application/json",
    )

    print("Sending request to Gemini API...")
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )

    return response


def extract_data_from_low_quality_document(extract_config_id: str, document_path: str):
    extract_config = CONFIGS["extraction_configs"][extract_config_id]

    prompt = EXTRACT_LOW_QUALITY_PROMPT_TEMPLATE.format(
        document_name=extract_config["document_name"],
        fields=json.dumps(extract_config["fields"], indent=4),
    )

    response = send_local_image_to_gemini(
        image_path=document_path,
        text_prompt=prompt,
        model="gemini-2.5-pro",
    )

    # # Enforce Indentation (Post-Processing)
    try:
        data = json.loads(response.text)
        formatted_text = json.dumps(data, indent=4)

        return formatted_text

    except json.JSONDecodeError:
        print("Warning: Model output was not valid JSON.")
        return response.text
    except Exception as e:
        print(f"An error occurred: {e}")
        return response.text

"""Scratch script for trying the Roboflow hosted bib-detection workflow.

Not part of the VideoPacer pipeline, so it is excluded from linting and type
checking. It needs the optional ``inference-sdk`` package and an API key.

Set the key in your environment before running, e.g.:

    $env:ROBOFLOW_API_KEY = "<your key>"
"""

import os

# 1. Import the library
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

# 2. Connect to your workflow
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ["ROBOFLOW_API_KEY"],
).configure(
    InferenceConfiguration(
        api_key_transport="header"  # header-based auth (inference v1.5.0+)
    )
)

# 3. Run your workflow on an image
result = client.run_workflow(
    workspace_name="thomas-lamalle-proton-me",
    workflow_id="bib-detection-vbib-detection-ysbht-1-rfdetr-nano-t1-logic",
    images={
        "image": "YOUR_IMAGE.jpg"  # Path to your image file
    },
    use_cache=True,  # Speeds up repeated requests
)

# 4. Get your results
print(result)

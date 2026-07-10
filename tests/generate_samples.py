"""Prepares sample assets used by the agent's tools: a synthetic invoice
image (for the OCR tool) and a real photo with people (for the object
counting tool)."""
import cv2
import numpy as np
import urllib.request
import os

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

# --- Synthetic invoice for OCR tool ---
img = np.ones((500, 700, 3), dtype=np.uint8) * 255
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, "ACME SUPPLIES INVOICE", (40, 60), font, 0.9, (0, 0, 0), 2)
cv2.putText(img, "Item A: 450", (40, 150), font, 0.85, (0, 0, 0), 2)
cv2.putText(img, "Item B: 1200", (40, 200), font, 0.85, (0, 0, 0), 2)
cv2.putText(img, "Item C: 800", (40, 250), font, 0.85, (0, 0, 0), 2)
cv2.putText(img, "Please review totals above.", (40, 330), font, 0.6, (0, 0, 0), 1)
cv2.imwrite(os.path.join(SAMPLES_DIR, "invoice.png"), img)
print("Wrote samples/invoice.png")

# --- Real photo for object detection tool ---
bus_path = os.path.join(SAMPLES_DIR, "street_scene.jpg")
if not os.path.exists(bus_path):
    url = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg"
    urllib.request.urlretrieve(url, bus_path)
    print("Downloaded samples/street_scene.jpg")
else:
    print("samples/street_scene.jpg already present")

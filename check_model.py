# Sanity check for the ad detector. Run: .venv/Scripts/python check_model.py [image]
# The class ids must match k_adClasses in SentisModelEditorConverter.cs and the order in SentisYoloClasses.txt
import sys
from ultralytics import YOLO

AD_CLASSES = {46: "Billboard", 403: "Poster"}
model = YOLO("yolov8s-oiv7.pt")
for i, name in AD_CLASSES.items():
    assert model.names[i] == name, f"class {i} is {model.names[i]!r}, expected {name!r}"

r = model.predict(sys.argv[1] if len(sys.argv) > 1 else ".venv/times_square.jpg", classes=list(AD_CLASSES), conf=0.15, verbose=False)[0]
assert len(r.boxes) > 0, "no ads detected"
r.save("pred.jpg")
print(f"ok: {len(r.boxes)} ads, see pred.jpg")

from pathlib import Path

from mmseg.apis import MMSegInferencer


img_dir = Path("../../data/真实数据")
inferencer = MMSegInferencer(
    model="../../deliverables/oilleak-20260615/oilleak-seg/v1/oilleak-b-20260615.py",
    weights="../../deliverables/oilleak-20260615/oilleak-seg/v1/iter_96240.pth"
)

jpgs = list(img_dir.glob("*.jpg"))
jpgs = [str(jpg) for jpg in jpgs]
inferencer(
    jpgs,
    out_dir="../../deliverables/oilleak-20260615/oilleak-seg/v1/真实数据",
    img_out_dir="vis",
    pred_out_dir="pred",
)

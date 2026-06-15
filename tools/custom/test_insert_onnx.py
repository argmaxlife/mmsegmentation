import numpy as np
import onnxruntime as ort
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=[2, 3], keepdim=True)

x = torch.randn(1, 1024, 16, 16)
model = Model()
model.eval()
torch.onnx.export(
    model,
    (x,),
    "./mean.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=16,
)

expr_dir = "../../experiments/sly_oilleak_20251010_v7.4.3-b-384"
patchs = [
    f"{expr_dir}/aap1.onnx",
    f"{expr_dir}/aap2.onnx",
    f"{expr_dir}/aap3.onnx",
    f"{expr_dir}/aap6.onnx",
]

x = np.random.rand(1, 1024, 16, 29).astype(np.float32)
for patch in patchs:
    print(f"test {patch}")
    session = ort.InferenceSession(patch, providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"input": x})
    for output in outputs:
        print(output.shape)

print("test patch done")
session = ort.InferenceSession(f'{expr_dir}/segmentation-oilleak-b-384-last-resort.onnx', providers=["CPUExecutionProvider"])
x = np.random.randint(0, 256, (1, 3, 512, 910), dtype=np.uint8)
outputs = session.run(None, {"input": x})
for output in outputs:
    print(output.shape)
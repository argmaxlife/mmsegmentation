import onnx
import torch


class Model(torch.nn.Module):
    def __init__(self, output_size):
        super(Model, self).__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        return self.pool(x)

x = torch.rand(1, 1024, 16, 29)
# x = torch.rand(1, 1024, 16, 16)
expr_dir = "../../experiments/sly_oilleak_20251010_v7.4.3-b-384"

for output_size in [1, 2, 3, 6]:
    export_path = f"{expr_dir}/aap{output_size}.onnx"
    model = Model(output_size)
    model.eval()
    torch.onnx.export(
        model,
        (x,),
        export_path,
        input_names=["input"],
        output_names=["output"],
        external_data=False,
    )
    model = onnx.load(export_path)
    model_lower_version = onnx.version_converter.convert_version(model, target_version=16)
    export_path = f"{expr_dir}/aap{output_size}.onnx"
    onnx.save(model_lower_version, export_path)
import argparse
import cv2 as cv
import importlib.util
import numpy as np
from pathlib import Path

import torch

import init_env  # noqa
from mmseg.apis import inference_model
from mmseg.apis import init_model


class DumpyAdaptiveAvgPool2d(torch.nn.Module):
    def __init__(self, output_size, channels):
        super().__init__()
        self.output_size = output_size
        self.channels = channels

    def forward(self, x):
        return torch.rand(1, self.channels, self.output_size, self.output_size)


class Model(torch.nn.Module):
    def __init__(self, config: str, checkpoint: str):
        super(Model, self).__init__()
        self.model = init_model(config, checkpoint)
        delattr(self.model, "data_preprocessor")
        self.register_buffer(
            "preprocess_mean",
            torch.tensor(
                [[[[123.675]], [[116.28]], [[103.53]]]],
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "preprocess_std",
            torch.tensor(
                [[[[58.395]], [[57.12]], [[57.375]]]],
                dtype=torch.float32,
            ),
        )

    def forward(self, inputs: torch.Tensor):
        floating = inputs.float()
        normalized = (floating - self.preprocess_mean) / self.preprocess_std

        batch_img_metas = [
            dict(
                ori_shape=normalized.shape[2:],
                img_shape=normalized.shape[2:],
                pad_shape=normalized.shape[2:],
                padding_size=[0, 0, 0, 0],
            )
        ] * normalized.shape[0]

        logits = self.model.inference(normalized, batch_img_metas)

        return (logits,)


def main():
    parser = argparse.ArgumentParser(description="mmseg prediction")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="checkppoint path"
    )
    parser.add_argument("--config", type=str, required=True, help="config file")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument(
        "--size", type=int, nargs="+", default=[512, 910], help="input size"
    )
    parser.add_argument("--name", type=str, help="export name")
    args = parser.parse_args()

    checkpoint = args.checkpoint
    config = args.config
    device = args.device
    size = args.size

    checkpoint = Path(checkpoint)
    cleaned = checkpoint.parent / (checkpoint.stem + "_clean" + checkpoint.suffix)
    if not cleaned.exists():
        try:
            model = torch.load(checkpoint, weights_only=False)
        except Exception as e:
            print(f"torch.load exception: {e}, try again")
            model = torch.load(checkpoint)
        obj = {"meta": model["meta"], "state_dict": model["state_dict"]}
        torch.save(obj, cleaned)

    baseline = init_model(config, str(cleaned), device=device)
    model = Model(config, str(cleaned))
    model.to(device)
    model.eval()

    img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    inputs = torch.from_numpy(rgb).unsqueeze(0).to(device)
    inputs = inputs.permute(0, 3, 1, 2)

    baseline_output = inference_model(baseline, img)
    baseline_logits = baseline_output.seg_logits.data.unsqueeze(0)
    logits = model(inputs)[0]
    iseq = torch.equal(baseline_logits, logits)
    print(f"Are the outputs of original and custom model equal? {iseq}")

    module_name = Path(config).stem
    spec = importlib.util.spec_from_file_location(module_name, config)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    channels = module.model["decode_head"]["in_channels"][-1]
    print(f"The last input channels of decoder head: {channels}")
    name = args.name or checkpoint.stem
    export_path = checkpoint.parent / f"{name}.onnx"
    if not export_path.exists():
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.AdaptiveAvgPool2d):
                continue

            print(f"replace AdaptiveAvgPool2d {name}")
            os = module.output_size
            if os in [1, 2, 3]:
                pool = DumpyAdaptiveAvgPool2d(os, channels).to(device)
                model.model.decode_head.psp_modules[os - 1][0] = pool
            elif os in [6]:
                pool = DumpyAdaptiveAvgPool2d(os, channels).to(device)
                model.model.decode_head.psp_modules[3][0] = pool

        print(model)
        torch.onnx.export(
            model,
            (inputs,),
            export_path,
            input_names=["input"],
            output_names=["output"],
            opset_version=16,
            external_data=False,
        )

    print("DONE")


if __name__ == "__main__":
    main()

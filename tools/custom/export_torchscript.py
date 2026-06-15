import argparse
import cv2 as cv
import numpy as np
from pathlib import Path

import torch

import init_env  # noqa
from mmseg.apis import inference_model
from mmseg.apis import init_model


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
        nchw = inputs.permute(0, 3, 1, 2)
        floating = nchw.float()
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

    img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    inputs = torch.from_numpy(rgb).unsqueeze(0).to(device)

    baseline_output = inference_model(baseline, img)
    baseline_logits = baseline_output.seg_logits.data.unsqueeze(0)
    logits = model(inputs)[0]
    iseq = torch.equal(baseline_logits, logits)
    print(f"Are the outputs of original and custom model equal? {iseq}")

    name = args.name or checkpoint.stem
    export_path = checkpoint.parent / f"{name}.torchscript"
    if not export_path.exists():
        ts = torch.jit.trace(model, inputs)
        ts.save(export_path)

    torchscript = torch.jit.load(export_path)
    torchscript_logits = torchscript(inputs)[0]
    iseq = torch.equal(logits, torchscript_logits)
    print(f"Are the outputs of pytorch and torchscript model equal? {iseq}")
    if not iseq:
        err = logits - torchscript_logits
        err_min = err.min()
        err_max = err.max()
        err_avr = err.mean()
        print(f"err_min = {err_min}, err_max = {err_max}, err_avr = {err_avr}")

    print("DONE")


if __name__ == "__main__":
    main()

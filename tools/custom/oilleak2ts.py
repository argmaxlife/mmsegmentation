import argparse
import cv2 as cv
import numpy as np
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, os.getcwd())
from mmseg.apis import inference_model
from mmseg.apis import init_model


class PureModel(torch.nn.Module):
    def __init__(self, config, checkpoint, device):
        super(PureModel, self).__init__()
        self.model = init_model(config, checkpoint)
        delattr(self.model, "data_preprocessor")
        self.model.to(device)
        self.register_buffer(
            "mean",
            torch.tensor(
                [[[[123.675]], [[116.28]], [[103.53]]]],
                dtype=torch.float32,
                device=device,
            ),
        )
        self.register_buffer(
            "std",
            torch.tensor(
                [[[[58.395]], [[57.12]], [[57.375]]]],
                dtype=torch.float32,
                device=device,
            ),
        )

    def forward(self, inputs: torch.Tensor):
        nchw = inputs.permute(0, 3, 1, 2)
        floating = nchw.float()
        normalized = (floating - self.mean) / self.std

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
        "--checkpoint", "-ckt", type=str, required=True, help="checkppoint path"
    )
    parser.add_argument("--config", "-cfg", type=str, required=True, help="config file")
    args = parser.parse_args()

    device = "cuda:0"
    config = args.config
    checkpoint = Path(args.checkpoint)
    pure_checkpoint = checkpoint.parent / (
        checkpoint.stem + "_pure" + checkpoint.suffix
    )
    if not pure_checkpoint.exists():
        try:
            model_data = torch.load(checkpoint, weights_only=False)
        except Exception as e:
            print(f"torch.load exception: {e}, try again")
            model_data = torch.load(checkpoint)
        obj = {"meta": model_data["meta"], "state_dict": model_data["state_dict"]}
        torch.save(obj, pure_checkpoint)

    model = PureModel(config, str(pure_checkpoint), device)
    model.to(device)

    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    inputs = torch.from_numpy(rgb).unsqueeze(0).to(device)

    in_bin = checkpoint.with_name("in.bin")
    if in_bin.exists():
        data = np.fromfile(in_bin, dtype=np.uint8)
        img = data.reshape(512, 512, 3)
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        inputs = torch.from_numpy(rgb).unsqueeze(0).to(device)

    logits = model(inputs)[0]
    print(f"output size {logits.size()}")

    out_bin = checkpoint.with_name("out.bin")
    if in_bin.exists() and out_bin.exists():
        data = np.fromfile(out_bin, dtype=np.float32)
        logits_cpp = torch.from_numpy(data.reshape(1, 3, 512, 512))
        logits_cpp = logits_cpp.to(device)

        eq_logits = torch.equal(logits,logits_cpp)
        print(f"PureModel vs C++: same output? {eq_logits}")

    export_path = checkpoint.with_name("segmentation-oilleak-GPU.torchscript")
    if not export_path.exists():
        ts = torch.jit.trace(model, inputs)
        ts.save(export_path)

        # ts = torch.jit.script(model)
        # ts.save(export_path)

        # torch._dynamo.mark_dynamic(inputs, (1, 2))
        # compiled = torch.compile(model)
        # ts = torch.jit.script(compiled)
        # ts.save(export_path)

        ts_model = torch.jit.load(export_path)
        logits_ts = ts_model(inputs)[0]
        eq_logits = torch.equal(logits, logits_ts)
        print(f"PureModel vs TorchScript: same output? {eq_logits}")

        ori_model = init_model(config, str(pure_checkpoint), device=device)
        result = inference_model(ori_model, img)
        print(f"{result.seg_logits.data.size()} {result.pred_sem_seg.data.size()}")
        logits_ori = result.seg_logits.data.unsqueeze(0)
        print(f"{logits_ori.size()}")

        dl = logits_ori - logits
        dl_min = dl.min()
        dl_max = dl.max()
        dl_avr = dl.mean()

        print(f"PureModel vs Ori: same logits? {torch.equal(logits, logits_ori)}")
        print(f"dl_min = {dl_min}, dl_max = {dl_max}, dl_avr = {dl_avr}")

    print("DONE")


if __name__ == "__main__":
    main()

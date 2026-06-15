import argparse
import cv2 as cv
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
import numpy as np

import torch
import torch.distributed as dist

import init_env  # noqa
from mmseg.apis import init_model


if not dist.is_initialized():
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:29500",
        world_size=1,
        rank=0,
    )


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
    parser.add_argument("--config", type=str, required=True, help="config file")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument(
        "--size", type=int, nargs="+", default=[512, 910], help="input size"
    )
    parser.add_argument("--name", type=str, help="export name")
    args = parser.parse_args()

    config = args.config
    device = args.device
    size = args.size

    model = Model(config, None)
    model.to(device)

    img = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    inputs = torch.from_numpy(rgb).unsqueeze(0).to(device)
    flops = FlopCountAnalysis(model, inputs)
    print(flop_count_table(flops))
    print("DONE")


if __name__ == "__main__":
    main()

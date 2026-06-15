import argparse
import cv2 as cv
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
import numpy as np
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.getcwd())
from mmseg.apis import init_model


if not dist.is_initialized():
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:29600",
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
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="checkppoint path"
    )
    parser.add_argument("--config", type=str, required=True, help="config file")
    parser.add_argument("--device", type=str, default="cpu", help="device")
    args = parser.parse_args()

    checkpoint = args.checkpoint
    config = args.config
    device = args.device
    
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

    model = Model(config, None)
    model.to(device)
    model.eval()

    for n, m in model.named_parameters():
        # if "spm" not in n:
        #     continue
        # # print(n)
        # if "model.backbone.spm.convnext.stages.0.layers.0.depthwise_conv.weight" in n:
        #     print(f"{n} = {m.detach().numpy().tolist()[0][:100]}, {m.requires_grad}")
        #     break
        if not m.requires_grad:
            print(n)


if __name__ == "__main__":
    main()

import math
import torch


class TestModel(torch.nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d(6)
        # self.pool = torch.nn.AvgPool2d(6, 6)
    
    def forward(self, x):
        return self.pool(x)


if __name__ == "__main__":
    x = torch.rand(1, 1, 16, 16, dtype=torch.float32, device="cpu") * 2000 - 1000
    pool_scales = (1, 2, 3, 6)
    base_modules = []
    for pool_scale in pool_scales:
        base_modules.append(torch.nn.AdaptiveAvgPool2d(pool_scale).to(x.device))

    base_outputs = [base_module(x) for base_module in base_modules]

    kernel_sizes = [16, 8, 6, 3]
    strides = [16, 8, 5, 3]

    identity_outputs = []
    for kernel_size, stride, pool_scale in zip(kernel_sizes, strides, pool_scales):
        print(f"kernel_size: {kernel_size}, stride: {stride}")
        identity_module = torch.nn.AvgPool2d(kernel_size, stride).to(x.device)
        if pool_scale in [1, 2, 3]:
            identity_outputs.append(identity_module(x))
        else:
            output = torch.zeros(
                (x.shape[0], x.shape[1], pool_scale, pool_scale), dtype=torch.float32
            )
            i0s = (torch.arange(pool_scale) * x.shape[-2] / pool_scale).floor().long()
            i1s = (torch.arange(1, pool_scale + 1) * x.shape[-2] / pool_scale).ceil().long()
            j0s = (torch.arange(pool_scale) * x.shape[-1] / pool_scale).floor().long()
            j1s = (torch.arange(1, pool_scale + 1) * x.shape[-1] / pool_scale).ceil().long()
            for i in range(pool_scale):
                for j in range(pool_scale):
                    # output[:, :, i, j] = torch.mean(x[:, :, i0s[i]:i1s[i], j0s[j]:j1s[j]], dim=(2, 3))
                    i0 = math.floor(i * x.shape[2] / pool_scale)
                    i1 = math.ceil((i + 1) * x.shape[2] / pool_scale)
                    j0 = math.floor(j * x.shape[3] / pool_scale)
                    j1 = math.ceil((j + 1) * x.shape[3] / pool_scale)
                    output[:, :, i, j] = torch.mean(x[:, :, i0:i1, j0:j1], dim=(2, 3))
            identity_outputs.append(output)

    for base_output, identity_output in zip(base_outputs, identity_outputs):
        isequal = torch.equal(base_output, identity_output)
        print(base_output)
        print(identity_output)
        print(
            f"base {base_output.shape}, identity {identity_output.shape}, isequal? {isequal}"
        )
        if base_output.shape == identity_output.shape:
            delta = torch.abs(base_output - identity_output).max()
            print(f"max abs delta: {delta}")
        print("\n")

    for i in range(3):
        start = math.floor(i * 16 / 3)
        end = math.ceil((i + 1) * 16 / 3)
        ids = list(range(start, end))
        print(f"start: {start}, end: {end}, {ids}")

    print("\n")
    for i in range(6):
        start = math.floor(i * 16 / 6)
        end = math.ceil((i + 1) * 16 / 6)
        ids = list(range(start, end))
        print(f"start: {start}, end: {end}, {ids}")

    
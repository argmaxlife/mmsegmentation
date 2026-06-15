from fvcore.nn import FlopCountAnalysis

inputs = torch.randn(1, 3, 224, 224)
flops = FlopCountAnalysis(model, inputs)

print(flops.total())          # 总 FLOPs
print(flops.by_module())      # 按模块统计

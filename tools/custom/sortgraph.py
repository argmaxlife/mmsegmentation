import onnx_graphsurgeon as gs
import onnx

expr_dir = "../../experiments/sly_oilleak_20251010_v7.4.3-b-384"
model = onnx.load(f'{expr_dir}/segmentation-oilleak-b-384-512-910-last.onnx')

nodes = list(model.graph.node)

# 所有 producer/output 名称可被认为是“生产者”
produced = set()

# 1) graph inputs本身就是可用的 producer
for inp in model.graph.input:
    produced.add(inp.name)

# 2) 把 initializer 也当成 producer
for init in model.graph.initializer:
    produced.add(init.name)

sorted_nodes = []
while nodes:
    matched = False
    for node in nodes:
        # 所有输入要么为空、要么在 produced 里
        if all((inp == "" or inp in produced) for inp in node.input):
            sorted_nodes.append(node)
            # 把这个节点的 outputs 标记成由它“生成”
            for out in node.output:
                produced.add(out)
            nodes.remove(node)
            matched = True
            break
    if not matched:
        raise RuntimeError("Graph has cycle or missing producer!")

model.graph.ClearField('node')
model.graph.node.extend(sorted_nodes)

print("resort done")
onnx.checker.check_model(model, full_check=True)
print("check done")

onnx.save(model, f'{expr_dir}/segmentation-oilleak-b-384-512-910-last-resort.onnx')
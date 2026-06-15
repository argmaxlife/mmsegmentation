import onnx
import onnx_graphsurgeon as gs

names = [
    "/psp_modules/psp_modules.0/psp_modules.0.0/RandomUniform",
    "/psp_modules/psp_modules.1/psp_modules.1.0/RandomUniform",
    "/psp_modules/psp_modules.2/psp_modules.2.0/RandomUniform",
    "/psp_modules/psp_modules.3/psp_modules.3.0/RandomUniform",
]

shapes = [
    [1, 1024, 1, 1], [1, 1024, 2, 2], [1, 1024, 3, 3], [1, 1024, 6, 6],
]

expr_dir = "../../experiments/sly_oilleak_20251010_v7.4.3-b-384"
model = gs.import_onnx(onnx.load(f'{expr_dir}/segmentation-oilleak-b-384-512-910.onnx'))

nodes = [n for n in model.nodes if n.name in names]
new_input_names = [node.outputs[0].name for node in nodes]
shape_info = {name: shape for name, shape in zip(new_input_names, shapes)}

for node in model.nodes:
    for input_var in node.inputs:
        if input_var.name in new_input_names:
            if input_var.dtype is None:
                input_var.dtype = onnx.TensorProto.FLOAT
                print(f"Set dtype for input {input_var.name}")
            if input_var.shape is None:
                input_var.shape = shape_info[input_var.name]
                print(f"Set shape for input {input_var.name}: {input_var.shape}")
            model.inputs.append(input_var)
            print(f"add new input: {input_var.name}")

for node in nodes:
    model.nodes.remove(node)

onnx.save(gs.export_onnx(model), f'{expr_dir}/segmentation-oilleak-b-384-512-910-marked.onnx')
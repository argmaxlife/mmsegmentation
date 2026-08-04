import onnx


def remove_noop_attribute(model_path, output_path=None):
    model = onnx.load(model_path)
    
    for node in model.graph.node:
        if node.op_type == "ReduceMean":
            new_attrs = []
            for attr in node.attribute:
                if attr.name != "noop_with_empty_axes":
                    if attr.name == "axes":
                        if "aap1.onnx" in model_path:
                            print(f"old attr.ints: {attr.ints}")
                            attr.ints.extend([2, 3])
                            print(f"new attr.ints: {attr.ints}")
                        elif "aap2.onnx" in model_path:
                        # elif "aap3.onnx" in model_path:
                            print(f"old attr.ints: {attr.ints}")
                            attr.ints.extend([3, 5])
                            print(f"new attr.ints: {attr.ints}")
                        print(f"update axes for {model_path}")
                    new_attrs.append(attr)
                else:
                    print(f"Removed noop_with_empty_axes from node: {node.name}")
            
            del node.attribute[:]
            node.attribute.extend(new_attrs)

    if output_path:
        onnx.save(model, output_path)
        print(f"Saved fixed model to: {output_path}")
    
    return model

x = [
    ("output", "/psp_modules/psp_modules.0/psp_modules.0.0/RandomUniform_output_0"),
    ("output", "/psp_modules/psp_modules.1/psp_modules.1.0/RandomUniform_output_0"),
    ("output", "/psp_modules/psp_modules.2/psp_modules.2.0/RandomUniform_output_0"),
    ("output", "/psp_modules/psp_modules.3/psp_modules.3.0/RandomUniform_output_0"),
]

expr_dir = "../../deliverables/oilleak-20260615/oilleak-seg/v1"
patchs = [
    f"{expr_dir}/aap1.onnx",
    f"{expr_dir}/aap2.onnx",
    f"{expr_dir}/aap3.onnx",
    f"{expr_dir}/aap6.onnx",
]

main_model = f'{expr_dir}/segmentation-oilleak-b-384-512-910-marked.onnx'
for i in range(4):
    model1 = onnx.load(patchs[i])
    if i in [0, 1]:
    # if i in [0, 2]:
        model1 = remove_noop_attribute(patchs[i], patchs[i])

    model2 = onnx.load(main_model)
    model1.ir_version = model2.ir_version
    io_map = [x[i]]
    model2 = onnx.compose.merge_models(model1, model2, io_map, prefix1=f"patch{i}_")
    main_model = f'{expr_dir}/segmentation-oilleak-b-384-512-910-patch{i}.onnx'
    onnx.save(model2, main_model)

graph = model2.graph
source = "/backbone/Transpose_3_output_0"
targets = ["patch0_input", "patch1_input", "patch2_input", "patch3_input"]

for node in graph.node:
    for i, inp in enumerate(node.input):
        if inp in targets:
            idx = targets.index(inp)
            node.input[i] = source

inputs = [x for x in graph.input if x.name not in targets]
graph.ClearField("input")
graph.input.extend(inputs)

main_model = f'{expr_dir}/segmentation-oilleak-b-384-512-910-last.onnx'
onnx.save(model2, main_model)

print("DONE, next call sortgraph.py")
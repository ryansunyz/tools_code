import os
import sys
import paddle.fluid.proto.framework_pb2 as framework_pb2

infile = open("model", "rb")
content = infile.read()
infile.close()

prog = framework_pb2.ProgramDesc()
prog.ParseFromString(content)

vars = prog.blocks[0].vars
ops = prog.blocks[0].ops

for op in ops:
    if op.type == "reshape2":
        shape = []
        if op.outputs[0].arguments[0] == "reshape2_0.tmp_0":
            shape = [1,99,2704]
        elif op.outputs[0].arguments[0] == "reshape2_1.tmp_0":
            shape = [1,99,676]
        elif op.outputs[0].arguments[0] == "reshape2_2.tmp_0":
            shape = [1,99,169]
        elif op.outputs[0].arguments[0] == "reshape2_3.tmp_0":
            shape = [1,36,2704]
        elif op.outputs[0].arguments[0] == "reshape2_4.tmp_0":
            shape = [1,36,676]
        elif op.outputs[0].arguments[0] == "reshape2_5.tmp_0":
            shape = [1,36,169]

        for attr in op.attrs:
            if attr.name == "shape":
                attr.ints.pop()
                attr.ints.pop()
                attr.ints.pop()
                attr.ints.extend(shape)

for var in vars:
    if var.name == "reshape2_0.tmp_0":
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.extend([1,99,2704])
    elif var.name == "reshape2_1.tmp_0":
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.extend([1,99,676])
    elif var.name == "reshape2_2.tmp_0":
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.extend([1,99,169])
    elif var.name == "reshape2_3.tmp_0":
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.extend([1,36,2704])
    elif var.name == "reshape2_4.tmp_0":
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.extend([1,36,676])
    elif var.name == "reshape2_5.tmp_0":
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.extend([1,36,169])
    elif var.name == "concat_2.tmp_0":
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.extend([1,99,3549])
    elif var.name == "concat_3.tmp_0":
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.extend([1,36,3549])
    elif var.name == "concat_4.tmp_0":
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.extend([1,135,3549])
    elif var.name == "save_infer_model/scale_0":
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.pop()
        var.type.lod_tensor.tensor.dims.extend([1,135,3549])
    else:
        if len(var.type.lod_tensor.tensor.dims) > 0 and var.type.lod_tensor.tensor.dims[0] == -1:
            var.type.lod_tensor.tensor.dims[0] = 1
        
content = prog.SerializeToString()
outfile = open("changed_model", "wb")
outfile.write(content)
outfile.close()
        
        

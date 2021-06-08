import os
import sys
import paddle.fluid.proto.framework_pb2 as framework_pb2

infile = open("__model__", "rb")
content = infile.read()
infile.close()

prog = framework_pb2.ProgramDesc()
prog.ParseFromString(content)

vars = prog.blocks[0].vars
ops = prog.blocks[0].ops


def modify_reshape2_transpose2():
    for op in ops:
        if op.type == "transpose2":
            axis = [1, 0, 2, 3]
        for attr in op.attrs:
            if attr.name == "axis":
                if len(attr.ints) >= 4:
                    attr.ints.pop()
                    attr.ints.pop()
                    attr.ints.pop()
                    attr.ints.pop()
                    attr.ints.pop()
                    attr.ints.extend(axis)

        if op.type == "reshape2":
            shape = []
            if op.outputs[0].arguments[0] == "reshape2_1.tmp_0":
                shape = [1,116,52,52]
            elif op.outputs[0].arguments[0] == "reshape2_3.tmp_0":
                shape = [1,116,52,52]
            elif op.outputs[0].arguments[0] == "reshape2_5.tmp_0":
                shape = [1,116,52,52]
            elif op.outputs[0].arguments[0] == "reshape2_7.tmp_0":
                shape = [1,116,52,52]            
            elif op.outputs[0].arguments[0] == "reshape2_9.tmp_0":
                shape = [1,232,26,26]            
            elif op.outputs[0].arguments[0] == "reshape2_11.tmp_0":
                shape = [1,232,26,26]            
            elif op.outputs[0].arguments[0] == "reshape2_13.tmp_0":
                shape = [1,232,26,26]            
            elif op.outputs[0].arguments[0] == "reshape2_15.tmp_0":
                shape = [1,232,26,26]  
            elif op.outputs[0].arguments[0] == "reshape2_17.tmp_0":
                shape = [1,232,26,26]            
            elif op.outputs[0].arguments[0] == "reshape2_19.tmp_0":
                shape = [1,232,26,26]  
            elif op.outputs[0].arguments[0] == "reshape2_21.tmp_0":
                shape = [1,232,26,26]              
            elif op.outputs[0].arguments[0] == "reshape2_23.tmp_0":
                shape = [1,232,26,26]              
            elif op.outputs[0].arguments[0] == "reshape2_25.tmp_0":
                shape = [1,464,13,13]              
            elif op.outputs[0].arguments[0] == "reshape2_27.tmp_0":
                shape = [1,464,13,13]  
            elif op.outputs[0].arguments[0] == "reshape2_29.tmp_0":
                shape = [1,464,13,13]              
            elif op.outputs[0].arguments[0] == "reshape2_31.tmp_0":
                shape = [1,464,13,13]              
            elif op.outputs[0].arguments[0] == "reshape2_32.tmp_0":
                shape = [1,32,2704]              
            elif op.outputs[0].arguments[0] == "reshape2_33.tmp_0":
                shape = [1,11,2704]              
            elif op.outputs[0].arguments[0] == "reshape2_34.tmp_0":
                shape = [1,32,676]              
            elif op.outputs[0].arguments[0] == "reshape2_35.tmp_0":
                shape = [1,11,676]  
            elif op.outputs[0].arguments[0] == "reshape2_36.tmp_0":
                shape = [1,32,169]              
            elif op.outputs[0].arguments[0] == "reshape2_37.tmp_0":
                shape = [1,11,169] 


            elif op.outputs[0].arguments[0] == "reshape2_0.tmp_0":
                shape = [2,58,52,52]
            elif op.outputs[0].arguments[0] == "reshape2_2.tmp_0":
                shape = [2,58,52,52]
            elif op.outputs[0].arguments[0] == "reshape2_4.tmp_0":
                shape = [2,58,52,52]
            elif op.outputs[0].arguments[0] == "reshape2_6.tmp_0":
                shape = [2,58,52,52]     
            elif op.outputs[0].arguments[0] == "reshape2_8.tmp_0":
                shape = [2,116,26,26]
            elif op.outputs[0].arguments[0] == "reshape2_10.tmp_0":
                shape = [2,116,26,26]
            elif op.outputs[0].arguments[0] == "reshape2_12.tmp_0":
                shape = [2,116,26,26]
            elif op.outputs[0].arguments[0] == "reshape2_14.tmp_0":
                shape = [2,116,26,26]
            elif op.outputs[0].arguments[0] == "reshape2_16.tmp_0":
                shape = [2,116,26,26]
            elif op.outputs[0].arguments[0] == "reshape2_18.tmp_0":
                shape = [2,116,26,26]
            elif op.outputs[0].arguments[0] == "reshape2_20.tmp_0":
                shape = [2,116,26,26]
            elif op.outputs[0].arguments[0] == "reshape2_22.tmp_0":
                shape = [2,116,26,26]
            elif op.outputs[0].arguments[0] == "reshape2_24.tmp_0":
                shape = [2,232,13,13]
            elif op.outputs[0].arguments[0] == "reshape2_26.tmp_0":
                shape = [2,232,13,13]
            elif op.outputs[0].arguments[0] == "reshape2_28.tmp_0":
                shape = [2,232,13,13]
            elif op.outputs[0].arguments[0] == "reshape2_30.tmp_0":
                shape = [2,232,13,13]
            else:
                continue
            for attr in op.attrs:
                if attr.name == "shape":
                    attr.ints.pop()
                    attr.ints.pop()
                    attr.ints.pop()
                    if len(attr.ints) > 0:
                        attr.ints.pop()
                    if len(attr.ints) > 0:
                        attr.ints.pop()
                    attr.ints.extend(shape)


    for var in vars:      
        if var.name == "reshape2_1.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([1,116,52,52])
        elif var.name == "reshape2_3.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([1,116,52,52])
        elif var.name == "reshape2_5.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([1,116,52,52])
        elif var.name == "reshape2_7.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([1,116,52,52])
        elif var.name == "reshape2_9.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([1,232,26,26])
        elif var.name == "reshape2_11.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([1,232,26,26])
        elif var.name == "reshape2_13.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([1,232,26,26])
        elif var.name == "reshape2_15.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([1,232,26,26])
        elif var.name == "reshape2_17.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([1,232,26,26])

        elif var.name == "reshape2_19.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([1,232,26,26])   

        elif var.name == "reshape2_21.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([1,232,26,26])

        elif var.name == "reshape2_23.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([1,232,26,26])

        elif var.name == "reshape2_25.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([1,464,13,13])

        elif var.name == "reshape2_27.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([1,464,13,13])

        elif var.name == "reshape2_29.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([1,464,13,13])

        elif var.name == "reshape2_31.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([1,464,13,13])

        elif var.name == "reshape2_32.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([1,32,2704])

        elif var.name == "reshape2_33.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([1,11,2704])

        elif var.name == "reshape2_34.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([1,32,676])

        elif var.name == "reshape2_35.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([1,11,676])

        elif var.name == "reshape2_36.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([1,32,169])
        elif var.name == "reshape2_37.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([1,11,169])

        elif var.name == "reshape2_0.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([2,58,52,52])
        elif var.name == "x2paddle_442.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([58,2,52,52])   
        elif var.name == "reshape2_2.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([2,58,52,52])
        elif var.name == "x2paddle_458.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([58,2,52,52])        
        elif var.name == "reshape2_4.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([2,58,52,52])
        elif var.name == "x2paddle_474.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([58,2,52,52])  
        elif var.name == "reshape2_6.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([2,58,52,52])
        elif var.name == "x2paddle_490.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([58,2,52,52]) 
        elif var.name == "reshape2_8.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([2,116,26,26])
        elif var.name == "x2paddle_509.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([116,2,26,26]) 
        elif var.name == "reshape2_10.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([2,116,26,26])
        elif var.name == "x2paddle_525.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([116,2,26,26]) 
        elif var.name == "reshape2_12.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([2,116,26,26])
        elif var.name == "x2paddle_541.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([116,2,26,26]) 
        elif var.name == "reshape2_14.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([2,116,26,26])
        elif var.name == "x2paddle_557.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([116,2,26,26]) 
        elif var.name == "reshape2_16.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([2,116,26,26])
        elif var.name == "x2paddle_573.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([116,2,26,26]) 
        elif var.name == "reshape2_18.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([2,116,26,26])
        elif var.name == "x2paddle_589.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([116,2,26,26]) 
        elif var.name == "reshape2_20.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([2,116,26,26])
        elif var.name == "x2paddle_605.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([116,2,26,26]) 
        elif var.name == "reshape2_22.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([2,116,26,26])
        elif var.name == "x2paddle_621.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([116,2,26,26]) 
        elif var.name == "reshape2_24.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([2,232,13,13])
        elif var.name == "x2paddle_640.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([232,2,13,13]) 
        elif var.name == "reshape2_26.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([2,232,13,13])
        elif var.name == "x2paddle_656.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([232,2,13,13]) 
        elif var.name == "reshape2_28.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([2,232,13,13])
        elif var.name == "x2paddle_672.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([232,2,13,13]) 
        elif var.name == "reshape2_30.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([2,232,13,13])
        elif var.name == "x2paddle_688.tmp_0":
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.pop()
            var.type.lod_tensor.tensor.dims.extend([232,2,13,13]) 
        else:
            if len(var.type.lod_tensor.tensor.dims) > 0 and var.type.lod_tensor.tensor.dims[0] == -1:
                var.type.lod_tensor.tensor.dims[0] = 1

modify_reshape2_transpose2()

content = prog.SerializeToString()
outfile = open("changed_model", "wb")
outfile.write(content)
outfile.close()
        
        


from tensorflow.python.tools.freeze_graph import freeze_graph
from convert_to_coreml import tfcoremllocal as tfcoreml
import coremltools.proto.FeatureTypes_pb2 as ft


def input_output_to_float32(spec):
    for feature in spec.description.output:
        update_multiarray_to_float32(feature)
    for feature in spec.description.input:
        update_multiarray_to_float32(feature)


def update_multiarray_to_float32(feature):
    if feature.type.HasField("multiArrayType"):
        feature.type.multiArrayType.dataType = ft.ArrayFeatureType.FLOAT32


def save_model(output_names, input_name_shape_dict, model_dir="./model", model_name="model",
               add_custom_layers=False, custom_conversion_functions=None):
    freeze_graph(input_graph=f"{model_dir}/{model_name}.pbtxt",
                 input_saver="",
                 input_binary=False,
                 input_checkpoint=f"{model_dir}/{model_name}.ckpt",
                 output_node_names=",".join(output_names),
                 output_graph=f"{model_dir}/{model_name}.pb",
                 clear_devices=True,
                 initializer_nodes="",
                 restore_op_name="save/restore_all",
                 filename_tensor_name="save/Const:0"
                 )

    return tfcoreml.convert(
        tf_model_path=f"{model_dir}/{model_name}.pb",
        mlmodel_path=f"{model_dir}/{model_name}.mlmodel",
        input_name_shape_dict=input_name_shape_dict,
        output_feature_names= [f"{output_name}:0" for output_name in output_names],
        add_custom_layers=add_custom_layers,
        custom_conversion_functions=custom_conversion_functions)


def get_nn(spec):
    if spec.WhichOneof("Type") == "neuralNetwork":
        return spec.neuralNetwork
    elif spec.WhichOneof("Type") == "neuralNetworkClassifier":
        return spec.neuralNetworkClassifier
    elif spec.WhichOneof("Type") == "neuralNetworkRegressor":
        return spec.neuralNetworkRegressor
    else:
        raise ValueError("MLModel does not have a neural network")


def rename_input(spec, input, new_name):
    old_name = input.name
    input.name = new_name
    nn = get_nn(spec)
    for i in range(len(nn.layers)):
        for k in range(len(nn.layers[i].input)):
            if nn.layers[i].input[k] == old_name:
                nn.layers[i].input[k] = new_name


def rename_output(spec, output, new_name):
    old_name = output.name
    output.name = new_name
    nn = get_nn(spec)
    for i in range(len(nn.layers)):
        for k in range(len(nn.layers[i].output)):
            if nn.layers[i].output[k] == old_name:
                nn.layers[i].output[k] = new_name



from hls4ml.converters.pytorch_to_hls import pytorch_handler

@pytorch_handler('LayerNorm')
def parse_layernorm_layer(operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config):
    assert 'LayerNorm' in operation
    layer = {}
    layer['embed_dim'] = input_shapes[0][-1]
    layer['seq_len'] = input_shapes[0][-2]
    layer['name'] = layer_name
    layer['inputs'] = input_names
    layer['scale_data'] = class_object.weight.data.numpy()
    layer['bias_data'] = class_object.bias.data.numpy()
    layer['class_name'] = 'LayerNorm'
    layer['data_format'] = 'channels_first'
    #only implemented for in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias
    #TODO: implement for other weights and biases

    output_shapes = input_shapes   
    return layer, output_shapes
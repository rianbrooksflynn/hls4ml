from hls4ml.converters.pytorch_to_hls import pytorch_handler


@pytorch_handler('HEPT')
def parse_hept_layer(
    operation, layer_name, input_names, input_shapes, node, class_object, data_reader, config
):
    assert 'HEPT' in operation
    assert len(input_shapes) == 2

    layer = {}

    layer['class_name'] = 'HEPT'
    layer['name'] = layer_name
    layer['inputs'] = input_names

    layer['n_blocks'] = class_object.n_blocks
    layer['block_size'] = class_object.block_size
    layer['dim_per_head'] = class_object.dim_per_head
    layer['coords_dim'] = class_object.coords_dim
    layer['query_shape'] = input_shapes[0]
    layer['key_shape'] = input_shapes[1]

    output_shape = (layer['n_blocks'], layer['block_size'], layer['block_size'])
    layer['output_shape'] = output_shape

    return layer, output_shape

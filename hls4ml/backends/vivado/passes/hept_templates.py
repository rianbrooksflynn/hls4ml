from hls4ml.backends.backend import get_backend
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate
from hls4ml.model.layers import HEPT
import numpy as np
from math import prod

# dense layer template
dense_config_template = """struct config{index}_dense_{label} : nnet::dense_config {{
    static const unsigned n_in = {n_in};
    static const unsigned n_out = {n_out};
    static const unsigned io_type = nnet::{iotype};
    static const unsigned strategy = nnet::{strategy};
    static const unsigned reuse_factor = {reuse};
    static const unsigned n_zeros = {nzeros};
    static const unsigned n_nonzeros = {nonzeros};
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef {accum_t.name} accum_t;
    typedef {bias_type} bias_t;
    typedef {weight_type} weight_t;
    typedef ap_{index_t} index_t;
    template<class data_T, class res_T, class CONFIG_T>
    using kernel = nnet::{dense_function}<data_T, res_T, CONFIG_T>;
    template<class x_T, class y_T>
    using product = nnet::product::{product_type}<x_T, y_T>;
}};\n"""

transpose_config_template = """struct {config_name} {{
    static const unsigned dims = {dims};
    static const unsigned N = {N};
    static const unsigned* const from_shape;
    static const unsigned* const to_shape;
    static const unsigned* const perm;
    static const unsigned* const perm_strides;
}};

unsigned {config_name}_from_shape[{dims}] = {{{from_shape}}};
unsigned {config_name}_to_shape[{dims}] = {{{to_shape}}};
unsigned {config_name}_perm[{dims}] = {{{perm}}};
unsigned {config_name}_perm_strides[{dims}] = {{{perm_strides}}};

const unsigned* const {config_name}::from_shape = {config_name}_from_shape;
const unsigned* const {config_name}::to_shape = {config_name}_to_shape;
const unsigned* const {config_name}::perm = {config_name}_perm;
const unsigned* const {config_name}::perm_strides = {config_name}_perm_strides;
"""

def transpose_config_gen(name: str, shape: tuple[int, ...], perm: tuple[int, ...]):
    new_shape = tuple(shape[i] for i in perm)
    strides = np.cumprod((shape[1:] + (1,))[::-1])[::-1]
    perm_strides = tuple(int(strides[i]) for i in perm)
    return transpose_config_template.format(
        dims=len(shape),
        N=prod(shape),
        from_shape=', '.join(str(x) for x in shape),
        perm=', '.join(str(x) for x in perm),
        perm_strides=', '.join(str(x) for x in perm_strides),
        to_shape=', '.join(str(x) for x in new_shape),
        config_name=name,
    )

hept_config_template = """struct config{index} : nnet::hept_config {{
    static const unsigned table_size = {table_size};
    static const int table_min = {table_min};
    static const int table_max = {table_max};

    typedef {accum_t.name} accum_t;
    typedef {table_t.name} table_t;
    typedef {config_dense_qk} dense_conf_qk;
    typedef {config_dense_qkv} dense_conf_qkv;
    typedef {config_transpose_qk} transpose_conf_qk;

    static const unsigned n_heads = {n_heads};
    static const unsigned n_blocks = {n_blocks};
    static const unsigned block_size = {block_size};
    static const unsigned dim_per_head = {dim_per_head};
    static const unsigned coords_dim = {coords_dim};

    static const unsigned io_type = nnet::{iotype};
    static const unsigned strategy = nnet::{strategy};
    static const unsigned reuse_factor = {reuse};
    static const unsigned parallelization_factor = {parallelization_factor};
    static const bool store_weights_in_bram = false;
}};\n"""

hept_function_template = """nnet::hept<{input_t}, {output_t}, {config}>({input_q}, {input_k}, {input_v}, {output});"""

hept_include_list = ['nnet_utils/nnet_hept.h']

class HeptConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(HEPT)
        self.template = hept_config_template
        self.dense_conf_qk_template = dense_config_template
        self.dense_conf_qkv_template = dense_config_template

    def format(self, node):
        params = self._default_config_params(node)
        params['n_heads'] = node.get_attr('n_heads')
        params['n_blocks'] = node.get_attr('n_blocks')
        params['block_size'] = node.get_attr('block_size')
        params['dim_per_head'] = node.get_attr('dim_per_head')
        params['coords_dim'] = node.get_attr('coords_dim')
        params['table_size'] = node.get_attr('table_size')
        params['table_min'] = node.get_attr('table_min')
        params['table_max'] = node.get_attr('table_max')
        params['config_dense_qk'] = f'config{node.index}_dense_qk'
        params['config_dense_qkv'] = f'config{node.index}_dense_qkv'
        params['config_transpose_qk'] = f'config{node.index}_transpose_qk'
        params['strategy'] = node.get_attr('strategy')
        params['parallelization_factor'] = node.get_attr('parallelization_factor')
        hept_config = self.template.format(**params)

        # dense qk config
        dense_qk_params = self._default_config_params(node)
        dense_qk_params['label'] = 'qk'
        dense_qk_params['strategy'] = 'latency'
        dense_qk_params['n_in'] = params['dim_per_head'] + params['coords_dim']
        dense_qk_params['n_out'] = params['block_size']
        dense_qk_params['product_type'] = get_backend('vivado').product_type(
            node.get_input_variable().type.precision, node.get_input_variable().type.precision
        )
        dense_qk_params['weight_type'] = node.get_input_variable().type.name
        dense_qk_params['bias_type'] = node.get_input_variable().type.name
        dense_qk_params['reuse'] = params['reuse']
        dense_qk_params['index'] = str(node.index)
        dense_qk_params['nzeros'] = 0
        dense_qk_params['nonzeros'] = params['block_size'] * (params['dim_per_head'] + params['coords_dim'])
        dense_qk_params['dense_function'] = 'DenseLatency'
        dense_qk_config = self.dense_conf_qk_template.format(**dense_qk_params)

        # dense qkv config
        dense_qkv_params = self._default_config_params(node)
        dense_qkv_params['label'] = 'qkv'
        dense_qkv_params['strategy'] = 'latency'
        dense_qkv_params['n_in'] = params['block_size']
        dense_qkv_params['n_out'] = params['dim_per_head']
        dense_qkv_params['product_type'] = get_backend('vivado').product_type(
            node.get_input_variable().type.precision, node.get_input_variable().type.precision
        )
        dense_qkv_params['weight_type'] = node.get_input_variable().type.name
        dense_qkv_params['bias_type'] = node.get_input_variable().type.name
        dense_qkv_params['reuse'] = params['reuse']
        dense_qkv_params['index'] = str(node.index)
        dense_qkv_params['nzeros'] = 0
        dense_qkv_params['nonzeros'] = params['block_size'] * params['dim_per_head']
        dense_qkv_params['dense_function'] = 'DenseLatency'
        dense_qkv_config = self.dense_conf_qkv_template.format(**dense_qkv_params)

        # transpose qk config
        trans_qk_shape = tuple(node.get_input_variable().shape)
        trans_qk_indices = (0, 1, 3, 2)
        trans_qk_config_name = f'config{node.index}_transpose_qk'
        trans_qk_config = transpose_config_gen(trans_qk_config_name, trans_qk_shape, trans_qk_indices)

        return "\n".join([dense_qk_config,
                          dense_qkv_config,
                          trans_qk_config,
                          hept_config])


class HeptFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(HEPT, include_header=hept_include_list)
        self.template = hept_function_template

    def format(self, node):
        params = {}
        params.update(node.attributes)
        params['config'] = f'config{node.index}'
        params['input_t'] = node.get_input_variable().type.name
        params['output_t'] = node.get_output_variable().type.name

        params['input_q'] = node.model.get_layer_output_variable(node.inputs[0]).name
        params['input_k'] = node.model.get_layer_output_variable(node.inputs[1]).name
        params['input_v'] = node.model.get_layer_output_variable(node.inputs[2]).name
        params['output'] = node.get_output_variable().name

        return self.template.format(**params)

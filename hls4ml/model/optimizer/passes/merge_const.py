import numpy as np

from hls4ml.model.layers import ApplyAlpha, Constant, Merge
from hls4ml.model.optimizer import OptimizerPass
from hls4ml.model.quantizers import QuantNodeQuantizer
from hls4ml.model.types import FixedPrecisionType, IntegerPrecisionType

_base_attributes = ('Trace', 'reuse_factor', 'n_in')


class MergeTwoConstants(OptimizerPass):
    """Merge of two constants makes another constant"""

    def match(self, node):
        is_match = (
            isinstance(node, Merge)
            and isinstance(node.get_input_node(node.inputs[0]), Constant)
            and isinstance(node.get_input_node(node.inputs[1]), Constant)
        )

        return is_match

    def transform(self, model, node):
        """
        Merge of two constants makes another constant.

        Note:  full precision is used in the calculation, and precision is not propagated.
        The precision
        """
        const_node0 = node.get_input_node(node.inputs[0])
        const_node1 = node.get_input_node(node.inputs[1])

        val0 = const_node0.attributes['value']
        val1 = const_node1.attributes['value']

        op = node.attributes['op']
        if op in ('add', 'sum'):
            new_val = val0 + val1
        elif op == 'sub':
            new_val = val0 - val1
        elif op == 'mul':
            new_val = val0 * val1
        elif op == 'div':
            new_val = val0 / val1
        elif op == 'average':
            new_val = np.mean(np.array([val0, val1]), axis=0)
        elif op == 'max':
            new_val = np.maximum(val0, val1)
        elif op == 'min':
            new_val = np.minimum(val0, val1)
        else:
            raise RuntimeError(f'Unexpected op_type: {op}')

        quantizer = node.get_attr('quantizer')  # None if not defined
        const_node0.set_attr('quantizer', quantizer)  # overwrite the quantizer
        if quantizer:
            const_node0.set_attr('quantizer', quantizer)

        const_node0.set_attr('value', new_val)

        # reinitialize (which also runs quantization if quantizer exists)
        const_node0.initialize()

        model.remove_node(const_node1, rewire=False)

        # remove the batch norm node
        model.remove_node(node, rewire=True)

        return True


class MergeToApplyAlpha(OptimizerPass):
    """Convert Add, Sub, Mul, or Div Merges with consant to ApplyAlpha"""

    def match(self, node):
        is_match = (
            isinstance(node, Merge)
            and node.attributes['op'] in ('add', 'sum', 'sub', 'mul')  # Div is separate
            and (
                isinstance(node.get_input_node(node.inputs[0]), Constant)
                != isinstance(node.get_input_node(node.inputs[1]), Constant)
            )
        )
        # note: != for booleans is xor.
        return is_match

    def transform(self, model, node):
        node1 = node.get_input_node(node.inputs[1])

        node1const = isinstance(node1, Constant)
        if node1const:
            const_node = node1
            input_node_idx = 0
        else:
            const_node = node.get_input_node(node.inputs[0])
            input_node_idx = 1

        input_shape = node.get_input_variable(node.inputs[input_node_idx]).shape
        n_in = np.prod(input_shape)

        # Note:  precision is ignored if quantizer is not None
        scale_precision = None
        scale_quantizer = None
        bias_precision = None
        bias_quantizer = None

        op = node.attributes['op']
        if op in ('add', 'sum'):
            scale = np.array(1)
            scale_precision = IntegerPrecisionType(1, False)
            bias = const_node.attribute['value']
            bias_quantizer = const_node.get_attr('quantizer')
        elif op == 'sub':
            bias_quantizer = const_node.get_attr('quantizer')
            if node1const:
                scale = np.array(1)
                scale_precision = IntegerPrecisionType(1, False)
                bias = -const_node.attribute['value']
                if (
                    bias_quantizer is not None
                    and isinstance(bias_quantizer.hls_type, (IntegerPrecisionType, FixedPrecisionType))
                    and not bias_quantizer.hls_type.signed
                ):
                    # need to make signed and increas the bit, if unsigned
                    bias_precision = FixedPrecisionType(
                        bias_quantizer.hls_type.width + 1,
                        bias_quantizer.hls_type.integer + 1,
                        True,
                        bias_quantizer.hls_type.rounding_mode,
                        bias_quantizer.hls_type.saturation_mode,
                        bias_quantizer.hls_type.saturation_bits,
                    )
                    bias_quantizer = QuantNodeQuantizer(bias_precision)
            else:
                scale = np.array(-1)
                scale_precision = IntegerPrecisionType(2, True)
                bias = const_node.attribute['value']

        elif op == 'mul':
            scale = const_node.attribute['value']
            scale_quantizer = const_node.get_attr('quantizer')
            bias = np.array(0)
            bias_precision = IntegerPrecisionType(1, False)

        # because C++ doesn't do broadcasting, we may have to change the shapes of the scale and bias
        if scale.shape != tuple(input_shape) and np.squeeze(scale).shape != tuple(input_shape):
            scale = np.broadcast_to(scale, input_shape)
        if bias.shape != tuple(input_shape) and np.squeeze(bias).shape != tuple(input_shape):
            bias = np.broadcast_to(bias, input_shape)

        attributes = {k: node.attributes.get(k, None) for k in _base_attributes}
        attributes.update(
            {
                'scale_data': scale,
                'bias_data': bias,
                'n_in': n_in,
                'n_out': n_in,
                'n_filt': -1,
                'scale_precision': scale_precision,
                'scale_quantizer': scale_quantizer,
                'bias_precision': bias_precision,
                'bias_quantizer': bias_quantizer,
            }
        )

        aa_layer = model.make_node(
            ApplyAlpha, f'bn_{node.name}', attributes, [node.inputs[input_node_idx]], [x for x in node.outputs]
        )

        model.remove_node(const_node, rewire=False)
        model.replace_node(node, aa_layer)

        return True


class MergeToApplyAlphaDiv(OptimizerPass):
    """
    Convert Div Merges with consant to ApplyAlpha

    TODO:  propagate precision
    """

    def match(self, node):
        is_match = (
            isinstance(node, Merge)
            and node.attributes['op'] == 'div'
            and isinstance(node.get_input_node(node.inputs[1]), Constant)
        )  # only second can be const

        return is_match

    def transform(self, model, node):
        input_shape = node.get_input_variable().shape
        n_in = np.prod(input_shape)
        const_node = node.get_input_node(node.inputs[1])
        scale = 1 / const_node.attribute['value']
        scale_quantizer = const_node.get_attr('quantizer')
        if scale_quantizer:
            scale_precision = scale_quantizer.hls_type
            i_new = 1 + int(scale_precision.signed) + scale_precision.fractional
            w_new = 1 + int(scale_precision.signed) + max(scale_precision.fractional, 0)
            new_scale_precision = FixedPrecisionType(
                w_new,
                i_new,
                scale_precision.signed,
                rounding_mode=scale_precision.rounding_mode,
                saturation_mode=scale_precision.saturation_mode,
                saturation_bits=scale_precision.saturation_bits,
            )
            scale_quantizer = QuantNodeQuantizer(new_scale_precision)

        bias = np.array(0)
        bias_precision = IntegerPrecisionType(1, False)

        # because C++ doesn't do broadcasting, we may have to change the shapes of the scale and bias
        if scale.shape != tuple(input_shape) and np.squeeze(scale).shape != tuple(input_shape):
            scale = np.broadcast_to(scale, input_shape)
        if bias.shape != tuple(input_shape) and np.squeeze(bias).shape != tuple(input_shape):
            bias = np.broadcast_to(bias, input_shape)

        attributes = {k: node.attributes.get(k, None) for k in _base_attributes}
        attributes.update(
            {
                'scale_data': scale,
                'bias_data': bias,
                'scale_quantizer': scale_quantizer,
                'bias_precision': bias_precision,
                'n_in': n_in,
                'n_out': n_in,
                'n_filt': -1,
            }
        )

        bn_layer = model.make_node(ApplyAlpha, f'bn_{node.name}', attributes, [node.inputs[0]], [x for x in node.outputs])

        model.remove_node(const_node, rewire=False)
        model.replace_node(node, bn_layer)

        return True
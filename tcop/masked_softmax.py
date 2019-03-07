import torch
import masked_softmax_cuda

# pylint: disable=arguments-differ


class MaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, mask, scale):
        scale = torch.tensor(scale)

        output = masked_softmax_cuda.forward(
            inputs.contiguous(), mask.contiguous(), scale)[0]
        ctx.save_for_backward(output, mask, scale)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        d_input = masked_softmax_cuda.backward(
            grad_output.contiguous(), *ctx.saved_tensors)[0]
        return d_input, None, None

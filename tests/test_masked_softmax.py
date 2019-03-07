import unittest

import torch

from tcop.masked_softmax import MaskedSoftmax


class TestMaskedSoftmax(unittest.TestCase):

    def test_forward(self):
        inputs = torch.tensor([[0.1, 0.2, 0.3],
                               [0.2, 0.3, 0.4],
                               [-0.3, -0.4, -0.5]])
        mask = torch.tensor([[0, 0, 0],
                             [0, 0, 0],
                             [0, 0, 0]], dtype=torch.float)

        inputs = inputs.view(1, 1, 3, 3).cuda()
        mask = mask.view(1, 3, 3).cuda()

        with torch.no_grad():
            expected = torch.nn.functional.softmax(inputs, dim=3)

        with torch.no_grad():
            mask = mask.size(2) - mask.sum(dim=2, dtype=torch.int32)
            output = MaskedSoftmax.apply(inputs, mask, 1.0)
            # torch.set_printoptions(precision=10)
            # print(output)
            # print(expected)
            self.assertTrue((output == expected).all())

    def _forward_test(self, inputs, mask, k, scale, debug=False):
        with torch.no_grad():
            x = inputs * scale
            if k is not None:
                x = torch.matmul(x, k)
            t_mask = mask.unsqueeze(1).byte()
            x = x + torch.zeros_like(x).masked_fill_(t_mask, -1e9)
            expected = torch.nn.functional.softmax(x, dim=3)

        with torch.no_grad():
            mask = mask.size(2) - mask.sum(dim=2, dtype=torch.int32)
            x = torch.matmul(inputs, k) if k is not None else inputs
            output = MaskedSoftmax.apply(x, mask, scale)
            if debug:
                torch.set_printoptions(precision=10)
                print("output", output)
                print("expected", expected)
                print("diff", torch.abs(output - expected))
            # We do not use equality here because MaskedSoftmax doesn't multiply
            # scale at first, so it has precesion differences.
            self.assertTrue((torch.abs(output - expected) < 1e-7).all())
            # self.assertTrue((output == expected).all())

    def test_forward_mask(self):
        inputs = torch.tensor([[0.1, 0.2, 0.3],
                               [-0.2, -0.3, -0.4],
                               [0.5, 0.4, 0.3]])
        mask = torch.tensor([[0, 1, 1],
                             [0, 0, 1],
                             [0, 0, 0]], dtype=torch.float)
        k = torch.tensor([[0.5, 2, 4],
                          [-0.2, -1, -0.4],
                          [0.5, 0.3, 0.3]])

        inputs = inputs.view(1, 1, 3, 3).cuda()
        mask = mask.view(1, 3, 3).cuda()
        k = k.view(1, 1, 3, 3).cuda()
        scale = 0.12

        self._forward_test(inputs, mask, k, scale)

    def test_forward_long(self):
        inputs = torch.tensor([1000] + list(range(69)), dtype=torch.float)
        mask = torch.tensor([0] * 70, dtype=torch.float)

        inputs = inputs.view(1, 1, 1, 70).cuda()
        mask = mask.view(1, 1, 70).cuda()
        scale = 0.1

        self._forward_test(inputs, mask, None, scale)

    def test_forward_last_all_mask(self):
        inputs = torch.tensor(list(range(64)), dtype=torch.float)
        mask = torch.tensor([0] * 32 + [1] * 32, dtype=torch.float)

        inputs = inputs.view(1, 1, 1, 64).cuda()
        mask = mask.view(1, 1, 64).cuda()
        scale = 0.1

        self._forward_test(inputs, mask, None, scale)

    def test_forward_multi_batch(self):
        inputs = torch.tensor([list(range(4)) * 4] * 12, dtype=torch.float)
        mask = torch.tensor([[0, 0, 1, 1]] * 4, dtype=torch.float)

        inputs = inputs.view(4, 3, 4, 4).cuda()
        mask = mask.view(4, 1, 4).cuda()
        scale = 0.1

        self._forward_test(inputs, mask, None, scale, debug=True)

    def test_forward_mini_seq(self):
        inputs = torch.tensor([list(range(64))] * 3, dtype=torch.float)
        mask = torch.tensor([[0] * 32 + [1] * 32] * 3, dtype=torch.float)

        inputs = inputs.view(1, 1, 3, 64).cuda()
        mask = mask.view(1, 3, 64).cuda()
        scale = 0.1

        self._forward_test(inputs, mask, None, scale)

    def test_forward_mini_batch_seq(self):
        inputs = torch.tensor([[list(range(64))] * 3] * 4, dtype=torch.float)
        mask = torch.tensor([[[0] * 32 + [1] * 32] * 3] * 4, dtype=torch.float)

        inputs = inputs.view(4, 1, 3, 64).cuda()
        mask = mask.view(4, 3, 64).cuda()
        scale = 0.1

        self._forward_test(inputs, mask, None, scale)

    def test_backward(self):
        inputs1 = torch.tensor([[0.3, 0.2, 0.1],
                                [0.2, 0.3, 0.4],
                                [0.3, 0.4, 0.5]], requires_grad=True)
        inputs2 = torch.tensor([[0.3, 0.2, 0.1],
                                [0.2, 0.3, 0.4],
                                [0.3, 0.4, 0.5]], requires_grad=True)
        mask = torch.tensor([[0, 1, 1],
                             [0, 0, 1],
                             [0, 0, 0]], dtype=torch.float)
        k = torch.tensor([[0.5, 2, 4],
                          [-0.2, -1, -0.4],
                          [0.5, 0.3, 0.3]], requires_grad=True)

        inputs1_cuda = inputs1.view(1, 1, 3, 3).cuda()
        inputs2_cuda = inputs2.view(1, 1, 3, 3).cuda()
        mask = mask.view(1, 3, 3).cuda()
        k = k.view(1, 1, 3, 3).cuda()
        scale = 0.1

        x = inputs1_cuda * scale
        x = torch.matmul(x, k)
        x = x + torch.zeros_like(x).masked_fill_(mask.byte(), -1e9)
        expected = torch.nn.functional.softmax(x, dim=3)
        loss = torch.mean(expected)
        loss.backward()

        x = torch.matmul(inputs2_cuda, k)
        mask = mask.size(2) - mask.sum(dim=2, dtype=torch.int32)
        output = MaskedSoftmax.apply(x, mask, scale)
        loss = torch.mean(output)
        loss.backward()

        # torch.set_printoptions(precision=10)
        # print(output)
        # print(expected)
        # print(inputs1.grad)
        # print(inputs2.grad)
        self.assertTrue((output == expected).all())
        self.assertTrue((torch.abs(inputs1.grad - inputs2.grad) < 1e-8).all())


if __name__ == '__main__':
    unittest.main()

from chainer.functions.activation import lstm
from chainer import link
from chainer.links.connection import linear
from chainer import variable
import numpy as np
import chainerUtil as C


class VLSTM(link.Chain):

    """Fully-connected LSTM layer.

    This is a fully-connected LSTM layer as a chain. Unlike the
    :func:`~chainer.functions.lstm` function, which is defined as a stateless
    activation function, this chain holds upward and lateral connections as
    child links.

    It also maintains *states*, including the cell state and the output
    at the previous time step. Therefore, it can be used as a *stateful LSTM*.

    Args:
        in_size (int): Dimensionality of input vectors.
        out_size (int): Dimensionality of output vectors.

    Attributes:
        upward (chainer.links.Linear): Linear layer of upward connections.
        lateral (chainer.links.Linear): Linear layer of lateral connections.
        c (chainer.Variable): Cell states of LSTM units.
        h (chainer.Variable): Output at the previous timestep.

    """
    def __init__(self, in_size, out_size):
        w = np.zeros((4 * out_size,in_size))
        super(LSTM, self).__init__(
            upward= C.Linear(in_size, 4 * out_size,initialW=w),
            lateral=C.Linear(out_size, 4 * out_size, nobias=True,initialW=w),
        )
        self.state_size = out_size
        self.reset_state()

    def reset_state(self):
        """Resets the internal state.

        It sets None to the :attr:`c` and :attr:`h` attributes.

        """
        self.c = self.h = None

    def __call__(self, x):
        """Updates the internal state and returns the LSTM outputs.

        Args:
            x (~chainer.Variable): A new batch from the input sequence.

        Returns:
            ~chainer.Variable: Outputs of updated LSTM units.

        """
        lstm_in = self.upward(x)
        if self.h is not None:
            lstm_in += self.lateral(self.h)
        if self.c is None:
            xp = self.xp
            self.c = variable.Variable(
                xp.zeros((len(x.data), self.state_size), dtype=x.data.dtype),
                volatile='auto')
        self.c, self.h = lstm.lstm(self.c, lstm_in)
        return self.h


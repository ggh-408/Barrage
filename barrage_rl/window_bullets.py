"""Array-backed window physics with an on-demand mutable row adapter."""
from collections.abc import Sequence
import numpy as np


class BulletRow(Sequence):
    def __init__(self, field, index):
        self.field, self.index = field, index

    def __len__(self):
        return 5

    def __getitem__(self, column):
        if isinstance(column, slice):
            return [self[i] for i in range(*column.indices(5))]
        column = range(5)[column]
        return (bool(self.field.targeted[self.index]) if column == 4
                else float(self.field.state[self.index, column]))

    def __setitem__(self, column, value):
        if isinstance(column, slice):
            indices = range(*column.indices(5))
            values = list(value)
            if len(indices) != len(values):
                raise ValueError('Bullet rows have exactly five fields')
            for i, v in zip(indices, values):
                self[i] = v
        elif range(5)[column] == 4:
            self.field.targeted[self.index] = value
        else:
            self.field.state[self.index, column] = value


class WindowBulletField(Sequence):
    def __init__(self):
        self.state = np.empty((0, 4), np.float32)
        self.targeted = np.empty(0, np.bool_)
        self._rows = []

    def __len__(self):
        return len(self.state)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = range(len(self))[index]
        if self._rows[index] is None:
            self._rows[index] = BulletRow(self, index)
        return self._rows[index]

    @property
    def positions(self):
        return self.state[:, :2]

    @property
    def velocities(self):
        return self.state[:, 2:4]

    def append_batch(self, positions, velocities, targeted):
        self.state = np.concatenate((self.state, np.column_stack((positions, velocities))))
        self.targeted = np.concatenate((self.targeted, targeted))
        self._rows.extend([None] * (len(self.state) - len(self._rows)))

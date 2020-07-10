import numpy as np

from astropy.io import fits


class AbstractVisibilities(np.ndarray):

    def __new__(cls, array, *args, **kwargs):

        obj = array.view(cls)
        obj.shape = array.shape

        return obj

    @property
    def real(self):
        return self[..., 0]

    @property
    def imag(self):
        return self[..., 1]

    @property
    def phases(self):
        return np.arctan2(
            self.imag,
            self.real
        )

    @property
    def amplitudes(self):
        return np.hypot(
            self.real,
            self.imag
        )


class Visibilities(AbstractVisibilities):

    def __init__(self, array):

        self.array = array

    @classmethod
    def manual(cls, array):

        if type(array) is list:
            array = np.asarray(array)

        return Visibilities(array=array)

    @classmethod
    def from_fits(cls, filename):

        array = fits.getdata(filename=filename)

        return cls.manual(array=array)

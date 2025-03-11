import abc


class MPIFileTypeError(Exception):
    pass


class MPIFileIncompatibleResolutionError(Exception):
    pass


class MPIFileView(metaclass=abc.ABCMeta):
    def __init__(self, fn, comm):
        self.fn = fn
        self.comm = comm
        # if hasattr read, it is a stream and it should not close the file
        self.close_file_on_error = not hasattr(fn, "read")

    @abc.abstractmethod
    def _read_header(self):
        pass

    @abc.abstractmethod
    def read(self):
        pass

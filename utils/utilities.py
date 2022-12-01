from typing import Any, List
from torch.utils.tensorboard import SummaryWriter

def singleton(cls):
    _instance = {}

    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]
    return inner

@singleton
class _Printer():

    def __init__(self) -> None:
        self._printer = print
        self._debug = False

    def print(self, debug: bool, *args: List[str]) -> None:
        if debug and not self._debug:
            return
        
        args = list(map(str, args))
        self._printer(' '.join(args))

    def setPrinter(self, **kwargs) -> None:
        if 'printer' in kwargs:
            self._printer = kwargs['printer']
        if 'debug' in kwargs:
            self._debug = kwargs['debug']

class Console(object):

    def __init__(self) -> None:
        pass

    @staticmethod
    def setPrinter(**kwargs) -> None:
        """ Set printer for Console

        Args:
            printer: Callable object, core output function
            debug: bool type, whether output debug information 
        """
        p = _Printer()
        p.setPrinter(**kwargs)
    
    @staticmethod
    def log(*args: List[Any]) -> None:
        """ Output log information

        Args:
            args: each element in the input list must be str type
        """
        p = _Printer()
        p.print(False, *args)
    
    @staticmethod
    def debug(*args: List[Any]) -> None:
        """ Output debug information

        Args:
            args: each element in the input list must be str type
        """
        p = _Printer()
        p.print(True, *args)

@singleton
class _Writer():
    def __init__(self) -> None:
        self.writer = None

    def write(self, write_dict: dict) -> None:
        if self.writer is None:
            raise Exception('[ERR-CFG] Writer is None!')
        
        for key in write_dict.keys():
            if write_dict[key]['plot']:
                self.writer.add_scalar(key, write_dict[key]['value'], write_dict[key]['step'])

    def setWriter(self, writer: SummaryWriter) -> None:
        self.writer = writer

class Ploter():
    def __init__(self) -> None:
        pass

    @staticmethod
    def setWriter(writer: SummaryWriter) -> None:
        w = _Writer()
        w.setWriter(writer)
    
    @staticmethod
    def write(write_dict: dict) -> None:
        w = _Writer()
        w.write(write_dict)
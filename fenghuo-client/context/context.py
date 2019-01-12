import sys
#sys.path.append('../')
from callbacks.callback import FengHuoCallBack

class Context:
    callbacks = []

    def __init__(self, callbacks=None):
        if not callbacks:
            return
        for callback in callbacks:
            self.register(callback)
        #self.callbacks.append(callbacks)

    def register(self, callback):
        if not isinstance(callback, FengHuoCallBack):
            raise ValueError("callback MUST be instance of FengHuoCallBack.")
        self.callbacks.append(callback)

    def unregister(self, callback):
        if not isinstance(callback, FengHuoCallBack):
            raise ValueError("callback MUST be instance of FengHuoCallBack.")
        self.callbacks.remove(callback)


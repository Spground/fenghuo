class FengHuoCallBack:
    def onHandIn(self, ctx):
        raise NotImplementedError("Not Implemented!")

    def onPredictTop1(self, ctx, prediction_top1):
        raise NotImplementedError("Not Implemented!")

    def onPredictTop5(self, ctx, prediction_top5):
        raise NotImplementedError("Not Implemented!")

    def onFrameSampled(self, ctx, frm):
        raise NotImplementedError("Not Implemented!")


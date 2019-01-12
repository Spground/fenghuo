class SimplePredictionProtocol:

    def __init__(self, pred_top5=None):
        self.pred = []
        if not pred_top5 == None:
            self.pred.extend(pred_top5)

    def top1(self):
        if len(self.pred) >= 1:
            return self.pred[0]
        return None

    def top5(self):
        if len(self.pred) == 5:
            return self.pred
        return None

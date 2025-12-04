from scipy.signal import butter, lfilter_zi, lfilter

class ButterworthLPF:
    def __init__(self, cutoff, fs, order=2):
        self.b, self.a = butter(order, cutoff / (0.5 * fs), btype='low')
        self.zi = lfilter_zi(self.b, self.a)

    def update(self, x):
        y, self.zi = lfilter(self.b, self.a, [x], zi=self.zi)
        return y[0]
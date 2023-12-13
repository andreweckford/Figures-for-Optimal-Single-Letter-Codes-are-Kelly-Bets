import time
from IPython.display import display,HTML

class ProgressBar:

    def __init__(self,maxCount,barWidth):
        self.t = time.time() # initialization time
        self.maxCount = maxCount # maximum number of iterations
        self.barWidth = barWidth # width of the bar to print
        self.count = 0
        self.remainChar = '.'
        self.doneChar = '='
        self.out = display("",display_id=True)
        self.out.update(HTML('<tt>|'+self.remainChar * self.barWidth+'|</tt>'))

    def iterate(self):
        self.count += 1
        bbb = int(self.barWidth * self.count / self.maxCount)
        lll = self.barWidth - bbb
        ir = ' ir: '+str(self.maxCount - self.count) # iterations remaining
        spi_val = (time.time()-self.t)/self.count
        spi = f' spi: {spi_val:.1f}' # seconds per iteration
        self.out.update(HTML('<tt>|' + self.doneChar * bbb + self.remainChar * lll + '|' + ir + spi + '</tt>'))

    def hide(self):
        self.out.update(HTML(''))

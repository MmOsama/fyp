import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

while True:
    try:
        ax = pickle.load(open('plot.pickle', 'rb'))
        plt.show()
    except:
        pass

from plotty_mcplotface import *
#import scienceplots, install for latex looking plots

def linear_example(x):
    y = 2*x + 30
    return y
# Generate synthetic data (example)
np.random.seed(0)
x = np.linspace(0,100,100)  # Input feature
noise = np.random.normal(0,1,100)
y = linear_example(x) + noise
data = [x,y]

# alternative, for more presentable plots, requires install of scienceplots
#with plt.style.context(['science', 'no-latex']):
 #plotty_mcplotface(data)

plotty_mcplotface(data)
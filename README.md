# # Plotty_McPlotface

Plotty_McPlotface is a Python-based tool designed to streamline the process of regression analysis for scientific papers. This software aims to automate and simplify the regression process, allowing researchers to focus on their analyses rather than the intricacies of coding.
## Usage

```python
from plotty_mcplotface import *

# Data without errors
x = [0,1,2,3,4,5]
y = [2,4,6,8,10,12]
data = [x,y]

# displays residual and regression fit
plotty_mcplotface(data)
```

```python
from plotty_mcplotface import *

# Data with errors
x = [0,1,2,3,4,5]
y = [2,4,6,8,10,12]
errors = [0.1,0.1,0.1,0.1,0.1,0.1]
data = [x,y,errors]

# displays residual and regression fit
plotty_mcplotface(data)
```
## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

License can be found in the license file.

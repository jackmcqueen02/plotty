# # Plotty_McPlotface

Plotty_McPlotface is a Python-based tool designed to streamline the process of regression analysis for scientific papers. This software aims to automate and simplify the regression process, allowing researchers to focus on their analyses rather than the intricacies of coding.

## Recommended Modules
scienceplots is recommended for improved figure presentation.
https://github.com/garrettj403/SciencePlots
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

## Examples
![carbon_emmision_latex](https://github.com/jackmcqueen02/plotty_mcplotface/assets/157049725/6b550af7-2917-43ab-b049-17d1435e2df1)
![uk annual temp change](https://github.com/jackmcqueen02/plotty_mcplotface/assets/157049725/0a4338fb-fc60-4eec-b3f1-fa247c748007)


## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

License can be found in the license file.

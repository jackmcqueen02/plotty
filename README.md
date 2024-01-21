# # plotty

plotty is a Python-based tool designed to streamline the process of regression analysis for scientific papers. This software aims to automate and simplify the regression process.

## Recommended Modules
scienceplots is recommended for improved figure presentation.
https://github.com/garrettj403/SciencePlots
## Usage

```python
from plotty import plotty_array

# Data without errors
x = [0,1,2,3,4,5]
y = [2,4,6,8,10,12]
data = [x,y]

# displays residual and regression fit
plotty_array(data)
```

```python
from plotty import plotty_array

# Data with errors
x = [0,1,2,3,4,5]
y = [2,4,6,8,10,12]
errors = [0.1,0.1,0.1,0.1,0.1,0.1]
data = [x,y,errors]

# displays residual and regression fit
plotty_array(data)
```
```python
from plotty import plotty_file

# displays residual and regression fit with data read from file
plotty_file('data.txt')
```
## Examples
![carbon_emmision_latex](https://github.com/jackmcqueen02/plotty_mcplotface/assets/157049725/6b550af7-2917-43ab-b049-17d1435e2df1)
![uk annual temp change](https://github.com/jackmcqueen02/plotty_mcplotface/assets/157049725/0a4338fb-fc60-4eec-b3f1-fa247c748007)
![image](https://github.com/jackmcqueen02/plotty/assets/157049725/07b722d9-abdd-4769-81cd-e6a03702b87a)




## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

License can be found in the license file.

## Badges
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

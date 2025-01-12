# Benchmark Results

## JuliaBUGS

| Model | Evaluation Time (μs) | Density Time (μs) | Density+Gradient Time (μs) | Parameter Count | Data Count |
|-------|---------------------|-------------------|---------------------------|-----------------|------------|
| rats | 62.375 | 53.917 | 233.583 | 65 | 150 |
| pumps | 8.944 | 8.542 | 14.500 | 12 | 10 |
| dogs | 193.730 | 141.041 | 314.417 | 2 | 720 |
| seeds | 19.542 | 19.792 | 45.541 | 26 | 21 |
| surgical_simple | 8.861 | 6.615 | 20.250 | 12 | 12 |
| surgical_realistic | 11.396 | 11.021 | 22.459 | 14 | 12 |
| magnesium | 110.292 | 94.459 | 179.208 | 108 | 96 |
| salm | 16.167 | 15.917 | 36.125 | 22 | 18 |
| equiv | 12.729 | 14.355 | 44.958 | 15 | 20 |
| dyes | 9.229 | 9.070 | 38.083 | 9 | 30 |
| stacks | 18.209 | 17.000 | 41.458 | 6 | 21 |
| epil | 177.041 | 197.875 | 511.916 | 303 | 236 |
| blockers | 41.000 | 41.417 | 75.000 | 47 | 44 |
| oxford | 248.584 | 230.750 | 438.875 | 244 | 240 |
| lsat | 1671.000 | 1302.000 | 2720.000 | 1006 | 5000 |
| bones | 314.541 | 292.354 | 326.938 | 33 | 422 |
| mice | 24.709 | 20.083 | 73.708 | 20 | 65 |
| kidney | 48.709 | 45.500 | 149.938 | 64 | 58 |
| leuk | 187.334 | 136.834 | 403.584 | 18 | 714 |
| leukfr | 208.458 | 155.375 | 535.916 | 40 | 714 |
| dugongs | 10.959 | 8.861 | 35.541 | 4 | 27 |
| orange_trees | 21.125 | 23.125 | 64.166 | 22 | 35 |
| orange_trees_multivariate | 32.584 | 30.875 | 158.500 | 8 | 35 |
| air | 4.095 | 4.361 | 5.525 | 5 | 3 |
| jaws | 36.917 | 34.750 | 602.875 | 3 | 20 |
| birats | 93.605 | 80.333 | 455.459 | 33 | 150 |
| schools | 1349.000 | 1136.000 | 5347.000 | 50 | 1978 |
| beetles | 5.633 | 4.903 | 8.542 | 2 | 8 |
| alligators | 25.750 | 26.125 | 51.958 | 28 | 40 |
| endo | 106.083 | 85.625 | 150.562 | 1 | 183 |
| stagnant | 12.459 | 11.646 | 35.292 | 5 | 29 |
| asia | 3.401 | missing | missing | 5 | 1 |
| biopsies | 94.542 | missing | missing | 161 | 157 |
| eyes | 39.500 | missing | missing | 50 | 50 |
| hearts | 15.958 | missing | missing | 14 | 12 |
| cervix | 2227.000 | missing | missing | 1936 | 4203 |

## Stan

| Model | Dimension | Evaluation Time (μs) | Density Time (μs) | Density+Gradient Time (μs) |
|-------|-----------|---------------------|-------------------|---------------------------|
| rats | 65 | missing | 4.965 | 7.281 |
| pumps | 12 | missing | 0.836 | 1.277 |
| dogs | missing | missing | missing | missing |
| seeds | 26 | missing | 1.958 | 2.617 |
| surgical_realistic | 14 | missing | 1.153 | 1.586 |
| magnesium | 108 | missing | 10.584 | 11.896 |
| salm | 22 | missing | 3.443 | 4.077 |
| equiv | 15 | missing | 3.306 | 4.107 |
| dyes | 9 | missing | 1.546 | 2.012 |
| stacks | 6 | missing | 1.153 | 1.763 |
| epil | 303 | missing | 38.667 | 42.750 |
| blockers | 47 | missing | 2.316 | 2.846 |
| oxford | 244 | missing | 10.521 | 12.750 |
| lsat | 1006 | missing | 94.958 | 132.708 |
| bones | 13 | missing | 71.708 | 81.584 |
| mice | 20 | missing | 6.396 | 8.083 |
| kidney | 46 | missing | 8.514 | 11.438 |
| leuk | 18 | missing | 16.959 | 21.542 |
| leukfr | 40 | missing | 21.791 | 27.792 |
| dugongs | 4 | missing | 1.226 | 1.886 |
| air | 5 | missing | 0.801 | 1.233 |
| birats | 66 | missing | 24.542 | 28.334 |
| schools | 133 | missing | 223.312 | 334.709 |
| beetles | 2 | missing | 0.805 | 1.250 |
| alligators | 28 | missing | 4.347 | 5.108 |

## Nimble

| Model | Non-compiled (ms) | Compiled (μs) | NLL (μs) | Gradient (μs) |
|-------|------------------|---------------|-----------|---------------|
| rats | 26.104 | 6.560 | 9.430 | 21.525 |
| pumps | 2.240 | 4.715 | 5.289 | 8.692 |
| dogs | NA | NA | NA | NA |
| seeds | 5.126 | 4.674 | 6.724 | 10.742 |
| surgical | 2.812 | 4.510 | 5.330 | 8.446 |
| magnesium | 23.905 | 6.437 | 12.956 | 23.698 |
| salm | 4.394 | 4.387 | 6.560 | 10.414 |
| equiv | NA | NA | NA | NA |
| dyes | 3.176 | 4.305 | 4.879 | 8.385 |
| stacks | 11.705 | 6.027 | 5.084 | 11.111 |
| epil | 84.904 | 13.284 | 35.957 | 66.502 |
| blocker | 9.307 | 6.724 | 8.487 | 15.191 |
| oxford | 57.984 | 18.942 | 26.814 | 53.013 |
| lsat | 903.323 | 79.561 | 189.051 | 490.524 |
| bones | 189.076 | 41.123 | NA | NA |
| mice | 22.787 | 8.979 | 8.733 | NA |
| kidney | 26.327 | 7.011 | 10.496 | NA |
| leuk | 0.734 | 4.182 | 4.387 | 6.396 |
| leukfr | 2.305 | 4.346 | 5.371 | 8.200 |
| dugongs | 4.265 | 4.346 | 5.002 | 10.045 |
| orange | 8.086 | 5.371 | 5.945 | 12.546 |
| mvotree | 7.131 | 7.257 | 6.478 | 14.063 |
| biopsies | NA | NA | NA | NA |
| eyes | 12.000 | 87.371 | NA | NA |
| hearts | NA | NA | NA | NA |
| air | 0.961 | 4.059 | 4.633 | 6.724 |
| cervix | 969.712 | 18294.790 | NA | NA |
| jaw | NA | NA | NA | NA |
| birats | 24.279 | 10.681 | 8.733 | 22.960 |
| schools | 939.536 | 79.581 | 72.283 | 270.662 |
| ice | NA | NA | NA | NA |
| beetles | 1.861 | 4.264 | 5.002 | 7.544 |
| alli | 11.169 | 6.929 | 8.487 | 14.412 |
| endo | NA | NA | NA | NA |
| stagnant | 7.097 | 4.387 | 5.043 | 12.669 |
| asia | NA | NA | NA | NA |

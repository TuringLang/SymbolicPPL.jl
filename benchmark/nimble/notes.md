# Model Issues

## Unsupported Features

- Dynamic indexing of constants:
  - `asia`: In `p.lung.cancer[smoking, 1:2]`
  - `equiv`: In `sign[T[i, k]]`

- Cyclic dependencies:
  - `hearts`
  - `ice`

## Other Issues

- `magnesium`: Requires manual variable transformation
- `bones`: Contains discrete parameters
- `kidney`: Contains non-differentiable components
- `mice`: Using JAGS version due to censoring syntax
- `biopsies`: Contains keywords and dynamic indexing loops

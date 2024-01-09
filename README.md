# Eidolon
*eidolon*: in ancient Greek mythology, an eidolon was the spirit-image of a living or deceased person.

Here, tensor networks aim to capture the spirit of language-like data: the Motzkin spin chains.
This code was written for my PhD thesis project under the supervision of Dr. David Schwab and Dr. John Terilla.


## Running the learning loops
1. Create a conda environment using `conda [environment name] create -f environment.yml`.
Replace "*[environment name]*" with the desired environment name. Then, activate the environment with `conda activate [environment name]`.
2. N.b. `environment.yml` uses the cpu version of Jax, and `motzkin_gpu.yml` uses the gpu version.
3. Configure the various parameters as desired in the `conf/` directory.
4. From the `src/` directory, run `python main.py name=[run name]`, where "*[run_name]*" is the name the logger will save the logfiles to (output directory printed by the code).
They should end up in an "experiments" directory; if an error arises, one might need to make that directory first.

## Running tests
From the root directory, run `pytest tests`.
To see printouts of the errors calculated in each test, run `pytest -s tests`.

The tests are run on sequences of length 8.
Note that for each test, a scaled error is calculated.
When comparing the outputs of two calculation methods, call them `x` and `y`, the scaled error is `2*|x+y|/(x+y)`; that is, the distance between outputs divided by their average value.
The threshold to pass is that the scaled error should be belore 1e-6.

The tests are:
1. `test_contract_einsum`: compare the output of contraction of the dense model with a single input sequence to that calculated using jax numpy einsum.
2. `test_john_contractr`': compare the output of contraction of the dense model with a single input sequence to that calculated using John's function.
3. `test_tests`: compare the outputs of the einsum and John's methods from the previous tests as a sanity check.
4. `test_einsum norm`: compare the norm calculation of the dense model to one calculated using jax numpy einsum.
5. `test_john_norm`: compare the norm calculation of the dense model to one calculated using John's method.
6. `test_dense_sum_contractions_norm`: compare the norm calculation of the dense model to the sum of all unnormalized probabilities of the dense model.
7. `test_factored_sum_contractions_norm`: compare the norm calculation of the factored model to the sum of all unnormalized probabilities of the factored model.
8. `test_grad_batching`: test the parameter gradient output (for the dense model) from a pmap'ed batch to one calculated by adding the gradients from each element of the batch, then dividing by the total number of elements.

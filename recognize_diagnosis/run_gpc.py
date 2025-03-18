"""
    Giza Pyramids Construction (GPC) Algorithm

    Author : Sasan Harifi

    Paper  : Giza Pyramids Construction: an ancient-inspired metaheuristic algorithm for optimization
    DOI    : http://dx.doi.org/10.1007/s12065-020-00451-3

    Copyright (c) 2020, All rights reserved.
    Please read the "license.txt" for license terms.

    Code Publisher: http://www.harifi.com
 -------------------------------------------------
    This demo only implements a standard version of GPC for minimization of
    a standard test function (Sphere) on Python 3.
 -------------------------------------------------
    Note:
    Due to the stochastic nature of metaheuristc algorithms, different runs
    may lead to slightly different results.
 -------------------------------------------------
"""
# External Libraries
import numpy as np
import matplotlib.pyplot as plt

# From This Project
import ypstruct
import gpc
import benchmarks

# Problem Definition
problem = ypstruct.structure()
problem.objfunc = benchmarks.sphere  # See benchmarks module for other functions
problem.nvar = 30
problem.varmin = -5.12
problem.varmax = 5.12

# Parameters of Giza Pyramids Construction (GPC)
params = ypstruct.structure()
params.maxit = 100                   # Maximum Number of Iterations (Days of work)
params.npop = 20                     # Number of workers
params.G = 9.8                       # Gravity
params.Tetha = 14                    # Angle of Ramp
params.MuMin = 1                     # Minimum Friction 
params.MuMax = 10                    # Maximum Friction
params.pSS = 0.5                     # Substitution Probability
params.DisplayInfo = True

# Run GPC
out = gpc.run(problem, params)

# Print Final Result
print("Final Best Solution: {0}".format(out.best_worker))

# Plot of Best Costs History
plt.semilogy(out.bestcosts)
plt.xlim(0, params.maxit)
plt.xlabel("Iterations")
plt.ylabel("Best Cost")
plt.title("Giza Pyramids Construction")
plt.grid(True)
plt.show()

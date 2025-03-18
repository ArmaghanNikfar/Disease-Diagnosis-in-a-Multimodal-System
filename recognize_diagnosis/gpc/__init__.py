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
import numpy as np
import ypstruct

# Run Giza Pyramids Construction (GPC)
def run(problem, params):
    
    # Problem Definition
    objfunc = problem.objfunc          # Cost Function
    nvar = problem.nvar                # Number of Decision Variables
    varmin = problem.varmin            # Decision Variables Lower Bound
    varmax = problem.varmax            # Decision Variables Upper Bound

    # Params
    maxit = params.maxit               # Maximum Number of Iterations (Days of work)
    npop = params.npop                 # Number of workers
    G = params.G                       # Gravity
    Tetha = params.Tetha               # Angle of Ramp
    MuMin = params.MuMin               # Minimum Friction 
    MuMax = params.MuMax               # Maximum Friction
    pSS = params.pSS                   # Substitution Probability
    DisplayInfo = params.DisplayInfo

    # Empty Stones or Workers (Individual)
    empty_stones = ypstruct.structure()
    empty_stones.position = None
    empty_stones.cost = None

    # Best Solution
    best_worker = empty_stones.deepcopy()
    best_worker.cost = np.inf

    # Best Costs History
    bestcosts = np.empty(maxit)

    # Initialization
    pop = empty_stones.repeat(npop)
    for i in range(0, npop):
        pop[i].position = np.random.uniform(varmin, varmax, nvar)
        pop[i].cost = objfunc(pop[i].position)
        if pop[i].cost <= best_worker.cost:
            best_worker = pop[i].deepcopy()        # as Pharaoh's special agent

    # Construction Main Loop
    for it in range(0, maxit):
        for i in range(0, npop):
            
            V0 = np.random.rand(1)                                                             # Initial Velocity  
            temp_rand = np.random.rand(1)                                                      # Temp Random Number
            Mu= MuMin+(MuMax-MuMin)*temp_rand[0]                                               # Friction
            d = (V0[0]**2)/((2*G)*(np.sin(np.deg2rad(Tetha))+(Mu*np.cos(np.deg2rad(Tetha)))))  # Stone Destination  
            x = (V0[0]**2)/((2*G)*(np.sin(np.deg2rad(Tetha))))                                 # Worker Movement
            epsilon = np.random.uniform(-((varmax-varmin)/2),((varmax-varmin)/2), nvar)        # Epsilon
            new_position = apply_bounds((pop[i].position + d) * (x * epsilon), varmin, varmax) # Position of Stone and Worker
          # new_position = apply_bounds((pop[i].position + d) + (x * epsilon), varmin, varmax) # In some obj func use this to get better results

            # Substitution
            newsol = empty_stones.deepcopy()
            newsol.position = substitution(pop[i].position, new_position, pSS)
            newsol.cost = objfunc(newsol.position)

            if newsol.cost <= pop[i].cost:
                pop[i] = newsol
                if pop[i].cost <= best_worker.cost:
                    best_worker = pop[i].deepcopy()

        # Store Best Cost of Iteration
        bestcosts[it] = best_worker.cost

        # Show Iteration Info
        if DisplayInfo:
            print("Iteration {0}: Best Cost = {1}".format(it, best_worker.cost))
    
    # Return Results
    out = ypstruct.structure()
    out.best_worker = best_worker
    out.bestcost = best_worker.cost
    out.bestcosts = bestcosts
    out.pop = pop
    return out

# Apply Decision Variable Ranges
def apply_bounds(x, varmin, varmax):
    x = np.maximum(x, varmin)
    x = np.minimum(x, varmax)
    return x

# Substitution
def substitution(x, new_position, pSS):
    z = np.copy(x)
    nvar = x.size
    k = np.where(np.random.rand(nvar) <= pSS)
    k = np.append(k, np.random.randint(0, nvar))
    z[k] = new_position[k]
    return z
    
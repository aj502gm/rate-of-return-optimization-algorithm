'''
-------------- DOCUMENTATION --------------
    Goal: Optimize portfolio investment risk
    Developers:
        * Jorge Anibal Velasquez 
        * Andrés de Jesús Gonzalez Melgar
    Packages used:
        * cvxpy: https://www.cvxpy.org/
        * Numpy: https://numpy.org/
        * Pandas: https://pandas.pydata.org/
'''

import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import math as math
from tabulate import tabulate

def calculate_efficient_frontier(data, nombre_archivo, rend_min, rend_max):
    # IGNORE THE YEARS AND JUST TAKE INTO CONSIDERATION THE FINANCIAL ASSETS VALUES
    financial_assets_values = data.iloc[:, 1:]
    rates_of_change = financial_assets_values.pct_change().dropna()
    # STOCKING VECTOR
    stocking_vector = rates_of_change.mean().values
    # COVARIANCE MATRIX
    covariance = rates_of_change.cov().values

    n = len(stocking_vector)
    x = cp.Variable(n) 

    # VALIDATE FOR THE OPTIMAL DISTRIBUTION INSIDE THE RANGE
    expected_return_of_rates = np.linspace(rend_min, rend_max, 50)

    risks = []
    optimal_investment_distributions = []
    optimal_portfolios = []

    # AVANZAR ENTRE EL RENDIMIENTO MÍNIMO Y MÁXIMO
    for R in expected_return_of_rates:
        risk = cp.quad_form(x, covariance)
        constraints = [
            # RATE OF RETURN MUST BE GREATER THAN THE ACTUAL R
            stocking_vector @ x >= R,
            # THE INVESTMENT MUST BE DISTRIBUTED IN ITS ENTIRETY
            cp.sum(x) == 1,
            # ALL INVESTMENT DISTRIBUTION MUST BE EQUAL OR GREATER THAN 0
            x >= 0
        ]
        
        # SOLVE THE PROBLEM
        problem = cp.Problem(cp.Minimize(risk), constraints)
        problem.solve()

        riskAppend = np.sqrt(problem.value)
        isInfinite = math.isinf(np.sqrt(problem.value))

        # IF THE RISK IS INFINITE, IT MEANS THAT THERE ARE NO MORE FEASIBLE POINTS
        # THEREFORE, AVOID FURTHER ITERATING UNNECESSARYLY
        if(isInfinite == True):
            break
        else:
            # RISK
            risks.append(riskAppend)
            optimal_investment_distributions.append(R)
            # SUGGESTED INVESTMENT DISTRIBUTION
            optimal_portfolios.append(x.value)


    tabla_resultados = pd.DataFrame({
        "Expected Rate of Return (R)": expected_return_of_rates[:len(risks)],
        "Risk (Standar Deviation)": risks,
        "Optimal Portfolio": optimal_portfolios
    })

    plt.figure(figsize=(10, 6))
    plt.plot(risks, optimal_investment_distributions, marker='o', linestyle='-', label=nombre_archivo)
    plt.title(f"Efficient Frontier ({nombre_archivo}). Algorithm: {problem.solver_stats.solver_name}")
    plt.xlabel("Risk (Standar Deviation)")
    plt.ylabel("Expected Rate of Return (R)")
    plt.legend()
    plt.grid()
    plt.show()

    return tabla_resultados

# DATOS
file1 = 'Data1.csv'
file2 = 'Data2.csv'

data1 = pd.read_csv(file1)
data2 = pd.read_csv(file2)

print("----------------- Rate of Return Range ------------------------")
print("Example: 6.5% -> 0.065, 10.5% -> 0.105")

rend_min = float(input("Mininum Rate of Return: "))
rend_max = float(input("Maxinum Rate of Return: "))

# CALCULATE FOR BOTH THE HISTORICAL DATA
results1 = calculate_efficient_frontier(data1, "Data1.csv", rend_min, rend_max)
results2 = calculate_efficient_frontier(data2, "Data2.csv", rend_min, rend_max)

# SHOW TABULATED DATA
print("\nData1.csv results:")
print(tabulate(results1.head(50), headers='keys', tablefmt='fancy_grid'))

print("\nData2.csv results:")
print(tabulate(results2.head(50), headers='keys', tablefmt='fancy_grid'))

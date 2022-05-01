import numpy as np
import pandas as pd
from scipy.optimize import linprog
import seaborn as sns
import matplotlib.pyplot as plt

"""
Problem Statement and Information
"""

# Define data set
my_dict = {'Calories': [295, 1216, 394, 358, 128, 118, 279],
           'Iron': [0.2, 0.2, 4.3, 3.2, 3.2, 14.1, 2.2],
           'Fat': [16, 96, 9, 0.5, 0.8, 1.4, 0.5],
           'Protein': [16, 81, 74, 83, 7, 14, 8],
           'Carbohydrates': [22, 0, 0, 0, 28, 19, 63],
           'Cost': [0.6, 2.35, 1.15, 2.25, 0.58, 1.17, 0.33]}

# Food names
names = ['Milk',
         'Grd Meat',
         'Chicken',
         'Fish',
         'Beans',
         'Spinach',
         'Potatoes']

df = pd.DataFrame(my_dict, index=names)

# Function to define linear programming problem and solve
def f(inputs, df, method):

    max_cal, min_cal, min_iro, max_fat, min_pro, max_crb = inputs

    # Cost function
    c = df['Cost'].to_numpy()
    
    # Linear constraint matrix
    A = np.vstack(
                   (
                     df['Calories'].to_numpy(),
                    -df['Calories'].to_numpy(),
                    -df['Iron'].to_numpy(),
                     df['Fat'].to_numpy(),
                    -df['Protein'].to_numpy(),
                     df['Carbohydrates'].to_numpy()
                   )
                  )
    
    b = [ max_cal,
         -min_cal,
         -min_iro,
          max_fat,
         -min_pro,
          max_crb]
    
    # Solve
    res = linprog(c, A_ub=A, b_ub=b, method=method)
    
    return res

"""

Part A: Optimal Solution Under Given Constraints

"""

# Linear constraint bounds from problem statement
max_cal = 1500
min_cal = 900
min_iro = 4
max_fat = 50
min_pro = 26
max_crb = 50

inputs = [max_cal, min_cal, min_iro, max_fat, min_pro, max_crb]

# Call function
res = f(inputs, df, 'highs')

# Determine solution's value of constraint variables
values = np.matmul(np.transpose(df.to_numpy()),res.x)

# Define dataframes for solution and constraint variables
df_con = pd.DataFrame([values], columns=df.columns)
df_sol = pd.DataFrame([res.x], columns=names)

# Write to File
df_con.to_csv('Constraints_Results_Part_A.csv', index=False)
df_sol.to_csv('Solution_Part_A.csv', index=False)

# Plot result
plt.figure(1)
sns.set_theme(style='whitegrid')
sns.barplot(data=df_sol)
plt.ylabel('Optimal Quantity (lb)')
plt.savefig('Part_A_Optimal_Solution.tiff', dpi=300)

"""

Part B1: Sensitivity to Fat

"""

fat = np.linspace(50,60,11)

cost = np.zeros_like(fat)

for i in range(0,len(fat)):
    inputs = [max_cal, min_cal, min_iro, fat[i], min_pro, max_crb]
    res = f(inputs, df, 'highs')
    cost[i] = res.fun

plt.figure(2)
plt.plot(fat,cost,'-b')
plt.xlabel('Maximum Fat (mg)')
plt.ylabel('Optimized Cost (USD/lb)')
plt.xlim([fat.min(), fat.max()])
plt.savefig('Part_B1_Sensitivity_to_Fat.tiff', dpi=300)

# 54 appears to be the value at which the marginal benefit becomes zero
inputs_orig = [max_cal, min_cal, min_iro, max_fat, min_pro, max_crb]
inputs_fopt = [max_cal, min_cal, min_iro, 54, min_pro, max_crb]

# Call function
res_orig = f(inputs_orig, df, 'highs')
res_fopt = f(inputs_fopt, df, 'highs')

# Store in dataframe
df_orig = pd.DataFrame([res_orig.x], columns=names, index=['Reference'])
df_fopt = pd.DataFrame([res_fopt.x], columns=names, index=['Increased Fat'])

df_plot = pd.concat([df_orig, df_fopt], axis=0)
df_plot = df_plot.reset_index().melt(id_vars='index')

df_plot = df_plot.rename(columns={'index': 'Solution', 'variable': 'Ingredient', 'value': 'Optimal Quantity (lb)'})

plt.figure(3)
sns.barplot(data=df_plot, x='Ingredient', y='Optimal Quantity (lb)', hue='Solution')
plt.savefig('Part_B1_Fat_Optimized_Solution.tiff', dpi=300)

"""

Part B2: Sensitivity to Protein

"""

protein = np.linspace(0,65,66)

cost = np.zeros_like(protein)

for i in range(0,len(protein)):
    inputs = [max_cal, min_cal, min_iro, max_fat, protein[i], max_crb]
    res = f(inputs, df, 'highs')
    cost[i] = res.fun

plt.figure(4)
plt.plot(protein,cost,'-b')
plt.xlabel('Minimum Protein (mg)')
plt.ylabel('Optimized Cost (USD/lb)')
plt.xlim([protein.min(), protein.max()])
plt.savefig('Part_B2_Sensitivity_to_Protein.tiff', dpi=300)

"""

Part B3: Sensitivity to Iron

"""

iron = np.linspace(0,4,9)

cost = np.zeros_like(iron)

for i in range(0,len(iron)):
    inputs = [max_cal, min_cal, iron[i], max_fat, min_pro, max_crb]
    res = f(inputs, df, 'highs')
    cost[i] = res.fun

plt.figure(5)
plt.plot(iron,cost,'-b')
plt.xlabel('Minimum Iron (mg)')
plt.ylabel('Optimized Cost (USD/lb)')
plt.xlim([iron.min(), iron.max()])
plt.savefig('Part_B3_Sensitivity_to_Iron.tiff', dpi=300)

# 2.5 appears to be the value at which the marginal benefit becomes zero
inputs_orig = [max_cal, min_cal, min_iro, max_fat, min_pro, max_crb]
inputs_fopt = [max_cal, min_cal, 2.5, max_fat, min_pro, max_crb]

# Call function
res_orig = f(inputs_orig, df, 'highs')
res_iopt = f(inputs_fopt, df, 'highs')

# Store in dataframe
df_orig = pd.DataFrame([res_orig.x], columns=names, index=['Reference'])
df_iopt = pd.DataFrame([res_iopt.x], columns=names, index=['Decreased Iron'])

df_plot = pd.concat([df_orig, df_iopt], axis=0)
df_plot = df_plot.reset_index().melt(id_vars='index')

df_plot = df_plot.rename(columns={'index': 'Solution', 'variable': 'Ingredient', 'value': 'Optimal Quantity (lb)'})

plt.figure(6)
sns.barplot(data=df_plot, x='Ingredient', y='Optimal Quantity (lb)', hue='Solution')
plt.savefig('Part_B3_Iron_Optimized_Solution.tiff', dpi=300)


"""

Part C: Potato Price Sensitivity

"""

potato_cost = np.linspace(0.33, 1, 51)

cost = np.zeros_like(potato_cost)

df_temp = df.copy(deep=True)

for i in range(0,len(potato_cost)):
    inputs = [max_cal, min_cal, min_iro, max_fat, min_pro, max_crb]
    df_temp['Cost']['Potatoes'] = potato_cost[i]
    res = f(inputs, df_temp, 'highs')
    cost[i] = res.fun

plt.figure(7)
plt.plot(potato_cost, cost,'-b')
plt.xlabel('Potato Cost (USD/lb)')
plt.ylabel('Optimized Cost (USD/lb)')
plt.xlim([potato_cost.min(), potato_cost.max()])
plt.savefig('Part_C_Sensitivity_to_Potato_Prices.tiff', dpi=300)

# Call function
res_orig = f(inputs_orig, df, 'highs')
res_popt = f(inputs_orig, df_temp, 'highs')

# Store in dataframe
df_orig = pd.DataFrame([res_orig.x], columns=names, index=['Reference'])
df_popt = pd.DataFrame([res_popt.x], columns=names, index=['Contingency'])

df_plot = pd.concat([df_orig, df_popt], axis=0)
df_plot = df_plot.reset_index().melt(id_vars='index')

df_plot = df_plot.rename(columns={'index': 'Solution', 'variable': 'Ingredient', 'value': 'Optimal Quantity (lb)'})

plt.figure(8)
sns.barplot(data=df_plot, x='Ingredient', y='Optimal Quantity (lb)', hue='Solution')
plt.savefig('Part_C_Potato_Contingency_Solution.tiff', dpi=300)


"""

Plot Figures

"""

plt.show()



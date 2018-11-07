
import sympy

'''
Vs,iVs,R1,R2,Is,Vo, V1, V2 = sympy.symbols('Vs,iVs,R1,R2,Is,Vo,V1,V2')

# Create an empty list of equations
equations = []

# Nodal equations
equations.append(iVs-(V1-V2)/R1)       # Node 1
equations.append(Is-(V2-V1)/R1-V2/R2)  # Node 2

equations.append(sympy.Eq(Vs,V1))
equations.append(sympy.Eq(Vo,V2))

unknowns = [V1,V2,iVs,Vo] 
solution = sympy.solve(equations,unknowns) 
print (solution)
'''

######################################################

V1, V2, V3, V4, V5, V6 = sympy.symbols('V1, V2, V3, V4, V5, V6')
R1, R2, R3, R4, R5, R6 = sympy.symbols('R1, R2, R3, R4, R5, R6')
I = sympy.symbols('I')
Vx, Vy = sympy.symbols('Vx, Vy')
Rx = sympy.symbols('Rx')

unknowns = [I, Vx, Vy]
eq1 = ((V1 - Vx) / R1) + ((V2 - Vx) / R2) + ((V3 - Vx) / R3) - I
eq2 = ((V4 - Vy) / R4) + ((V5 - Vy) / R5) + ((V6 - Vy) / R6) - I
eq3 = ((Vx - Vy) / Rx - I)
solution = sympy.solve([eq1, eq2, eq3], unknowns) 
# print (solution)

subs = {V1:1., V2:1., V3:1., V4:1., V5:1., V6:1., \
        R1:1., R2:1., R3:1., R4:1., R5:1., R6:1., \
        Rx:1.
        }

print (solution[Vx].evalf(subs=subs))
print (solution[Vy].evalf(subs=subs))
print (solution[I].evalf(subs=subs))

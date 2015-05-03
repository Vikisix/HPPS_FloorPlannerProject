"""
@author Marco Rabozzi [marco.rabozzi@mail.polimi.it]
"""

with open('../fpga/XC5VLX110T.json') as f:
	fpga = eval(f.read())
with open('problem.json') as f:
	problem = eval(f.read())


import floorplanner
print floorplanner.solve(problem, fpga, False, None)

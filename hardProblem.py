# the following is a sample problem
# it shows how to call the floorplan directly

from gurobipy import *

problem = {}
problem['fpga_file'] =  '/home/marco/Dropbox/FCCM_Florplacer/2015/test/XC5VLX110T.json'
problem['regions'] = ['rec1','rec2','rec3','rec4','rec5','rec6','rec7','rec8','rec9','rec10','rec11','rec12','rec13','rec14','rec15','rec16','rec17','rec18','rec19','rec20','rec21','rec22','rec23','rec24','rec25']
problem['regRes'] = {
	('rec1','CLB') : 1520,
	('rec1','BRAM') : 4,
	('rec1','DSP') : 8,

	('rec2','CLB') : 620,
	('rec2','BRAM') : 0,
	('rec2','DSP') : 8,

	('rec3','CLB') : 500,
	('rec3','BRAM') : 16,
	('rec3','DSP') : 0,

	('rec4','CLB') : 500,
	('rec4','BRAM') : 4,
	('rec4','DSP') : 0,

	('rec5','CLB') : 920,
	('rec5','BRAM') : 4,
	('rec5','DSP') : 0,

	('rec6','CLB') : 320,
	('rec6','BRAM') : 0,
	('rec6','DSP') : 8,

	('rec7','CLB') : 500,
	('rec7','BRAM') : 16,
	('rec7','DSP') : 0,

	('rec8','CLB') : 200,
	('rec8','BRAM') : 4,
	('rec8','DSP') : 0,

	('rec9','CLB') : 92,
	('rec9','BRAM') : 0,
	('rec9','DSP') : 0,

	('rec10','CLB') : 120,
	('rec10','BRAM') : 4,
	('rec10','DSP') : 0,

	('rec11','CLB') : 500,
	('rec11','BRAM') : 0,
	('rec11','DSP') : 0,

	('rec12','CLB') : 400,
	('rec12','BRAM') : 0,
	('rec12','DSP') : 0,

	('rec13','CLB') : 92,
	('rec13','BRAM') : 0,
	('rec13','DSP') : 0,

	('rec14','CLB') : 120,
	('rec14','BRAM') : 4,
	('rec14','DSP') : 0,

	('rec15','CLB') : 200,
	('rec15','BRAM') : 0,
	('rec15','DSP') : 0,

	('rec16','CLB') : 100,
	('rec16','BRAM') : 0,
	('rec16','DSP') : 0,

	('rec17','CLB') : 200,
	('rec17','BRAM') : 0,
	('rec17','DSP') : 0,

	('rec18','CLB') : 200,
	('rec18','BRAM') : 0,
	('rec18','DSP') : 0,

	('rec19','CLB') : 200,
	('rec19','BRAM') : 0,
	('rec19','DSP') : 0,

	('rec20','CLB') : 200,
	('rec20','BRAM') : 0,
	('rec20','DSP') : 0,

	('rec21','CLB') : 200,
	('rec21','BRAM') : 0,
	('rec21','DSP') : 0,

	('rec22','CLB') : 100,
	('rec22','BRAM') : 0,
	('rec22','DSP') : 0,	

	('rec23','CLB') : 100,
	('rec23','BRAM') : 0,
	('rec23','DSP') : 0,	

	('rec24','CLB') : 100,
	('rec24','BRAM') : 0,
	('rec24','DSP') : 0,	

	('rec25','CLB') : 100,
	('rec25','BRAM') : 0,
	('rec25','DSP') : 0,	
}
problem['obj_weights'] = {
	'WL' : 1,
	'P' : 0,
	'R' : 0
}
problem['res_cost'] = {
	'CLB' : 1,
	'BRAM' : 1,
	'DSP' : 1
}
problem['communications'] = {
	('rec1','rec2') : 10,
	('rec4','rec5') : 5,
	('rec1','rec5') : 30,
	('rec7','rec2') : 5,
	('rec3','rec4') : 1,
	('rec6','rec3') : 2,
	('rec9','rec2') : 10,
	('rec10','rec2') : 10,
	('rec11','rec2') : 10,
	('rec13','rec2') : 10,
}

import floorplan2
floorplan2.solve(problem, False)
"""
@author Marco Rabozzi [marco.rabozzi@mail.polimi.it]
"""

from gurobipy import *
import traceback
import sys
import hashlib
import time

#whether to generates computing intensive WL cuts
USE_2REG_WL_CUT = False


"""
This method solves the floorplanning problem using a MILP
model based on the conflict graph of the intersection
relationship between regions

@param problem The problem data (see exampleProblem.py for an example)
@param fpgaData The fpga data (see the json files in fpga folder)
@param relocation Wheter to optimize the problem using the relocation metrics or not
@param fixedRegions a pre-defined solution for the problem

@return a dictionary containing the solution data
"""
def solve(problem, fpgaData, relocation, fixedRegions):

    result = {'status': False}
    startTime = time.time()
    #-----------------------------------------------------------
    # parameters setup
    #-----------------------------------------------------------

    #parse the fpga data
    fpga = {
        'maxX' : fpgaData['maxTX'],
        'maxY' : fpgaData['maxTY'],
        'tileW' : fpgaData['tileW'],
        'tileH' : fpgaData['tileH']
    }
    fpga['resources'] = [res for res in fpgaData['resources']]
    resPerType = {res:fpgaData['resources'][res]['number'] for res in fpgaData['resources']}

    fpga['resources'] += ['NULL','-F-']
    resPerType['NULL'] = 0
    resPerType['-F-'] = 1
    problem['res_cost']['NULL'] = 0
    problem['res_cost']['-F-'] = 0

    fpga['portions'] = range(1,len(fpgaData['portions']) + 1)
    fpga['por_x1'] = {por:fpgaData['portions'][por-1]['x1'] for por in fpga['portions']}
    fpga['por_y1'] = {por:fpgaData['portions'][por-1]['y1'] for por in fpga['portions']}
    fpga['por_x2'] = {por:fpgaData['portions'][por-1]['x2'] for por in fpga['portions']}
    fpga['por_y2'] = {por:fpgaData['portions'][por-1]['y2'] for por in fpga['portions']}
    fpga['porRes'] = {(por,fpgaData['portions'][por-1]['type']) : resPerType[fpgaData['portions'][por-1]['type']] for por in fpga['portions']}


    #parse the problem data
    problem['obj_weights']['WL'] = problem['obj_weights']['wirelength']
    problem['obj_weights']['P'] = problem['obj_weights']['perimeter']
    problem['obj_weights']['R'] = problem['obj_weights']['resources']
    
    problem['regions'] = [r for r in problem['regions_data']]
    problem['regRes'] = {}

    for regName in problem['regions_data']:
        region = problem['regions_data'][regName]
        for resName in region['resources']:
            res = region['resources'][resName]
            problem['regRes'][(regName,resName)] = res

    problem['io'] = {}
    for regName in problem['regions_data']:
        region = problem['regions_data'][regName]
        problem['io'][regName] = []
        for io in region['io']:
            problem['io'][regName].append((io['tileX'],io['tileY'],io['wires']))            


    communications = {}
    for comm in problem['communications']:
        communications[(comm['from'], comm['to'])] = comm['wires']

    problem['communications'] = communications



    #generate the fpga resource matrix
    matrix = {}

    #-->List to identify the type of homogeneous resource in x,y (tile hight)
    tileType = {}

    for p in fpga['portions']:
        for x in xrange(fpga['por_x1'][p], fpga['por_x2'][p] + 1):
            for y in xrange(fpga['por_y1'][p], fpga['por_y2'][p] + 1):
                for t in fpga['resources']:
                    if (p,t) in fpga['porRes']:
                        matrix[x,y,t] = fpga['porRes'][p,t]
                        #--> part added
                        tileType[x,y] = t
                    else:
                        matrix[x,y,t] = 0


    #-->HPPS_PROJECT ADDED PART: Recalculate the fpga resource matrix, to introduce a new bound for the clocl generator
    #-->line. We considere the type 'NULL' as the clock signal generators
    for x in xrange(0,fpga['maxX']-2): #-2 because from an heuristic we discover that the next two CLB on the left 
                                        #of a clock line cannot be used for a dynamic PR
        for y in xrange(0,fpga['maxY']-2):
            for t in fpga['resources']:
                print (matrix[x,y,t])
                if(tileType[x+1,y+1]=='NULL' or tileType[x+2,y+2]=='NULL'):
                    matrix[x,y,t] = 0


    #computes the resources on each column
    colRes = {}

    try:
        for x in xrange(0,fpga['maxX']):
            for t in fpga['resources']:
                colRes[x,0,t] = matrix[x,0,t]
                for y in xrange(1,fpga['maxY']):
                    colRes[x,y,t] = colRes[x,y-1,t] + matrix[x,y,t]
    except:
        print "The set of portions is not a partitioning of the fpga!";
        traceback.print_exc()
        return


    #computes the resources from the bottom-left corner to a specific coordinate
    blRes = {}

    for t in fpga['resources']:
        for y in xrange(0,fpga['maxY']):
            blRes[0,y,t] = colRes[0,y,t]
            for x in xrange(1,fpga['maxX']):
                blRes[x,y,t] = blRes[x-1,y,t] + colRes[x,y,t]

    #boundary conditions to simplify resource computations:
    for t in fpga['resources']:
        blRes[-1,-1,t] = 0
        for y in xrange(0,fpga['maxY']):
            blRes[-1,y,t] = 0
        for x in xrange(0,fpga['maxX']):
            blRes[x,-1,t] = 0


    #function for the computation of resources within an area
    #example: areaRes((x1,y1,x2,y2),'CLB')
    areaRes = lambda area, t: blRes[area[2],area[3],t] + blRes[area[0]-1,area[1]-1,t] - blRes[area[2],area[1]-1,t] - blRes[area[0]-1,area[3],t]

    #computes the upper bounds on the objective function metrics
    if not relocation:
        # max resource waste
        Rmax = 0
        for t in fpga['resources']:
            Rmax += areaRes((0,0,fpga['maxX']-1,fpga['maxY']-1), t)*problem['res_cost'][t]
            for r in problem['regions']:
                if (r,t) in problem['regRes']:
                    Rmax -= problem['regRes'][(r,t)]*problem['res_cost'][t]

        # max perimeter
        Pmax = len(problem['regions'])*(fpga['maxY']*fpga['tileH'] + fpga['maxX']*fpga['tileW'])*2

        # max wirelength
        WLmax = sum([problem['communications'][con] for con in problem['communications']])
        for r in problem['io']:
            WLmax += sum(io[2] for io in problem['io'][r])

        WLmax = WLmax*(fpga['tileW']*fpga['maxX'] + fpga['tileH']*fpga['maxY'])
        if(WLmax == 0):
            WLmax = 1

        

    
    #function for the computation of the cluster id of an area
    #example: computeCluster((x1,y1,x2,y2))
    def computeCluster(area):
        cluster = ''
        for t in fpga['resources']:
            cluster += '@' + t
            for y in xrange(area[1], area[3] + 1):
                count = 1
                cluster += '|' + str(matrix[area[0],y,t])
                for x in xrange(area[0] + 1, area[2] + 1):
                    if matrix[x,y,t] == matrix[x-1,y,t]:
                        count = count + 1
                    else:
                        cluster += '*' + str(count) + ',' + str(matrix[x,y,t])
                        count = 1
                cluster += '*' + str(count)

        return hashlib.sha224(cluster).hexdigest()


    def isFeasibile(r,area):
        wastedRes = {}
        for t in fpga['resources']:
            wastedRes[t] = areaRes(area,t)
            if(t == '-F-' and wastedRes[t] > 0):
                return False

            if (r,t) in problem['regRes']:
                wastedRes[t] -= problem['regRes'][r,t]
                if wastedRes[t] < 0:
                    return False
        return True

    # select the area placement generation strategy
    isDominated = None

    placementStrategy = problem['placement_generation_mode']
    if placementStrategy == 'width-reduced':
        isDominated = lambda r,x1,y1,x2,y2: isFeasibile(r,(x1,y1+1,x2,y2))
    elif placementStrategy == 'irreducible':
        isDominated = lambda r,x1,y1,x2,y2: isFeasibile(r,(x1+1,y1,x2,y2)) or isFeasibile(r,(x1,y1+1,x2,y2)) or isFeasibile(r,(x1,y1,x2,y2-1)) or isFeasibile(r,(x1,y1,x2-1,y2))
    elif placementStrategy == 'all':
        isDominated = lambda r,x1,y1,x2,y2: False


    #-----------------------------------------------------------
    # variables and constraints generation
    #-----------------------------------------------------------

    areas = tuplelist()
    clusters = tuplelist()
    numAreasForRegion = {}
    areaSize = {}
    areasXSquare = {}
    areasXCluster = {}
    areasIOCost = {}
    areasRCost = {}
    areasPCost = {}
    areasCXCost = {}
    areasCYCost = {}
    areasMinCCost = {}
    minWLComm = {}
    minWLComm2 = {}
    minWLCommX = {}
    minWLCommX2 = {}
    minWLCommY = {}
    minWLCommY2 = {}


    if fixedRegions == None:
        fixedRegions = {}

    for x in xrange(0, fpga['maxX']):
        for y in xrange(0, fpga['maxY']):
            areasXSquare[x,y] = tuplelist()


    for r in problem['regions']:
        numAreasForRegion[r] = 0


        if r in fixedRegions: # if requested create only the variable for the fixed region area
            x1Min = fixedRegions[r]['x1']
            x1Max = fixedRegions[r]['x1'] + 1
            y1Min = fixedRegions[r]['y1']
            y1Max = fixedRegions[r]['y1'] + 1
            x2Min = lambda x1: fixedRegions[r]['x2']
            x2Max = fixedRegions[r]['x2'] + 1
            y2Min = lambda y1: fixedRegions[r]['y2']
            y2Max = lambda y1: fixedRegions[r]['y2'] + 1
        else:
            x1Min = 0
            x1Max = fpga['maxX']
            y1Min = 0
            y1Max = fpga['maxY']
            x2Min = lambda x1: x1
            x2Max = fpga['maxX']
            y2Min = lambda y1: y1
            y2Max = lambda y1: fpga['maxY']

        # IMPORTANT: this version of the algorithm generates irreducible rectangles 
        for x1 in xrange(x1Min, x1Max):
            for y1 in xrange(y1Min, y1Max):
                for y2 in xrange(y2Min(y1), y2Max(y1)):
                    for x2 in xrange(x2Min(x1), x2Max):

                        #verify that the resource requirements are met
                        satisfied = True
                        forbidden = False
                        wastedRes = {};

                        for t in fpga['resources']:
                            wastedRes[t] = areaRes((x1,y1,x2,y2),t)
                            if(t == '-F-' and wastedRes[t] > 0):
                                forbidden = True;
                                break;

                            if (r,t) in problem['regRes']:
                                wastedRes[t] -= problem['regRes'][r,t]
                                if wastedRes[t] < 0:
                                    satisfied = False;
                                    break;

                        if forbidden:
                            break

                        if satisfied:
                            
                            # verify that the area is not dominated
                            if isDominated(r,x1,y1,x2,y2):
                                break
                            

                            # insert the area as a possible solution for the region
                            numAreasForRegion[r] = numAreasForRegion[r] + 1
                            a = (r,x1,y1,x2,y2)
                            areas += [a]
                            areaSize[a] = (x2-x1+1)*(y2-y1+1)

                            # update areasXSquare dict
                            for x in xrange(x1, x2 + 1):
                                for y in xrange(y1, y2 + 1):
                                    areasXSquare[x,y] += [a]


                            if relocation:
                                # insert new cluster
                                c = computeCluster((x1,y1,x2,y2))
                                if (r,c) not in clusters:
                                    clusters += [(r,c)]
                                    areasXCluster[(r,c)] = []

                                # update areasXCluster
                                areasXCluster[(r,c)] += [a]
                            else:

                                #computes area cost
                                cost = 0
                                for t in fpga['resources']:
                                    cost = cost + wastedRes[t]*problem['res_cost'][t]
                                areasRCost[a] = cost

                                #computes perimeter cost
                                areasPCost[a] = (x2-x1+1)*fpga['tileW'] + (y2-y1+1)*fpga['tileH']

                                #computes wirelength cost
                                areasCXCost[a] = fpga['tileW']*(x1 + x2 + 1)/2.0
                                areasCYCost[a] = fpga['tileH']*(y1 + y2 + 1)/2.0
                                areasMinCCost[a] = min((x2-x1+1)*fpga['tileW']/2.0, (y2-y1+1)*fpga['tileH']/2.0)

                                #computes io cost
                                areasIOCost[a] = 0
                                if(r in problem['io']):
                                    for io in problem['io'][r]:
                                        areasIOCost[a] += (abs(areasCXCost[a] - io[0]*fpga['tileW']) + abs(areasCYCost[a] - io[1]*fpga['tileH']))*io[2]
                            break   



    # verifies that each region has at least a feasible placement
    for r in problem['regions']:
        if numAreasForRegion[r] == 0:
            # no valid solution can exists
            return result


    def overlapping(a1,a2):
        wtot = a1[3] + a2[3] - a1[1] - a2[1]
        htot = a1[4] + a2[4] - a1[2] - a2[2]
        wcur = max(a1[3],a2[3]) - min(a1[1],a2[1])
        hcur = max(a1[4],a2[4]) - min(a1[2],a2[2])
        return wcur <= wtot and hcur <= htot


    if not relocation and USE_2REG_WL_CUT:

        for c in problem['communications']:
            r1 = c[0]
            r2 = c[1]
            print str(r1) + " " + str(r2)   
            minWLComm[c] = {}
            minWLCommX[c] = {}
            minWLCommY[c] = {}
            for a1 in areas.select(r1,'*','*','*','*'):
                minWLComm[c][a1] = fpga['tileW']*fpga['maxX'] + fpga['tileH']*fpga['maxY']
                minWLCommX[c][a1] = fpga['tileW']*fpga['maxX']
                minWLCommY[c][a1] = fpga['tileH']*fpga['maxY']
                for a2 in areas.select(r2,'*','*','*','*'):
                    if not overlapping(a1,a2):
                        minWLComm[c][a1] = min(minWLComm[c][a1], abs(areasCXCost[a1] - areasCXCost[a2]) + abs(areasCYCost[a1] - areasCYCost[a2]))
                        minWLCommX[c][a1] = min(minWLCommX[c][a1], abs(areasCXCost[a1] - areasCXCost[a2]))
                        minWLCommY[c][a1] = min(minWLCommX[c][a1], abs(areasCYCost[a1] - areasCYCost[a2]))

            r1 = c[1]
            r2 = c[0]
            print str(r1) + " " + str(r2)   
            minWLComm2[c] = {}
            minWLCommX2[c] = {}
            minWLCommY2[c] = {}
            for a1 in areas.select(r1,'*','*','*','*'):
                minWLComm2[c][a1] = fpga['tileW']*fpga['maxX'] + fpga['tileH']*fpga['maxY']
                minWLCommX2[c][a1] = fpga['tileW']*fpga['maxX']
                minWLCommY2[c][a1] = fpga['tileH']*fpga['maxY']
                for a2 in areas.select(r2,'*','*','*','*'):
                    if not overlapping(a1,a2):
                        minWLComm2[c][a1] = min(minWLComm2[c][a1], abs(areasCXCost[a1] - areasCXCost[a2]) + abs(areasCYCost[a1] - areasCYCost[a2]))
                        minWLCommX2[c][a1] = min(minWLCommX2[c][a1], abs(areasCXCost[a1] - areasCXCost[a2]))
                        minWLCommY2[c][a1] = min(minWLCommX2[c][a1], abs(areasCYCost[a1] - areasCYCost[a2]))



    print 'Areas: ' + str(len(areas))
    axql = 0
    for x in xrange(0, fpga['maxX']):
        for y in xrange(0, fpga['maxY']):
            axql += len(areasXSquare[x,y])
    print 'Non overlapping terms: ' + str(axql)
    print 'Clusters: ' + str(len(areasXCluster))


    # define variables
    areaVars = {}
    centroidXVars = {}
    centroidYVars = {}
    commXVars = {}
    commYVars = {}
    
    clusterVars = {}
    m = Model("floorplan")

    if relocation:
        for a in areas:
            areaVars[a] = m.addVar(0.0,1.0,areaSize[a],GRB.BINARY, str(a))
        for (r,c) in clusters:
            clusterVars[r,c] = m.addVar(0.0,1.0,0.0,GRB.BINARY, str((r,c)))
    else:
        for a in areas:
            areaVars[a] = m.addVar(0.0,1.0,0.0,GRB.BINARY, str(a))

        for r in problem['regions']:
            centroidXVars[r] = m.addVar(0.0,GRB.INFINITY,0.0,GRB.CONTINUOUS, 'centroid_x_' + str(r))
            centroidYVars[r] = m.addVar(0.0,GRB.INFINITY,0.0,GRB.CONTINUOUS, 'centroid_y_' + str(r))

        for c in problem['communications']:
            commXVars[c] = m.addVar(0.0,GRB.INFINITY,0.0,GRB.CONTINUOUS, 'comm_x_' + str(c))
            commYVars[c] = m.addVar(0.0,GRB.INFINITY,0.0,GRB.CONTINUOUS, 'comm_y_' + str(c))

        CCost = m.addVar(0.0, GRB.INFINITY, float(problem['obj_weights']['WL'])/WLmax, GRB.CONTINUOUS, 'CCost')
        IOCost = m.addVar(0.0, GRB.INFINITY, float(problem['obj_weights']['WL'])/WLmax, GRB.CONTINUOUS, 'IOCost')
        RCost = m.addVar(0.0, GRB.INFINITY, float(problem['obj_weights']['R'])/Rmax, GRB.CONTINUOUS, 'RCost')
        PCost = m.addVar(0.0, GRB.INFINITY, float(problem['obj_weights']['P'])/Pmax, GRB.CONTINUOUS, 'PCost')

    m.update()


    # define constraints    


    # no overlapping between regions
    for y in xrange(0, fpga['maxY']):
        for x in xrange(0, fpga['maxX']):
            if(len(areasXSquare[x,y]) > 0):
                m.addConstr(quicksum(areaVars[a] for a in areasXSquare[x,y]) <= 1, 'noOverlapping_' + str(x) + '_' + str(y))


    if relocation:

        # at least an area for a region
        for r in problem['regions']:
            m.addConstr(quicksum(areaVars[a] for a in areas.select(r,'*','*','*','*')) >= 1, 'atLeastOne_'+str(r))

        # no more than one cluster per region
        for r in problem['regions']:
            m.addConstr(quicksum(clusterVars[r,c] for (r,c) in clusters.select(r,'*')) <= 1, 'atMostOneCluster_' + str(r))

        # area - cluster association
        for (r,c) in clusters:
            for a in areasXCluster[r,c]:
                m.addConstr(clusterVars[r,c] >= areaVars[a], 'areaClusterAssoc_' + str(a) + '_' + str(c))

        m.modelSense = GRB.MAXIMIZE

    else:
        
        # one area for region
        for r in problem['regions']:
            m.addConstr(quicksum(areaVars[a] for a in areas.select(r,'*','*','*','*')) == 1, 'oneAndOnlyOne_'+str(r))

        # wasted resources computation
        m.addConstr(quicksum(areaVars[a]*areasRCost[a] for a in areas) == RCost, 'RCost_def')

        # perimeter computation
        m.addConstr(quicksum(areaVars[a]*areasPCost[a] for a in areas) == PCost, 'PCost_def')

        # centroid computation
        for r in problem['regions']:
            m.addConstr(quicksum(areaVars[a]*areasCXCost[a] for a in areas.select(r,'*','*','*','*')) == centroidXVars[r], 'centroid_x_def_'+str(r))
            m.addConstr(quicksum(areaVars[a]*areasCYCost[a] for a in areas.select(r,'*','*','*','*')) == centroidYVars[r], 'centroid_y_def_'+str(r))

        # communication bound
        for c in problem['communications']:
            m.addConstr(centroidXVars[c[0]] - centroidXVars[c[1]]  <= commXVars[c], 'comm_x_bound_1_' + str(c))
            m.addConstr(centroidXVars[c[1]] - centroidXVars[c[0]]  <= commXVars[c], 'comm_x_bound_2_' + str(c))
            m.addConstr(centroidYVars[c[0]] - centroidYVars[c[1]]  <= commYVars[c], 'comm_y_bound_1_' + str(c))
            m.addConstr(centroidYVars[c[1]] - centroidYVars[c[0]]  <= commYVars[c], 'comm_y_bound_2_' + str(c))

        # communication cuts
        for c in problem['communications']:
            
            
            m.addConstr(commXVars[c] + commYVars[c] >=
                quicksum(areaVars[a]*areasMinCCost[a] for a in areas.select(c[1],'*','*','*','*')) +
                quicksum(areaVars[a]*areasMinCCost[a] for a in areas.select(c[0],'*','*','*','*')), 'comm_region_cut_' + str(c))

            if USE_2REG_WL_CUT:


                m.addConstr(commXVars[c] + commYVars[c] >=
                    quicksum(areaVars[a]*minWLComm[c][a] for a in minWLComm[c]), 'comm_pair_cut_' + str(c))
                m.addConstr(commXVars[c] + commYVars[c] >=
                    quicksum(areaVars[a]*minWLComm2[c][a] for a in minWLComm2[c]), 'comm_pair_cut_2_' + str(c))

                m.addConstr(commXVars[c] >=
                    quicksum(areaVars[a]*minWLCommX[c][a] for a in minWLCommX[c]), 'comm_pair_cut_x_' + str(c))
                m.addConstr(commXVars[c] >=
                    quicksum(areaVars[a]*minWLCommX2[c][a] for a in minWLCommX2[c]), 'comm_pair_cut_x_2_' + str(c))

                m.addConstr(commYVars[c] >=
                    quicksum(areaVars[a]*minWLCommY[c][a] for a in minWLCommY[c]), 'comm_pair_cut_y_' + str(c))
                m.addConstr(commYVars[c] >=
                    quicksum(areaVars[a]*minWLCommY2[c][a] for a in minWLCommY2[c]), 'comm_pair_cut_y_2_' + str(c))

        # wirelength computation
        if(len(problem['communications']) > 0):
            m.addConstr(quicksum((commXVars[c]+commYVars[c])*problem['communications'][c] for c in problem['communications']) == CCost, 'CCost_def')

        # io computation
        m.addConstr(quicksum(areaVars[a]*areasIOCost[a] for a in areas) == IOCost, 'IOCost_def')


        m.modelSense = GRB.MINIMIZE


    for paramName in problem['gurobi_params']:
        value = problem['gurobi_params'][paramName]
        m.setParam(paramName, value)

    m.optimize()

    endTime = time.time()
    deltaTime = round((endTime-startTime)*1000) / 1000.0

    print ''
    print '---> total time: ' + str(deltaTime)
    print ''

    result['time'] = deltaTime

    if(m.getAttr('solCount') > 0):
        print('#### solution fuond ####')

        result['status'] = True
        result['objective'] = m.getAttr('ObjVal')

        if not relocation:
            result['metrics'] = {
                'absolute' : {
                    'wirelength' : CCost.getAttr('X') + IOCost.getAttr('X'),
                    'perimeter' : PCost.getAttr('X'),
                    'resources' : RCost.getAttr('X')
                },
                'relative' : {
                    'wirelength' : (CCost.getAttr('X') + IOCost.getAttr('X')) / WLmax ,
                    'perimeter' : PCost.getAttr('X') / Pmax,
                    'resources' : RCost.getAttr('X') / Rmax
                }

                
            }
            
        result['regions'] = {}

        index = 1
        if relocation:
            for a in areas:
                val = areaVars[a].getAttr('X')
                if val > 0.1:
                    info = eval(areaVars[a].getAttr('VarName'))
                    result['regions'][info[0] + '_' + str(index)] = {
                        'x1' : info[1],
                        'y1' : info[2],
                        'x2' : info[3],
                        'y2' : info[4]
                    }
                    index = index + 1
        else:
            for a in areas:
                val = areaVars[a].getAttr('X')
                if val > 0.1:
                    info = eval(areaVars[a].getAttr('VarName'))
                    result['regions'][info[0]] = {
                        'x1' : info[1],
                        'y1' : info[2],
                        'x2' : info[3],
                        'y2' : info[4]
                    }

        
    else:
        print('#### infeasible ####')

    return result

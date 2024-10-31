from FemObject import *
from CubeObject import *


if __name__ == '__main__':

    # volume fraction, default is 0.5
    alpha = 0.5

    # show intermediate results
    showResults = False

    # also evaluate materialized tri/tet inputs
    evalInput =  False

    # examples included in the paper
    objNames = ['cube', 'femur', 'fertility', 'kitten', 'spot', 'venus']
    #objNames = ['JEB', 'buddhaDown', 'buddhaBack', 'buddhaBelly']

    # uncomment to compute only 2D examples (quicker)
    #objNames = ['spot2D', 'bunny2D', 'bar2D']

    res = [['object', 'type', 'method', 'nVerts', 'nCells', 'v/v0', 'c/c0', 'ASJ', 'MSJ']]  
    for objName in objNames:
        # setup input FEM object and apply loads
        fInObj = FemObject('data/%s.frc'%objName, False)
        c0 = fInObj.computeCompliance()
        res.append([objName, fInObj.mTypeInit, None, fInObj.nVerts, fInObj.nElems, 1, 1])

        # write stress file
        fInObj.exportToStress()

        # show applied forces
        if showResults:
            fInObj.showForces()

        # materialize input and evaluate
        if evalInput:
            fInObj.materialize(alpha, False)
            c1 = fInObj.computeCompliance(complianceOnly = True)
            res.append([objName, fInObj.mTypeInit+'_b', False, fInObj.nVerts, fInObj.nElems, fInObj.vol / fInObj.volInit, c1 / c0])

            if fInObj.nDim == 3:
                fInObj = FemObject('data/%s.frc'%objName, False)
                c0 = fInObj.computeCompliance()
                fInObj.materialize(alpha, True)
                c1 = fInObj.computeCompliance(complianceOnly = True)
                res.append([objName, fInObj.mTypeInit+'_w', False, fInObj.nVerts, fInObj.nElems, fInObj.vol / fInObj.volInit, c1 / c0])

        # setup cubification object and optimize
        cfgParams = ConfigParams.fromFile('data/%s.cfg'%objName)
        cObj = CubeObject(cfgParams)
        cObj.optimize()
        cObj.saveState()

        # show deformation
        if showResults:
            cVis = CubeVisual()
            cVis.showPlot(cObj)

        #cObj.loadState()
        cObj.fillGrid()
        cObj.padHull()
        cObj.restoreInitShape()

        # show the optimized deformation
        if showResults:
            cObj.evalAlignment()
            cObj.show(False, False)
            mlab.show()

        # export result lattice
        cObj.exportCellsToMesh()

        # load the aligned structure as FEM input
        fResObj = FemObject('data/%s.frc'%objName, True)
        c1 = fResObj.computeCompliance(complianceOnly = True)
        SJs = fResObj.evalScaledJacobians(False)
        res.append([objName, fResObj.mTypeInit, None, len(cObj.gVerts), len(cObj.cells), fResObj.volInit / fInObj.volInit, c1 / c0, SJs.mean(), SJs.min()])

        # materialize with different modi
        for walled in [False] + [True] * (cObj.nDim == 3):
            for useOpti in [False, True]:
                fResObj = FemObject('data/%s.frc'%objName, True)
                c0 = fResObj.computeCompliance(complianceOnly = True)
                if useOpti:
                    weights = None if objName != 'spot2D' else icdf(fResObj.vmStress)**2 # experimental
                    fResObj.materializeOpti(alpha, walled, weights = weights)
                else:
                    fResObj.materialize(alpha, walled)
                c1 = fResObj.computeCompliance(complianceOnly = True)
                SJs = fResObj.evalScaledJacobians(False)
                mTag = 'w' if walled else 'b'
                res.append([objName, fResObj.mTypeInit + '_' + mTag, useOpti, fResObj.nVerts, fResObj.nElems, fResObj.vol / fResObj.volInit, c1 / c0, SJs.mean(), SJs.min()])

    with open(logDir + 'examples.log', 'w') as fh:
        ln = ', '.join(['% 9s'%s for s in res[0]])
        print(ln)
        fh.write(ln+'\n')
        for r in res[1:]:
            method = ('tau' + '*' * r[2]) if r[2] is not None else ''
            ln = ', '.join(['% 9s'%s for s in [r[0], r[1], method]])
            ln += ', ' + ', '.join(['% 9d'%d for d in [r[3],r[4]]])
            ln += ', ' + ', '.join(['% 0.6f'%f for f in r[5:]])
            print(ln)
            fh.write(ln+'\n')

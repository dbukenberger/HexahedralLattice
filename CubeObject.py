from femUtil import *
from PadObject import *


# quad and hex filter kernels
qKernels = [[1,-1,0,-1,1,0,0,0,0],
            [0,-1,1,0,1,-1,0,0,0],
            [0,0,0,-1,1,0,1,-1,0],
            [0,0,0,0,1,-1,0,-1,1]]
hKernels = [[0,1,0,0,-1,0,0,0,0,0,-1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,1,-1,0,0,0,0,0,0,0,-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,-1,1,0,0,0,0,0,0,0,1,-1,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,-1,0,0,1,0,0,0,0,0,1,0,0,-1,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,1,-1,0,-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,-1,1,0,1,-1,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,-1,1,0,1,-1,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,1,-1,0,-1,1,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,-1,0,0,1,0,0,0,0,0,1,0,0,-1,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,-1,1,0,0,0,0,0,0,0,1,-1,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,1,-1,0,0,0,0,0,0,0,-1,1,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,-1,0,0,0,0,0,-1,0,0,1,0],
            [1,-1,0,-1,-1,0,0,0,0,-1,-1,0,-1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,-1,1,0,-1,-1,0,0,0,0,-1,-1,0,1,-1,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,-1,-1,0,1,-1,0,0,0,0,-1,1,0,-1,-1,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,-1,-1,0,-1,1,0,0,0,0,1,-1,0,-1,-1,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,-1,-1,0,-1,1,0,0,0,0,1,-1,0,-1,-1,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,-1,-1,0,1,-1,0,0,0,0,-1,1,0,-1,-1,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,-1,1,0,-1,-1,0,0,0,0,-1,-1,0,1,-1,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,1,-1,0,-1,-1,0,0,0,0,-1,-1,0,-1,1]]
mKernels = {2: qKernels, 3: hKernels}


class ConfigParams:

    def __init__(self, objName, pLambda, lambdaRamp, autoScale, optiAlign, useNormals, stressWeighted, useConstraints, maxIters, stopDelta, gridRes, finalLayer, useProjection, useTransform, hullTolerance, hullSmoothing, autoSave = True):
        self.objName = os.path.basename(objName).split('.')[0]
        self.fileName = 'data/%s.stress'%self.objName
        
        self.pLambda = pLambda
        self.lambdaRamp = lambdaRamp
        if self.lambdaRamp:
            self.lambdaFun = lambda i: (i/self.lambdaRamp if i < self.lambdaRamp else 1) * self.pLambda
        else:
            self.lambdaFun = lambda i: self.pLambda

        self.autoScale = autoScale
        self.optiAlign = optiAlign
        self.useNormals = useNormals
        self.stressWeighted = stressWeighted
        self.useConstraints = useConstraints
        self.maxIters = maxIters
        self.stopDelta = stopDelta

        self.gridRes = gridRes
        self.finalLayer = finalLayer
        self.useProjection = useProjection
        self.useTransform = useTransform
        self.hullTolerance = hullTolerance
        self.hullSmoothing = hullSmoothing

        self.reset()
        if autoSave:
            self.save()

    @classmethod
    def fromFile(cls, fileName):
        lines = np.loadtxt(fileName, str)
        for line in lines:
            if 'objName' in line[0]:
                fileName = 'data/%s.stress'%line[1]
            if 'pLambda' in line[0]:
                pLambda = float(line[1])
            if 'lambdaRamp' in line[0]:
                lambdaRamp = int(line[1])
            if 'autoScale' in line[0]:
                autoScale = 'True' in line[1]
            if 'optiAlign' in line[0]:
                if 'False' in line[1]:
                    optiAlign = False
                elif 'True' in line[1]:
                    optiAlign = True
                else:
                    optiAlign = list(map(float, line[1].split(',')))
            if 'useNormals' in line[0]:
                useNormals = 'True' in line[1]
            if 'stressWeighted' in line[0]:
                stressWeighted = 'True' in line[1]
            if 'useConstraints' in line[0]:
                useConstraints = 'True' in line[1]
            if 'maxIters' in line[0]:
                maxIters = int(line[1])
            if 'stopDelta' in line[0]:
                stopDelta = float(line[1])
            if 'gridRes' in line[0]:
                gridRes = int(line[1])
            if 'finalLayer' in line[0]:
                finalLayer = 'True' in line[1]
            if 'useProjection' in line[0]:
                useProjection = 'True' in line[1]
            if 'useTransform' in line[0]:
                useTransform = 'True' in line[1]
            if 'hullTolerance' in line[0]:
                hullTolerance = 'True' in line[1]
            if 'hullSmoothing' in line[0]:
                if 'False' in line[1]:
                    hullSmoothing = False
                elif 'True' in line[1]:
                    hullSmoothing = True
                else:
                    hullSmoothing = float(line[1])

        return cls(fileName, pLambda, lambdaRamp, autoScale, optiAlign, useNormals, stressWeighted, useConstraints, maxIters, stopDelta, gridRes, finalLayer, useProjection, useTransform, hullTolerance, hullSmoothing, False)

    def reset(self):
        self.rhoInit = 1e-3
        self.abstol = 1e-5
        self.reltol = 1e-3
        self.mu = 10.0
        self.tao = 2.0
        self.maxIterADMM = 500

    def save(self):
        with open('data/%s.cfg'%self.objName, 'w') as fh:
            fh.write('objName:\t\t%s\n'%self.objName)
            fh.write('pLambda:\t\t%d\n'%self.pLambda)
            fh.write('lambdaRamp:\t\t%d\n'%self.lambdaRamp)
            fh.write('autoScale:\t\t%s\n'%self.autoScale)
            if type(self.optiAlign) == bool:
                fh.write('optiAlign:\t\t%s\n'%self.optiAlign)
            else:
                fh.write('optiAlign:\t\t'+(','.join(['%g']*len(self.optiAlign)))%tuple(self.optiAlign)+'\n')
            fh.write('useNormals:\t\t%s\n'%self.useNormals)
            fh.write('stressWeighted:\t%s\n'%self.stressWeighted)
            fh.write('useConstraints:\t%s\n'%self.useConstraints)
            fh.write('maxIters:\t\t%d\n'%self.maxIters)
            fh.write('stopDelta:\t\t%g\n'%self.stopDelta)
            fh.write('gridRes:\t\t%d\n'%self.gridRes)
            fh.write('finalLayer:\t\t%s\n'%self.finalLayer)
            fh.write('useProjection:\t%s\n'%self.useProjection)
            fh.write('useTransform:\t%s\n'%self.useTransform)
            fh.write('hullTolerance:\t%s\n'%self.hullTolerance)
            fh.write('hullSmoothing:\t%s\n'%self.hullSmoothing)


class CubeObject:
    
    def __init__(self, cfgParams):
        self.cfgParams = cfgParams

        if self.cfgParams.fileName.split('.')[-1] == 'stress':
            self.verts, self.ts, self.flxIdxs, forceVecs, self.fixIdxs, self.vmStress, self.pStress, self.pStressE, self.sMats = loadStressFile(self.cfgParams.fileName)
            self.aIdxs = np.concatenate([self.flxIdxs, self.fixIdxs])
            self.nDim = self.verts.shape[1]
        else:
            return

        if self.nDim == 2:
            self.edges = facesToEdges(self.ts, False)
            unq, inv, cnt = np.unique(cantorPiKV(self.edges), return_inverse = True, return_counts = True)
            self.boundaryEdgeMask = cnt[inv] == 1
            if self.cfgParams.useNormals:
                self.boundaryVertMask = np.zeros(len(self.verts), np.bool_)
                self.boundaryVertMask[np.unique(self.edges[self.boundaryEdgeMask].ravel())] = True
            
                boundaryEdges = self.edges[self.boundaryEdgeMask]
                boundaryLoop = edgesToPath(boundaryEdges)
                bEdges = np.transpose([boundaryLoop, np.roll(boundaryLoop, 1)])
                bEdgeVecs = self.verts[bEdges[:,1]] - self.verts[bEdges[:,0]]
                bVertDirs = np.dot(bEdgeVecs, Mr2D(-np.pi/2)) + np.dot(np.roll(bEdgeVecs, -1, axis=0), Mr2D(-np.pi/2))

                self.vNormals = np.zeros_like(self.verts)
                self.vNormals[boundaryLoop] = normVec(bVertDirs)

                self.pStress[self.aIdxs] = np.dstack([self.vNormals[self.aIdxs], np.dot(self.vNormals[self.aIdxs], Mr2D(np.pi/2))])
        else:
            self.tris = tetrasToFaces(self.ts)
            unq, inv, cnt = np.unique(cantorPiKV(self.tris), return_inverse = True, return_counts = True)
            self.boundaryTrisMask = cnt[inv] == 1
            if self.cfgParams.useNormals:
                if True: # default, normals only
                    self.vNormals = igl.per_vertex_normals(self.verts, self.tris[self.boundaryTrisMask])
                    m = np.bitwise_not(np.any(np.isnan(self.vNormals),axis=1))
                    aMask = np.zeros_like(m)
                    aMask[self.aIdxs] = True
                    for aIdx in self.aIdxs:
                        self.pStress[aIdx] = rotateAsToB(self.pStress[aIdx], self.vNormals[aIdx], self.pStress[aIdx,0])
                else:   # experimental, curvature tensor
                    ts = self.tris[self.boundaryTrisMask]
                    boundaryVertIdxs = np.unique(ts.ravel())
                    pc = igl.principal_curvature(self.verts[boundaryVertIdxs], reIndexIndices(ts))
                    allVertPc = np.zeros_like(self.pStress)
                    allVertPc[boundaryVertIdxs] = np.transpose(np.dstack([cross(pc[0], pc[1]), pc[0], pc[1]]), axes=(0,2,1))
                    self.pStress[self.aIdxs] = allVertPc[self.aIdxs]

        print("#v % 6d, #e % 6d"%(len(self.verts), len(self.ts)))

        self.isRestored = True

        # scale object
        self.BBinit = np.float32([self.verts.min(axis=0), self.verts.max(axis=0)])
        if self.cfgParams.autoScale:
            self.verts -= self.BBinit[0]
            self.verts /= max(self.BBinit[1]-self.BBinit[0])/2
            self.verts -= self.verts.max(axis=0)/2
            self.isRestored = False

        # orient object
        self.MinitRot = np.eye(self.nDim)
        if self.cfgParams.optiAlign:
            if type(self.cfgParams.optiAlign) == list:
                self.MinitRot = Mr2D(np.pi * self.cfgParams.optiAlign[0]) if self.nDim == 2 else Mr3D(np.pi * self.cfgParams.optiAlign[0], np.pi * self.cfgParams.optiAlign[1], np.pi * self.cfgParams.optiAlign[2])
            else:
                w = self.pStressE.sum(axis=1).reshape(-1,1,1) #* 0 + 1
                self.MinitRot = computePrincipalStress((self.sMats * w/w.sum()).sum(axis=0))[0]
                
            self.MinitRot = computeMinTransformation(self.MinitRot)
            
            self.verts = np.dot(self.verts, self.MinitRot.T)
            self.pStress = innerNxM(self.pStress, [self.MinitRot])
            self.isRestored = False

        self.sf = norm(self.verts.max(axis=0)-self.verts.min(axis=0)) / 1000
            
        self.cVerts = self.verts.copy()
        self.cVertss = [self.cVerts.copy()]

        self.ks = []

        self.energy = 0.0
        self.relativeDeltaV = sys.float_info.max
        self.tLocal = []
        self.tGlobal = []

        self.setupMatrices()
        self.Rs = [self.R.copy()]
    
        if self.cfgParams.useConstraints:
            cIdxs = self.aIdxs
            self.cRHS = np.zeros_like(self.verts)
            self.cRHS[cIdxs] = self.verts[cIdxs]
            E = sparse.dok_matrix((self.nVerts, self.nVerts), dtype=np.float64)
            E[cIdxs,cIdxs] = 1
            E = E.tocsc()
            M = (self.L.T @ self.L + E).tocsc()
        else:
            M = (self.L.T @ self.L).tocsc()

        # check sparseCond(M) and add eps * I
        c = 1e-12
        trys = 0
        print('Construct solver', M.shape)
        while not hasattr(self, 'solver'):
            try:
                Mplus = (M + sparse.eye(M.shape[0]).tocsc() * c) if trys else M                
                self.solver = pypardiso.factorized(Mplus) if pypardisoFound else spla.factorized(Mplus)
            except RuntimeError as e:
                if 'singular' in e.args[0]:
                    c *= 10
                    trys += 1
                    print('Desingularize trys:', trys)
        if trys:
            print('Mat singular fix: ', c, trys)
        else:
            print('SPLA solver done.')
            
    def setupMatrices(self):      
        self.L = igl.cotmatrix(np.float64(self.verts), self.ts)
        self.M = igl.massmatrix(self.verts, self.ts)
        self.vMass = self.M.diagonal()
        self.volumeInit = self.vMass.sum()
        self.centerInit = (self.verts.min(axis=0) + self.verts.max(axis=0))/2

        self.nVerts = len(self.verts)
        self.nNeighbors = self.L.getnnz(axis=0).max()-1

        nFan = 1 if self.nDim == 3 else 3
        self.nIdxs = np.ones((self.nVerts,2,self.nNeighbors*nFan),np.int32) * np.arange(self.nVerts)[...,None,None]
        self.nWeights = np.zeros_like(self.nIdxs[:,0], dtype=np.float32)
        self.Ralpha = np.ones((self.nVerts, self.nNeighbors*nFan)) / 2

        if self.nDim == 2:
            tm = trimesh.Trimesh(self.verts, self.ts)
            bLoop = np.bool_(igl.is_border_vertex(self.verts, self.ts))

        for idx, col in tqdm(enumerate(self.L), total = self.L.shape[0], ascii=True, desc='weights'):
            msk = col.indices != idx
            indices = col.indices[msk]
            values  = col.data[msk]
            if True or self.nDim == 3: # default, only spokes
                self.nIdxs[idx,0,:len(indices)] = indices
                self.nWeights[idx,:len(indices)] = values
            else: # experimental for 2D, with rims
                fIdxs = tm.vertex_faces[idx]
                es = facesToEdges(tm.faces[fIdxs[fIdxs >= 0]])

                p = edgesToPath(es[np.all(es!=idx,axis=1)])
                es = np.transpose([p, np.roll(p, -1)])
                es = np.vstack([es, es[:,::-1]])
                es = np.vstack([es, np.transpose([p, [idx]*len(p)])])

                self.nIdxs[idx,:,:len(es)] = es.T           
                em = bLoop[es].all(axis=1)
                em[:len(p)*2] = True
                
                cM = igl.cotmatrix(self.verts[col.indices], reIndexIndices(tm.faces[fIdxs[fIdxs>=0]]))
                fes = reIndexIndices(es)
                self.nWeights[idx,:len(es)] = cM[fes[:,0], fes[:,1]]
                self.nWeights[idx,:len(es)] *= (1+em)

        self.Ralpha = self.Ralpha.reshape(self.nVerts, -1, 1, 1)

        self.R = np.repeat([np.eye(self.nDim, dtype = np.float32)], self.nVerts, axis=0)
        self.zAll = np.zeros((self.nVerts, self.nDim))
        self.uAll = np.zeros((self.nVerts, self.nDim))
        self.rhoAll = np.full(self.nVerts, self.cfgParams.rhoInit)

        self.zsAll = np.zeros((self.nVerts, self.nDim, self.nDim)) + self.pStress
        self.usAll = np.zeros((self.nVerts, self.nDim, self.nDim))
        self.rhosAll = np.full((self.nVerts, self.nDim), self.cfgParams.rhoInit)
        self.lambdasAll = np.ones_like(self.vMass) * self.cfgParams.pLambda
        
        self.energyVec = np.zeros(self.nVerts, dtype = np.float32)

        self.truHoods = (self.verts[self.nIdxs[:,0]] - self.verts[self.nIdxs[:,1]]) * self.nWeights[:,:,None]

    def optimize(self, printProgress = True):
        self.cVertss = list(self.cVertss)

        pStressE = self.pStressE[:,0] / self.pStressE[:,0].mean()
        w = pStressE ** 0.5
        w = np.ones_like(pStressE)
        w /= w.sum()
        stressWeights = icdf(self.vmStress)

        ctsStr = lambda cts: (', cts: ' + ', '.join(['%f']*4))%tuple([cts.min(), cts.max(), np.dot(w, cts), (cts < np.deg2rad(0.1)).sum()/self.nVerts])

        start = time()
        for i in range(abs(self.cfgParams.maxIters)):
            self.lambdasAll[:] = self.cfgParams.lambdaFun(i)

            if self.cfgParams.useNormals:
                self.lambdasAll[self.aIdxs] *= 5

            if not i:
                heatScals = computeRotationAngles(innerNxM(self.pStress, self.R)) if self.nDim == 2 else computeMinEulerAngles(np.eye(3), innerNxM(self.pStress, self.R))
                self.ctss = [heatScals]

            if self.cfgParams.stressWeighted:
                self.lambdasAll *= (1 + stressWeights ** 0.5)

            st = time()
            k = self.localStep()
            self.tLocal.append(time() - st)
        
            st = time()
            self.globalStep()
            self.tGlobal.append(time() - st)

            self.relativeDeltaV = np.max(np.abs(self.cVertss[-1] - self.cVertss[-2])) / np.max(np.abs(self.cVertss[-1] - self.verts))

            heatScals = computeRotationAngles(innerNxM(self.pStress, self.R)) if self.nDim == 2 else computeMinEulerAngles(np.eye(3), innerNxM(self.pStress, self.R))
            self.ctss.append(heatScals)

            if printProgress:
                if not i:
                    print("It:% 4s (%3d), E: %f, rDV: %f, tL: %f, tG: %f"%("X", k, self.energy, self.relativeDeltaV, self.tLocal[-1], self.tGlobal[-1]) + ctsStr(self.ctss[0]))
                print("It:% 4d (%3d), E: %f, rDV: %f, tL: %f, tG: %f"%(i, k, self.energy, self.relativeDeltaV, self.tLocal[-1], self.tGlobal[-1]) + ctsStr(self.ctss[-1]))
            if (self.cfgParams.maxIters > 0 and self.relativeDeltaV < self.cfgParams.stopDelta):
                break
            
        tL = sum(self.tLocal)
        tG = sum(self.tGlobal)
        tS = tL+tG
        print('Time (s, l, g): %f, %f, %f'%(tS, tL, tG))
        print('Time (s, l, g): %f, %f, %f'%(tS/tS, tL/tS, tG/tS))

        self.cVertss = np.float64(self.cVertss)

    def globalStep(self):
        R = self.Ralpha * np.take(self.R, self.nIdxs[:,0], axis=0) + (1-self.Ralpha) * np.take(self.R, self.nIdxs[:,1], axis=0)
        rhs = np.sum((R @ self.truHoods[...,None]).squeeze(), axis=1)

        self.cVerts = self.solver(self.L @ rhs + self.cRHS) if self.cfgParams.useConstraints else self.solver(self.L.T @ rhs)

        if not self.cfgParams.useConstraints:
            cVerts = self.cVerts - self.cVerts.min(axis=0)
            self.cMass = igl.massmatrix(self.cVerts, self.ts, 0).diagonal()
            cVerts *= (self.volumeInit / self.cMass.sum()) ** (1/self.nDim)
            self.cVerts = cVerts + (self.centerInit - cVerts.max(axis=0)/2)

        self.cVertss.append(self.cVerts.copy())

    def localStep(self):
        self.energyVec *= 0
        rotHoods = (self.cVerts[self.nIdxs[:,0]] - self.cVerts[self.nIdxs[:,1]])
        Ss = np.transpose(self.truHoods, axes=[0,2,1]) @ rotHoods

        msk = np.ones(len(Ss), np.bool_)

        zs = self.zsAll.copy()
        us = self.usAll.copy()
        ns = self.pStress.copy()
        rhos = self.rhosAll.copy()

        for k in range(self.cfgParams.maxIterADMM):
            S = Ss[msk] + outer2dWeighted(ns[msk], zs[msk] - us[msk], rhos[msk]) #/ self.nDim
            R = np.transpose(orthogonalizeOrientations(S), axes=[0,2,1])

            # z step
            zsOld = zs[msk] + 0
            Rn = innerNxM(ns[msk], R)
            zs[msk] = lassoShrink(Rn + us[msk], (self.lambdasAll[msk] * self.vMass[msk]).reshape(-1,1,1) / rhos[msk].reshape(-1,self.nDim,1))

            # u step
            us[msk] += Rn - zs[msk]
            rNorms = norm(zs[msk] - Rn)
            sNorms = norm(zs[msk] - zsOld) * rhos[msk]

            # rho setup
            rMsk = np.transpose([msk.copy()]*self.nDim)
            rMsk[msk] *= rNorms > self.cfgParams.mu * sNorms
            sMsk = np.transpose([msk.copy()]*self.nDim)
            sMsk[msk] *= np.bitwise_and(np.bitwise_not(rMsk[msk]), sNorms > self.cfgParams.mu * rNorms)
            rhos[rMsk] *= self.cfgParams.tao
            us[rMsk] /= self.cfgParams.tao
            rhos[sMsk] /= self.cfgParams.tao
            us[sMsk] *= self.cfgParams.tao

            # stopping
            epsPri = np.sqrt(2.0 * self.nDim) * self.cfgParams.abstol + self.cfgParams.reltol * np.max([norm(Rn).max(axis=1), norm(zs[msk]).max(axis=1)], axis=0)
            epsDual = np.sqrt(self.nDim) * self.cfgParams.abstol + self.cfgParams.reltol * norm(rhos[msk].reshape(-1,self.nDim,1) * us[msk]).max(axis=1)

            bMs = np.bitwise_and(rNorms < epsPri.reshape(-1,1), sNorms < epsDual.reshape(-1,1)).all(axis=1)
            bMsk = msk.copy()
            bMsk[msk] *= bMs
            if np.any(bMsk):
                self.zsAll[bMsk] = zs[bMsk].copy()
                self.usAll[bMsk] = us[bMsk].copy()
                self.rhosAll[bMsk] = rhos[bMsk].copy()
                self.R[bMsk] = R[bMs].copy()

                RdVminusDU = innerNxM(self.truHoods[bMsk], R[bMs]) - rotHoods[bMsk]
                self.energyVec[bMsk] = self.cfgParams.pLambda * self.vMass[bMsk] * np.sum(np.abs(Rn[bMs]), axis=2).max(axis=1)
                #self.energyVec[bMsk] = self.lambdasAll[bMsk] * self.vMass[bMsk] * np.sum(np.abs(Rn[bMs]), axis=2).max(axis=1)
                self.energyVec[bMsk] += (inner1d(RdVminusDU * self.nWeights[bMsk][...,None], RdVminusDU)**2).sum(axis=1) / 2

            msk *= np.bitwise_not(bMsk)
            if not msk.sum():
                break

        self.energy = np.sum(self.energyVec)
        self.Rs.append(self.R.copy())
        return k

    def fillGrid(self):
        print('Generating inner grid.')
        m = self.cfgParams.gridRes
        n = m + 1

        e = self.cVerts.max(axis=0) - self.cVerts.min(axis=0)
        ns = np.int32((e/e.min()) * n + 0.5)
        if self.nDim == 2:
            nx,ny = ns
            rx = np.linspace(0, 1, nx)
            ry = np.linspace(0, 1, ny)
            verts = np.transpose([np.tile(rx, ny), np.repeat(ry, nx)])
        else:
            nx,ny,nz = ns
            rx = np.linspace(0, 1, nx)
            ry = np.linspace(0, 1, ny)
            rz = np.linspace(0, 1, nz)
            verts = np.transpose([np.tile(rx, ny*nz), np.repeat(np.tile(ry, nz), nx), np.repeat(rz, nx*ny)])

        self.cDims = (e / ns)
        offset = self.cDims * self.cfgParams.finalLayer
        verts *= self.cVerts.max(axis=0) - self.cVerts.min(axis=0) - offset
        verts += self.cVerts.min(axis=0) + offset/2
        tVerts = self.cVerts[self.ts]

        innerOnly = (self.cVerts, self.tris[self.boundaryTrisMask]) if self.nDim == 3 else None
        bWeights, tIdxs = computeBarycentricWeights(verts, tVerts, self.nDim, innerOnly, self.cfgParams.hullTolerance * self.cDims.mean())

        if self.nDim == 2:
            vIdxs = lambda i: [i, i+1, i+1+nx, i+nx]
            ijs = np.vstack([[i,j] for j in range(ny-1) for i in range(nx-1)])
            cells = np.vstack([vIdxs(i+nx*j) for i,j in ijs])

            mx, my = nx-1, ny-1
            nIdxs = []
            for x in [-1,0,1]:
                for y in [-1,0,1]:
                    nIdxs.append(x + mx*y)
            self.cnIdxs = np.arange(len(cells)).reshape(-1,1) + np.tile(nIdxs, len(cells)).reshape(-1,len(nIdxs))
        else:
            vIdxs = lambda i: [i, i+1, i+1+nx, i+nx, i+nx*ny, i+1+nx*ny, i+1+nx+nx*ny, i+nx+nx*ny]
            ijks = np.vstack([[i,j,k] for k in range(nz-1) for j in range(ny-1) for i in range(nx-1)])
            cells = np.vstack([vIdxs(i+nx*j+nx*ny*k) for i,j,k in ijks])

            mx, my, mz = nx-1, ny-1, nz-1
            nIdxs = []
            for x in [-1,0,1]:
                for y in [-1,0,1]:
                    for z in [-1,0,1]:
                        nIdxs.append(x + mx*y + mx*my*z)
            self.cnIdxs = np.arange(len(cells)).reshape(-1,1) + np.tile(nIdxs, len(cells)).reshape(-1,len(nIdxs))

        cMsk = np.all(tIdxs[cells] > -1, axis=1)

        self.cMsk = cMsk.copy()
        print('nCells:', self.cMsk.sum(), len(cells))

        # filter hull intersects
        if self.nDim == 2:
            if not self.cfgParams.hullTolerance:
                hullEdges = self.cVerts[self.edges[self.boundaryEdgeMask]]
                for cIdx, (cM, cell) in enumerate(zip(cMsk, cells)):
                    if cM:
                        es = faceToEdges(cell)
                        for edge in faceToEdges(cell):
                            hMsk = edgesIntersect2D(hullEdges[:,0], hullEdges[:,1], verts[edge[0]], verts[edge[1]])
                            if hMsk.any():
                                cMsk[cIdx] = False
                                break
        else:
            if embreeFound and not self.cfgParams.hullTolerance:
                embreeDevice = rtc.EmbreeDevice()
                scene = rtcs.EmbreeScene(embreeDevice)
                mesh = TriangleMesh(scene, self.cVerts[self.tris[self.boundaryTrisMask]])
                for cIdx, (cM, cell) in enumerate(zip(cMsk, cells)):
                    if cM:
                        es = hexaToEdges(cell)
                        eDirs, eLens = normVec(verts[es[:,1]] - verts[es[:,0]], True)
                        dsts = scene.run(np.float32(verts)[es[:,0]], np.float32(eDirs), query='DISTANCE')
                        if any(dsts < eLens):
                            cMsk[cIdx] = False

        # filter kernels
        nStart = cMsk.sum()
        validNeighborMsk = np.bitwise_and(self.cnIdxs >= 0, self.cnIdxs < len(cells))
        i = 0
        for i in range(len(cells)):
            cMsks = cMsk[self.cnIdxs * validNeighborMsk] * validNeighborMsk
            orphanMsk = np.bitwise_and(np.sum(cMsks, axis=1) == 1, cMsks[:,(self.nDim**3)//2])
            if np.any(orphanMsk):
                cMsk[orphanMsk] = False
                continue
            
            cOrder = np.argsort(np.sum(cMsks,axis=1))
            for cIdx in cOrder:
                if cMsk[cIdx]:
                    msk = cMsk[self.cnIdxs[cIdx] * validNeighborMsk[cIdx]] * validNeighborMsk[cIdx]
                    dps = np.dot(mKernels[self.nDim], msk)
                    cMsk[cIdx] = np.all(dps != 2)
                    if not cMsk[cIdx]:
                        break
            else:
                break
        print('filtered:', nStart - cMsk.sum())

        print('nCells:', cMsk.sum(), len(cells))
        cells = cells[cMsk]
        usedVertIdxs = np.unique(cells.ravel())

        self.bWeights = bWeights[usedVertIdxs]
        self.gVertTIdxs = tIdxs[usedVertIdxs]
        self.gVerts = verts[usedVertIdxs]
        self.cells = reIndexIndices(cells) # flip lr?
        if self.nDim == 2:
            self.cells = self.cells[:,::-1]

        diffs = self.verts - self.cVerts
        ts = self.ts[self.gVertTIdxs]
        ds = diffs[ts]
        t = innerVxM(self.bWeights, np.transpose(ds,axes=[0,2,1]))
        self.gVertz = self.gVerts + t

        self.gInnerMask = np.ones(len(t), np.bool_)
        
    def computeHulls(self, cubed = False, updateHullIdxs = True):
        if self.nDim == 2:
            triHullEdges = filterForSingleEdges(facesToEdges(self.ts, False))
            tLoops = edgesToPaths(triHullEdges)
            qHullEdges = filterForSingleEdges(facesToEdges(self.cells, False))
            qLoops = edgesToPaths(qHullEdges)

            tVertIdxs = np.unique(flatten(tLoops))
            qVertIdxs = np.unique(flatten(qLoops))

            self.qLoops = qLoops

            if updateHullIdxs:
                if hasattr(self, 'hullElementHashs'):
                    self.innerHullElementHashs = self.hullElementHashs.copy()
                    self.innerHullVertIdxs = self.hullVertIdxs.copy()
                self.hullElementHashs = cantorPiV(qHullEdges)
                self.hullVertIdxs = qVertIdxs

            tVerts = self.cVerts[tVertIdxs] if cubed else self.verts[tVertIdxs]
            qVerts = self.gVerts[qVertIdxs] if cubed else self.gVertz[qVertIdxs]
            tEdges = reIndexIndices(np.vstack([pathToEdges(tLoop) for tLoop in tLoops]))
            qEdges = reIndexIndices(np.vstack([pathToEdges(qLoop) for qLoop in qLoops]))

            return tVerts, tEdges, qVerts, qEdges
        else:
            tVerts = self.cVerts if cubed else self.verts
            oTris = self.tris[self.boundaryTrisMask]
            tVerts = tVerts[np.unique(oTris.ravel())]
            tris = reIndexIndices(oTris)

            quads = hexasToFaces(self.cells)
            unq, inv, cnt = np.unique(cantorPiKV(quads), return_inverse = True, return_counts = True)
            boundaryQuadsMask = cnt[inv] == 1

            if updateHullIdxs:
                if hasattr(self, 'hullElementHashs'):
                    self.innerHullElementHashs = self.hullElementHashs.copy()
                    self.innerHullVertIdxs = self.hullVertIdxs.copy()
                self.hullElementHashs = cantorPiKV(quads[boundaryQuadsMask])
                self.hullVertIdxs = np.unique(quads[boundaryQuadsMask].ravel())

            qVerts = self.gVerts if cubed else self.gVertz
            self.oQuads = quads[boundaryQuadsMask]
            qVertIdxs = np.unique(self.oQuads.ravel())
            qVerts = qVerts[np.unique(self.oQuads.ravel())]
            quads = reIndexIndices(self.oQuads)

            return tVerts, tris, qVerts, quads

    def padHull(self):
        print('Padding hull.')
        if self.nDim == 2:
            tVerts, tEdges, qVerts, qEdges = self.computeHulls(self.cfgParams.useTransform)
            self.pVerts = PadObject.getDeformedVerts(tVerts, tEdges, qVerts, qEdges, 0.1, True)

            newCells = []
            qEs = np.vstack([pathToEdges(qLoop) for qLoop in self.qLoops])
            pEdges = qEdges + len(self.gVertz)
            for qe, pe in zip(qEs[:,::-1], pEdges):
                newCells.append(np.concatenate([qe, pe]))
        else:
            tVerts, tris, qVerts, quads = self.computeHulls(self.cfgParams.useTransform)
            self.pVerts = PadObject.getDeformedVerts(tVerts, tris, qVerts, quads[:,::-1], 0.1, self.cfgParams.useProjection)
            newHullVertIdxs = np.unique(quads.ravel()) + len(self.gVertz)

            newCells = []
            for oq, pq in zip(self.oQuads, quads):
                newCells.append(np.concatenate([oq[::-1], pq[::-1] + len(self.gVertz)]))

        self.cells = np.vstack([self.cells, newCells])
        bWeights, tIdxs = computeBarycentricWeights(self.pVerts, self.cVerts[self.ts], self.nDim)

        if self.cfgParams.hullTolerance:
            outMask = np.any(self.bWeights < 0, axis=1)

            if self.nDim == 2:
                self.gVerts[outMask] = self.cVerts[self.ts[self.gVertTIdxs[outMask]]].mean(axis=1)
                self.gVertz[outMask] = self.verts[self.ts[self.gVertTIdxs[outMask]]].mean(axis=1)
            else:               
                qNormals = igl.per_vertex_normals(self.pVerts, quadsToTris(np.vstack([quads[:,::-1], np.roll(quads[:,::-1], 1, axis=1)])))

                self.gVerts[self.hullVertIdxs]  = self.pVerts - qNormals * self.cDims.mean()/2
                self.bWeights[self.hullVertIdxs] = bWeights
                self.gVertTIdxs[self.hullVertIdxs] = tIdxs

                diffs = self.verts - self.cVerts
                ts = self.ts[self.gVertTIdxs]
                ds = diffs[ts]
                t = innerVxM(self.bWeights, np.transpose(ds,axes=[0,2,1]))
                self.gVertz = self.gVerts + t

        self.bWeights = np.vstack([self.bWeights, bWeights])
        self.gVertTIdxs = np.concatenate([self.gVertTIdxs, tIdxs])
        if self.cfgParams.useTransform:
            self.gVerts = np.vstack([self.gVerts, self.pVerts])
            diffs = self.verts - self.cVerts
            ds = diffs[self.ts[tIdxs]]
            t = innerVxM(bWeights, np.transpose(ds, axes=[0,2,1]))
            self.pVertz = self.pVerts + t
        else: # experimental
            self.gVerts = np.vstack([self.gVerts, qVerts])
            self.pVertz = self.pVerts
        self.gVertz = np.vstack([self.gVertz, self.pVertz])

        vms = self.vmStress[self.ts[self.gVertTIdxs]]
        self.vmStrezz = inner1d(self.bWeights, vms)

        self.gInnerMask = np.concatenate([self.gInnerMask, np.zeros(len(self.pVerts), np.bool_)])

        self.computeHulls()
        if self.cfgParams.hullSmoothing:
            self.smoothenHull()

    def smoothenHull(self):
        es = facesToEdges(self.cells) if self.nDim == 2 else hexasToEdges(self.cells)
        newVertPozs = np.zeros((len(self.innerHullVertIdxs), self.nDim), np.float32)
        newVertPoss = np.zeros((len(self.innerHullVertIdxs), self.nDim), np.float32)

        cells = self.cells if self.nDim == 2 else hexOrderBT2Dg(self.cells)
        for k in range(30):
            SJs = computeJacobians(self.gVertz[cells], True)
            sjMask = SJs < (0.01 if type(self.cfgParams.hullSmoothing) is bool else self.cfgParams.hullSmoothing)

            print('MSJ:', SJs.min(), sjMask.sum(), len(sjMask))
            if not np.any(sjMask):
                break
                
            vIdxs = cells[sjMask]
            vIdxz = np.unique(np.concatenate([es[np.any(es == vIdx, axis=1)].ravel() for vIdx in vIdxs]))
            
            newVertPozs = np.zeros((len(vIdxz), self.nDim), np.float32)
            newVertPoss = np.zeros((len(vIdxz), self.nDim), np.float32)
            for i, vIdx in enumerate(vIdxz):
                nIdxs = np.unique(es[np.any(es == vIdx, axis=1)].ravel())

                ws = [0.75, 0.25]
                if k >= 5 and vIdx in vIdxs:
                    optiPt = computeOptiPoint(vIdx, cells, self.gVertz)
                    if vIdx in self.hullVertIdxs:
                        ws = [0.5, 0.5]
                else:
                    optiPt = self.gVertz[nIdxs].mean(axis=0)
                newVertPozs[i] = np.dot(ws, [optiPt, self.gVertz[vIdx]])
                newVertPoss[i] = self.gVerts[nIdxs].mean(axis=0)
            
            self.gVertz[vIdxz] = newVertPozs
            self.gVerts[vIdxz] = newVertPoss

            # reproject outer hull
            tVerts, tElems, qVerts, qElems = self.computeHulls(False, updateHullIdxs = False)
            pVerts = PadObject.getDeformedVerts(tVerts, tElems, qVerts, qElems, 0.1, self.nDim < 3)

            hMsk = np.bool_([hvi in vIdxz for hvi in self.hullVertIdxs])
            self.gVertz[self.hullVertIdxs[hMsk]] = pVerts[hMsk]
        else:
            print('Untangling failed, MSJ:', SJs.min())    

    def evalScaledJacobians(self):
        if not hasattr(self, 'cells'):
            print('object has no cells yet')
            return
        cells = self.cells if self.nDim == 2 else hexOrderBT2Dg(self.cells)
        self.SJs = computeJacobians(self.gVertz[cells], True)
        return self.SJs

    def evalAlignment(self):
        self.gStress = np.zeros((len(self.gVertz), self.nDim, self.nDim), np.float32)
        self.gStressE = np.zeros((len(self.gVertz), self.nDim), np.float32)

        sMats = (self.sMats[self.ts[self.gVertTIdxs]] * self.bWeights.reshape(-1,self.nDim+1,1,1)).sum(axis=1)

        for gIdx in range(len(self.gVertz)):
            self.gStress[gIdx], self.gStressE[gIdx] = computePrincipalStress(sMats[gIdx])

        if not self.isRestored:
            self.gStress = innerNxM(self.gStress, [self.MinitRot])
       
        self.gHeat = np.zeros(len(self.gStress), np.float32)
        self.gNeighbors = []
        gEdges = facesToEdges(self.cells) if self.nDim == 2 else hexasToEdges(self.cells)

        for vIdx in range(len(self.gVerts)):
            nEdges = gEdges[np.any(gEdges == vIdx, axis=1)]
            self.gNeighbors.append(nEdges[nEdges != vIdx])
            nDirs = normVec(self.gVertz[self.gNeighbors[-1]] - self.gVertz[vIdx])

            self.gHeat[vIdx] = np.arccos(np.clip(np.abs(np.dot(self.gStress[vIdx], nDirs.T)),0,1).max(axis=0)).mean()

        gHeat = np.rad2deg(self.gHeat)
        for heat in [gHeat, gHeat[self.gInnerMask]]:
            vals = [heat.min(), heat.max(), heat.mean()] + [(heat < d).sum()/len(heat) for d in [1,5,10]]
            print('Ea: ' + ('%0.6f, '*len(vals))%tuple(vals))

    def show(self, cubed = True, withField = True):
        mlab.figure()

        es = facesToEdges(self.ts) if self.nDim == 2 else tetsToEdges(self.ts)
        eTris = toEdgeTris(es)

        verts = self.cVerts if cubed else self.verts
        verts = pad2Dto3D(verts) if self.nDim == 2 else verts
        col = (1,1,1) if cubed else (0,0,0)

        # BBox
        M = (mat2Dto3D(self.MinitRot) if self.nDim == 2 else self.MinitRot) if cubed else np.eye(3)
        vMin = np.dot(verts, M.T).min(axis=0)
        vMax = np.dot(verts, M.T).max(axis=0)
        vs = (cubeVerts/2 + 0.5) * (vMax-vMin) + vMin
        vs = np.dot(vs, M)
        bx,by,bz = vs.T
        mlab.triangular_mesh(bx,by,bz, toEdgeTris(facesToEdges(sixCubeFaces)), representation = 'mesh', color = (0.5,0.5,0.5), tube_radius = self.sf*0.5)

        # tris / tets
        x,y,z = verts.T
        mPlot = mlab.triangular_mesh(x, y, z, eTris, representation = 'mesh',  color = col, tube_radius = self.sf/5)

        if withField:
            x,y,z = np.repeat(verts.T, self.nDim, axis=1)
            pStress = np.vstack(innerNxM(self.pStress, self.R)) if cubed else self.pStress
            sDirs = np.vstack(pStress)
            u,v,w = pad2Dto3D(sDirs).T if self.nDim == 2 else sDirs.T
            scals = np.tile(np.arange(self.nDim)[::-1], len(verts))
            self.sPlot = mlab.quiver3d(x, y, z, u, v, w, scalars = scals)
            self.sPlot.glyph.color_mode = 'color_by_scalar'
            self.sPlot.glyph.glyph_source.glyph_source.glyph_type = 'dash'
            self.sPlot.glyph.glyph_source.glyph_position = 'center'

        # quads / hexas
        if hasattr(self, 'gVertz'):
            verts = pad2Dto3D(self.gVertz) if self.nDim == 2 else self.gVertz
            x,y,z = np.repeat(verts.T, self.nDim, axis=1)
            sDirs = np.vstack(self.gStress)

            if not cubed and withField:
                u,v,w = pad2Dto3D(sDirs).T if self.nDim == 2 else sDirs.T
                scals = np.tile(np.arange(self.nDim), len(verts))
                self.sPlot = mlab.quiver3d(x, y, z, u, v, w, scalars = scals)
                self.sPlot.glyph.color_mode = 'color_by_scalar'
                self.sPlot.glyph.glyph_source.glyph_source.glyph_type = 'dash'
                self.sPlot.glyph.glyph_source.glyph_position = 'center'
                self.sPlot.module_manager.scalar_lut_manager.lut.table = rgb2rgba(np.eye(3)*255)

        if hasattr(self, 'cells'):
            mSJs = computeJacobians(self.gVertz[self.cells], True)
            self.gHeat[:] = 1
            for c, mSJ in zip(self.cells, mSJs):
                self.gHeat[c] = np.minimum(self.gHeat[c], mSJ)

            eTris = toEdgeTris(facesToEdges(self.cells) if self.nDim == 2 else hexasToEdges(self.cells))
            gVerts = self.gVerts if cubed else self.gVertz
            x,y,z = pad2Dto3D(gVerts).T if self.nDim == 2 else gVerts.T
            mPlot = mlab.triangular_mesh(x, y, z, eTris, scalars = self.gHeat, representation = 'mesh', tube_radius = self.sf)

    def restoreInitShape(self):
        if self.isRestored:
            print('Skip Restore')
            return
        print('Restore ...')

        queue = [self.verts, self.cVerts]
        if hasattr(self, 'pVerts'):
            queue += [self.pVerts, self.pVertz, self.gVerts, self.gVertz]
        elif hasattr(self, 'gVerts'):
            queue += [self.gVerts, self.gVertz]

        for verts in queue:
            verts[:] = np.dot(verts, self.MinitRot)
            verts *= (self.BBinit[1]-self.BBinit[0]).max()/2
            verts += self.BBinit.mean(axis=0)

        if len(self.cVertss) > 1:
            for cVerts in self.cVertss:
                cVerts[:] = np.dot(cVerts, self.MinitRot)
            self.cVertss *= (self.BBinit[1]-self.BBinit[0]).max()/2
            self.cVertss[:] += self.BBinit.mean(axis=0)

        self.pStress = innerNxM(self.pStress, [self.MinitRot.T])
        if hasattr(self, 'gOri'):
            self.gStress = innerNxM(self.gStress, [self.MinitRot.T])
            self.gOri = innerNxM(self.gOri, [self.MinitRot.T])

        self.sf = norm(self.verts.max(axis=0)-self.verts.min(axis=0)) / 1000
        self.isRestored = True

    def saveState(self):
        np.savez_compressed(tmpDir + '%s%s.npz'%(self.cfgParams.objName, 'Cubed'), cVerts = self.cVerts, R = self.R, isRestored = self.isRestored, protocol = 2)
        np.savez_compressed(tmpDir + '%s%s.npz'%(self.cfgParams.objName, 'State'), cVertss = self.cVertss, Rs = self.Rs, ctss = self.ctss, protocol = 2)

    def loadState(self, light = False):
        cubedName = tmpDir + '%s%s.npz'%(self.cfgParams.objName, 'Cubed')
        if os.path.exists(cubedName):
            fileIn = np.load(cubedName, allow_pickle=True, encoding = 'latin1')
            self.cVerts = fileIn['cVerts']
            self.R = fileIn['R']
            self.isRestored = fileIn['isRestored']
                           
        stateName = tmpDir + '%s%s.npz'%(self.cfgParams.objName, 'State')
        if not light and os.path.exists(stateName):
            fileIn = np.load(stateName, allow_pickle=True, encoding = 'latin1')
            self.cVertss = fileIn['cVertss']
            self.Rs = fileIn['Rs']
            self.ctss = fileIn['ctss']

    def exportHexasToObj(self, fileName = None, cubed = False):
        fileName = 'data/%sHexas.obj'%self.cfgParams.objName if fileName is None else fileName
        verts = []
        faces = []
        gVerts = self.gVerts if cubed else self.gVertz
        for i, cell in enumerate(self.cells):
            verts.append(gVerts[np.unique(cell.ravel())])
            faces.append(reIndexIndices(hexaToFaces(cell)) + i*8)
        writeObjFile(fileName, np.vstack(verts), np.vstack(faces))

    def exportHexasToPly(self, fileName = None, cubed = False):
        fileName = 'data/%s%sHexas.ply'%(self.cfgParams.objName, 'Cubed' if cubed else 'Res') if fileName is None else fileName
        verts = []
        cols = []
        faces = []
        gVerts = self.gVerts if cubed else self.gVertz
        
        for cIdx, cell in enumerate(self.cells):
            vIdxs = cell[np.argsort(cell)]
            verts.append(gVerts[vIdxs])
            cols.append(self.gHeat[vIdxs])
            faces.append(reIndexIndices(hexaToFaces(cell)) + cIdx * 8)
        writePlyFile(fileName, np.vstack(verts), np.vstack(faces)[:,::-1], verticesColors = np.transpose([np.concatenate(cols)]*3))

    def exportTetsToPly(self, fileName = None, cubed = False):
        fileName = 'data/%s%sTets.ply'%(self.cfgParams.objName, 'Cubed' if cubed else 'Init') if fileName is None else fileName
        vs = self.cVerts if cubed and hasattr(self, 'cVerts') else self.verts
        verts = []
        cols = []
        tris = []
        tTris = tetraToFaces(np.arange(4))
        cs = self.ctss[-1] if cubed else self.ctss[0]
        for tIdx, tet in enumerate(self.ts):
            verts.append(vs[tet])
            cols.append(cs[tet])
            tris.append(tTris + tIdx * 4)

        writePlyFile(fileName, np.vstack(verts), np.vstack(tris), verticesColors = np.transpose([np.concatenate(cols)]*3))

    def exportHullsToObj(self, fileName = None, cubed = False):
        fileNameTris = 'data/%sHullTris.obj'%self.cfgParams.objName if fileName is None else fileName
        fileNameQuads = 'data/%sHullQuads.obj'%self.cfgParams.objName if fileName is None else fileName

        tVerts, tris, qVerts, quads = self.computeHulls(cubed)
        writeObjFile(fileNameTris, tVerts, tris)
        writeObjFile(fileNameQuads, qVerts, quads)
            
    def exportCellsToMesh(self, fileName = None):
        fileName = 'data/%s%s'%(self.cfgParams.objName, '.ply' if self.nDim == 2 else '.mesh') if fileName is None else fileName

        if self.nDim == 2:
            writePlyFile(fileName, pad2Dto3D(self.gVertz), self.cells)
        else:
            writeMeshFile(fileName, self.gVertz, self.cells)
        if hasattr(self, 'vmStrezz'):
            np.savetxt(fileName.split('.')[0]+'.vms', self.vmStrezz)
        
if sys.platform == 'win32': # some TVTK issue on linux?

    class CubeVisual(HasTraits):

        p = Range(0, 1000, 0)
        scene = Instance(MlabSceneModel, ())
        meshPlot = Instance(PipelineBase)

        view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene), show_label=False),
                    Group('p'), resizable=True)

        def showPlot(self, cObj):
            self.cObj = cObj
            self.eTris = toEdgeTris(facesToEdges(self.cObj.ts) if self.cObj.nDim == 2 else tetsToEdges(self.cObj.ts))
            self.configure_traits()        


        @on_trait_change('p, scene.activated')
        def update_plot(self):
            p = min(len(self.cObj.cVertss)-1, self.p)

            cVerts = pad2Dto3D(self.cObj.cVertss[p]) if self.cObj.nDim == 2 else self.cObj.cVertss[p]
            rStress = innerNxM(innerNxM(innerNxM(self.cObj.pStress, [self.cObj.MinitRot]), self.cObj.Rs[p]), [self.cObj.MinitRot.T])
            R = np.vstack(rStress)
            Rdirs = pad2Dto3D(R) if self.cObj.nDim == 2 else R
            Rdirs *= self.cObj.pStressE.reshape(-1,1) ** 0.5

            heatScals = self.cObj.ctss[p] if hasattr(self.cObj, 'ctss') else computeMinEulerAngles(np.eye(3), rStress)
            M = self.cObj.MinitRot if self.cObj.isRestored else np.eye(self.cObj.nDim)
            if self.cObj.nDim == 2:
                vMin = np.dot(cVerts[:,:-1], M.T).min(axis=0)
                vMax = np.dot(cVerts[:,:-1], M.T).max(axis=0)
                boxVerts = (cubeVerts[:,:-1]/2 + 0.5) * (vMax-vMin) + vMin   
                boxVerts = pad2Dto3D(np.dot(boxVerts, M))
            else:
                vMin = np.dot(cVerts, M.T).min(axis=0)
                vMax = np.dot(cVerts, M.T).max(axis=0)
                boxVerts = (cubeVerts/2 + 0.5) * (vMax-vMin) + vMin
                boxVerts = np.dot(boxVerts, M)
            bx,by,bz = boxVerts.T

            if not hasattr(self, 'vPlot'):

                x,y,z = pad2Dto3D(self.cObj.verts).T if self.cObj.nDim == 2 else self.cObj.verts.T
                self.vPlot = mlab.triangular_mesh(x, y, z, self.eTris, representation = 'mesh', color = (1,1,1), tube_radius = self.cObj.sf/5)

                if hasattr(self.cObj, 'aIdxs'):
                    s = np.ones(len(self.cObj.aIdxs))
                    self.sPlot = mlab.quiver3d(x[self.cObj.aIdxs], y[self.cObj.aIdxs], z[self.cObj.aIdxs], s, s, s, color=(1,1,1), scale_factor = self.cObj.sf*2, mode='sphere')
                    self.sPlot.glyph.color_mode = 'color_by_scalar'
                    self.sPlot.glyph.glyph_source.glyph_position = 'center'

                x,y,z = cVerts.T
                self.uPlot = mlab.triangular_mesh(x, y, z, self.eTris, representation = 'mesh', color = (0,0,0), tube_radius = self.cObj.sf/5)
                self.bPlot = mlab.triangular_mesh(bx, by, bz, toEdgeTris(facesToEdges(sixCubeFaces)), representation = 'mesh', color = (0.75,0.75,0.75), tube_radius = self.cObj.sf*0.5)

                self.hPlot = mlab.quiver3d(x, y, z, x*0+1, y*0+1, z*0+1, scalars = heatScals, mode='sphere', scale_factor = self.cObj.sf*5)
                self.hPlot.glyph.color_mode = 'color_by_scalar'
                self.hPlot.glyph.glyph_source.glyph_position = 'center'

                x,y,z = np.repeat(cVerts, self.cObj.nDim, axis=0).T
                u,v,w = Rdirs.T
                scals = np.tile(np.arange(self.cObj.nDim)[::-1], self.cObj.nVerts)
                self.rPlot = mlab.quiver3d(x, y, z, u, v, w, scalars = scals)
                self.rPlot.glyph.color_mode = 'color_by_scalar'
                self.rPlot.glyph.glyph_source.glyph_source.glyph_type = 'dash'
                self.rPlot.glyph.glyph_source.glyph_position = 'center'

                cntr = np.repeat([self.cObj.centerInit], 3, axis=0)
                x,y,z = pad2Dto3D(cntr).T if self.cObj.nDim == 2 else cntr.T
                u,v,w = mat2Dto3D(self.cObj.MinitRot).T if self.cObj.nDim == 2 else self.cObj.MinitRot.T
                self.mPlot = mlab.quiver3d(x, y, z, u, v, w, scalars = [3,2,1])
                self.mPlot.glyph.color_mode = 'color_by_scalar'
                self.mPlot.glyph.glyph_source.glyph_source.glyph_type = 'dash'
                self.mPlot.glyph.glyph_source.glyph_position = 'center'

            else:
                self.uPlot.mlab_source.points = cVerts
                self.bPlot.mlab_source.points = boxVerts
                self.rPlot.mlab_source.points = np.repeat(cVerts, self.cObj.nDim, axis=0)
                self.rPlot.mlab_source.vectors = Rdirs
                self.hPlot.mlab_source.points = cVerts
                self.hPlot.mlab_source.scalars = heatScals
                
                if hasattr(self.cObj, 'aIdxs'):
                    self.sPlot.mlab_source.points = cVerts[self.cObj.aIdxs]                

if __name__ == "__main__":

    # create own config
    cfgParams = ConfigParams(objName = 'femur',
                             pLambda = 2,
                             lambdaRamp = 100,
                             autoScale = True,
                             optiAlign = True,
                             useNormals = True,
                             stressWeighted = False,
                             useConstraints = False,
                             maxIters = 1000,
                             stopDelta = 0.0001,
                             gridRes = 14,
                             finalLayer = False,
                             useProjection = False,
                             useTransform = True,
                             hullTolerance = False,
                             hullSmoothing = True)

    # or load an existing one
    #cfgParams = ConfigParams.fromFile('data/bar2D.cfg')
    #cfgParams = ConfigParams.fromFile('data/bunny2D.cfg')
    #cfgParams = ConfigParams.fromFile('data/spot2D.cfg')
    
    #cfgParams = ConfigParams.fromFile('data/cube.cfg')
    cfgParams = ConfigParams.fromFile('data/femur.cfg')
    #cfgParams = ConfigParams.fromFile('data/fertility.cfg')
    #cfgParams = ConfigParams.fromFile('data/kitten.cfg')
    #cfgParams = ConfigParams.fromFile('data/spot.cfg')
    #cfgParams = ConfigParams.fromFile('data/venus.cfg')

    #cfgParams = ConfigParams.fromFile('data/JEB.cfg')    
    #cfgParams = ConfigParams.fromFile('data/buddhaDown.cfg')
    #cfgParams = ConfigParams.fromFile('data/buddhaBack.cfg')
    #cfgParams = ConfigParams.fromFile('data/buddhaBelly.cfg')

    # load the stress file
    cObj = CubeObject(cfgParams)

    # either
    if True: # compute field deformation and save it
        cObj.optimize()
        cObj.saveState()
    else: # or load it from previous run
        cObj.loadState()

    # compute inner grid and hull
    cObj.fillGrid()
    #cObj.exportHullsToObj()
    cObj.padHull()
    cObj.restoreInitShape()
    
    cObj.evalAlignment()
    SJs = cObj.evalScaledJacobians()
    print('#hv:\t%d, #he:\t%d'%(len(cObj.gVertz), len(cObj.cells)))
    print('SJs:', SJs.min(), SJs.max(), SJs.mean())

    # save results as mesh
    cObj.exportCellsToMesh()

    # show results in mayavi
    #cObj.show(True, False)
    cObj.show(False, False)

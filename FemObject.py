from femUtil import *

# FEM constants
youngs_modulus = 1000.0
poisson_ratio = 0.3

pLambda, pMu = lame_parameters(youngs_modulus, poisson_ratio)
C = linear_stress(pLambda, pMu)

class FemObject:

    def __init__(self, inputFile = None, loadResult = False):
        if inputFile is None:
            return

        self.materialized = False

        self.inputFile = inputFile

        self.fixFuns = []
        self.flxFuns = []
        self.frcVecs = []

        self.funData = []

        if '.stress' in inputFile:
            self.fromStressFile(inputFile)
            return
        
        with open(self.inputFile, 'r') as fh:
            for line in fh.readlines():
                line = line.strip().split(' ')
                if '#' in line:
                    continue
                
                if line[0] == 'file':
                    self.meshFile = os.path.dirname(self.inputFile) + '/' + line[1]
                    continue
                if line[0] == 'type':
                    self.mType = line[1]
                    continue
                    
                if line[1] == 'd':
                    d = int(line[2])
                    
                    if '-' in line[2]:
                        fData = (-d, float(line[-1]))
                        f = lambda vs, fIdx, tol = eps: vs[:,self.funData[fIdx][0]] <= (self.funData[fIdx][1] + tol)
                    else:
                        fData = (d, float(line[-1]))
                        f = lambda vs, fIdx, tol = eps: vs[:,self.funData[fIdx][0]] >= (self.funData[fIdx][1] - tol)
                        
                elif line[1] == 'r':
                    fData = (np.float32(line[2:-1]), float(line[-1]))
                    f = lambda vs, fIdx, tol = eps: norm(vs - self.funData[fIdx][0]) < self.funData[fIdx][1] + tol
    
                if line[0] == 'fix':
                    self.fixFuns.append(f)
                    self.funData.append(fData)
                elif line[0] == 'flx':
                    self.flxFuns.append(f)
                    self.funData.append(fData)
                elif line[0] == 'vec':
                    self.frcVecs.append(np.float32(line[1:]))
            self.frcVecs = np.float32(self.frcVecs)

        self.nDim = 2 if self.mType in ['tri', 'quad'] else 3

        if loadResult:
            if self.nDim == 2:
                self.meshFile = self.inputFile.replace('.frc', '.ply')
                self.mType = 'quad'
            if self.nDim == 3:
                self.meshFile = self.inputFile.replace('.frc', '.mesh')
                self.mType = 'hex'
            if not os.path.exists(self.meshFile):
                print('Result does not yet exist.')

            vmsPath = self.meshFile.split('.')[0] + '.vms'
            if os.path.exists(self.meshFile.split('.')[0] + '.vms'):
                self.vmStress = np.loadtxt(vmsPath)           
                
        self.mTypeInit = self.mType
        if self.mType in ['tri', 'quad']:
            if '.obj' in self.meshFile:
                vs, self.ts = loadObjFile(self.meshFile)
            else:
                vs, self.ts = loadPlyFile(self.meshFile)
            self.verts = vs[:,:-1]
            if self.mType == 'tri':
                self.meshInit = fem.MeshTri(self.verts.T, self.ts.T)
                self.eType = fem.ElementTriP1()
            elif self.mType == 'quad':
                self.meshInit = fem.MeshQuad(self.verts.T, self.ts.T)
                self.eType = fem.ElementQuad1()
        elif self.mType == 'tet':
            self.meshInit = fem.MeshTet.load(self.meshFile)
            self.eType = fem.ElementTetP1()
        elif self.mType == 'hex':
            self.meshInit = fem.MeshHex.load(self.meshFile)
            self.eType = fem.ElementHex1()           

        self.setupFEM()

    def fromStressFile(self, filePath):
        self.verts, self.ts, self.vIdxsFlx, self.forceVecs, self.vIdxsFix, self.vmStress, self.pStress, self.pStressE, self.sMats = loadStressFile(filePath)
        self.nDim = self.verts.shape[1]
        
        if self.nDim == 2:
            if self.ts.shape[1] == 3:
                self.mType = 'tri'
                self.meshInit = fem.MeshTri(self.verts.T, self.ts.T)
                self.eType = fem.ElementTriP1()
            else:
                self.mType = 'quad'
                self.meshInit = fem.MeshQuad(self.verts.T, self.ts.T)
                self.eType = fem.ElementQuad1()
        else:
            if self.ts.shape[1] == 4:
                self.mType = 'tet'
                self.meshInit = fem.MeshTet(self.verts.T, self.ts.T)
                self.eType = fem.ElementTetP1()
            else:
                self.mType = 'hex'
                self.meshInit = fem.MeshHex(self.verts.T, self.ts.T)
                self.eType = fem.ElementHex1()

        self.mTypeInit = self.mType
        self.setupFEM()

        fVecs = np.zeros((self.nVerts, self.nDim), np.float32)
        fVecs[self.vIdxsFlx] = self.forceVecs
        self.f = fVecs.ravel()

    def setupFEM(self):
        self.sf = norm(self.meshInit.doflocs.max(axis=1)-self.meshInit.doflocs.min(axis=1)) / 1000
        
        self.nVerts = len(self.meshInit.doflocs.T)
        self.nElems = len(self.meshInit.t.T)
        print("#v % 6d, #e % 6d"%(self.nVerts, self.nElems))

        vs = self.meshInit.doflocs.T
        es = self.meshInit.edges.T if self.nDim == 3 else facesToEdges(self.meshInit.t.T)
        self.mel = norm(vs[es[:,0]] - vs[es[:,1]]).mean()
            
        self.eVector = fem.ElementVector(self.eType)
        self.basis = fem.Basis(self.meshInit, self.eVector, fem.MappingIsoparametric(self.meshInit, self.eType), 0)
        self.K = fem.asm(linear_elasticity(pLambda, pMu), self.basis)    

        if self.nDim == 2:
            tris = self.ts if self.mType == 'tri' else facesToTris(self.ts)
            self.volInit = computeTriangleAreas(self.verts[tris], False).sum()
        else:
            verts = self.meshInit.doflocs.T
            if self.mType == 'tet':
                tets = self.meshInit.t.T
            else:
                hexas = self.meshInit.t.T
                tets = np.vstack([hexaToTetras(hexa) for hexa in hexOrderDg2BT(hexas)])
            self.volInit = computeTetraVolumes(verts[tets], False).sum()
        self.vol = self.volInit
            
        self.setupBoundaryMasks()
        self.setupBoundaryWeights()

    def setupBoundaryWeights(self):
        self.bWeights = np.zeros(self.nVerts, np.float32)

        bElements = self.meshInit.facets.T[self.meshInit.boundary_facets()]
        bElements = bElements[np.any(self.boundaryVertMask[bElements], axis=1)]
        if self.nDim == 2:
            bElementWeights = norm(self.verts[bElements[:,0]] - self.verts[bElements[:,1]]) / 2
        elif self.mType == 'tet':            
            bElementWeights = computeTriangleAreas(self.meshInit.doflocs.T[bElements]) / 3           
        elif self.mType == 'hex':
            bElementWeights = computeTriangleAreas(self.meshInit.doflocs.T[tetrasToFaces(bElements)]).reshape(-1,4).sum(axis=1) / 8
            
        for eIdx, bElement in tqdm(enumerate(bElements), total = len(bElements), ascii=True, desc='boundaryWeights'):
            self.bWeights[bElement] += bElementWeights[eIdx]

    def materialize(self, alpha = 0.5, walled = False):
        self.materialized = [alpha, walled]
        if self.mType in ['tri', 'quad']:
            self.materializeTriQuad(alpha)
        elif self.mType == 'tet':
            self.materializeTet(alpha, walled)
        elif self.mType == 'hex':
            self.materializeHex(alpha, walled)

    def materializeOpti(self, alpha = 0.5, walled = False, weights = None):
        self.materialized = [alpha, walled]
        alpha = abs(alpha)

        ws = (self.vmStress/self.vmStress.max()) ** 2 if weights is None else weights

        if self.mType in ['tri', 'quad']:
            vertsOrig = self.verts
            facesOrig = self.ts
            edgesOrig = facesToEdges(facesOrig)
            
            self.materializeTriQuad(alpha, False, False)

            taus = optimizeTaus(self.volInit * alpha, ws, vertsOrig, [edgesOrig, facesOrig], self.ts, self.mTypeInit)
            self.verts = materializeFaceVerts(taus, ws, vertsOrig, edgesOrig, facesOrig, self.mTypeInit)

        elif self.mType == 'tet':
            vertsOrig = self.meshInit.doflocs.T
            edgesOrig = self.meshInit.edges.T
            trisOrig = self.meshInit.facets.T
            tetsOrig = self.meshInit.t.T

            self.materializeTet(alpha, walled, False, False)

            taus = optimizeTaus(self.volInit * alpha, ws, vertsOrig, [edgesOrig, trisOrig, tetsOrig], self.ts, self.mTypeInit, walled)
            self.verts = materializeTetVerts(taus, ws, vertsOrig, edgesOrig, trisOrig, tetsOrig, self.mTypeInit, walled)

        elif self.mType == 'hex':
            vertsOrig = self.meshInit.doflocs.T
            edgesOrig = self.meshInit.edges.T
            quadsOrig = self.meshInit.facets.T
            hexasOrig = self.meshInit.t.T

            self.materializeHex(alpha, walled, False, False)

            taus = optimizeTaus(self.volInit * alpha, ws, vertsOrig, [edgesOrig, quadsOrig, hexasOrig], self.ts, self.mTypeInit, walled)
            self.verts = materializeHexVerts(taus, ws, vertsOrig, edgesOrig, quadsOrig, hexasOrig, self.mTypeInit, walled)

        self.initMaterializedFEM()

    def initMaterializedFEM(self):
        if self.mTypeInit in ['tri', 'quad', 'cell']:
            self.mType = 'quad'
            self.meshInit = fem.MeshQuad(self.verts.T, self.ts.T)
            self.eType = fem.ElementQuad1()

            self.vol = computeTriangleAreas(self.verts[facesToTris(self.ts)], False).sum()
        else:
            self.mType = 'hex'
            self.meshInit = fem.MeshHex(self.verts.T, self.ts.T)
            self.eType = fem.ElementHex1()

            hexaTets = np.vstack([hexaToTetras(hexa) for hexa in hexOrderDg2BT(self.ts)])
            self.vol = computeTetraVolumes(self.verts[hexaTets], False).sum()

        self.SJs = self.evalScaledJacobians()
        if self.SJs.max() < 0:
            self.ts = self.ts[:,::-1]
            self.SJs *= -1

        self.nVerts = len(self.meshInit.doflocs.T)
        self.nElems = len(self.meshInit.t.T)
        print('nV: %d, nE: %d'%(self.nVerts, self.nElems))

        st = time()
        self.eVector = fem.ElementVector(self.eType)
        self.basis = fem.Basis(self.meshInit, self.eVector, fem.MappingIsoparametric(self.meshInit, self.eType), 0)
        print('basis', time() - st)

        st = time()
        self.K = fem.asm(linear_elasticity(pLambda, pMu), self.basis)
        print('K', time() - st)

        st = time()
        self.setupBoundaryWeights()
        print('weights', time() - st)        

    def materializeTriQuad(self, alpha = 0.5, withVerts = True, withFEM = True):
           
        tris = self.ts if self.mType == 'tri' else facesToTris(self.ts)
        self.volInit = computeTriangleAreas(self.verts[tris], False).sum()

        verts = self.meshInit.doflocs.T
        faces = self.cs if self.mTypeInit == 'cell' else self.ts
        edges = facesToEdges(faces)

        if withVerts:
            if self.mType == 'tri':
                tau = (1 - np.sqrt(1 - alpha)) / 3
            if self.mType == 'quad':
                tau = (1 - np.sqrt(1 - alpha)) / 2
            self.verts = materializeFaceVerts(None, tau, verts, edges, faces, self.mType)

        boundaryVertMask = flatten([[self.boundaryVertMask[edge].all()] * 2 for edge in edges])
        es = np.repeat(edges, 2, axis=0)
        es[1::2] = edges[:,::-1]
        edgeHashs = cantorPiV(es, False)

        quads = []
        idxOffset = 0
        for fIdx, face in enumerate(faces):
            m = len(face)
            for vIdx in range(m):
                heCL = cantorPiO(face[vIdx], face[(vIdx+1)%m])
                heCR = cantorPiO(face[vIdx], face[(vIdx-1)%m])
                vCLidx = np.where(edgeHashs == heCL)[0][0] + len(verts)
                vCRidx = np.where(edgeHashs == heCR)[0][0] + len(verts)
                vCIidx = idxOffset + vIdx + len(verts) + len(edges)*2
                quads.append([face[vIdx], vCLidx, vCIidx, vCRidx])  # corner

                heLC = cantorPiO(face[(vIdx+1)%m], face[vIdx])
                vLCidx = np.where(edgeHashs == heLC)[0][0] + len(verts)
                vLIidx = idxOffset + (vIdx+1)%m + len(verts) + len(edges)*2
                quads.append([vCLidx, vLCidx, vLIidx, vCIidx])      # edge
            idxOffset += m

            boundaryVertMask += [False] * m

        self.ts = np.vstack(quads)
        self.boundaryVertMask = np.concatenate([self.boundaryVertMask, boundaryVertMask])

        if withFEM:
            self.initMaterializedFEM()

    def materializeTet(self, alpha = 0.5, walled = False, withVerts = True, withFEM = True):

        def tetToEdges(tet):
            return np.int64(np.transpose([tet[[0,0,0,1,2,3]], tet[[1,2,3,2,3,1]]]))

        def tetToFaces(tet):
            return np.int64(tet[[[0,2,3], [0,3,1], [0,1,2], [1,3,2]]])

        verts = self.meshInit.doflocs.T
        edges = self.meshInit.edges.T
        faces = self.meshInit.facets.T
        tets = self.meshInit.t.T

        self.volInit = computeTetraVolumes(verts[tets], False).sum()
    
        if walled:
            tau = (1 - (1 - alpha)**(1/3)) / 4
            numEdgeVerts = 3
            numFaceVerts = 7
            numCellVerts = 14
        else:
            taus = {0.25: 0.0943680425, 0.5: 0.1448414, 0.75: 0.195888335}
            if alpha in taus.keys(): # exact values
                tau = taus[alpha]
            else: # approximation with magic numbers
                p = np.sqrt(42)/5 + 2 # 3.29615807
                q = -np.pi/2 - 1 - 2**(1/3)/2 + np.sqrt(6)/2 + 3*np.sqrt(2)/2 # 0.14528211
                tau = np.arcsin(alpha*2-1)/(p*np.pi)+q
            numEdgeVerts = 2
            numFaceVerts = 3
            numCellVerts = 4

        if withVerts:
            self.verts = materializeTetVerts(None, tau, verts, edges, faces, tets, self.mType, walled)

        boundaryVertMask = []
        
        eHashs = cantorPiV(np.int64(edges))
        ePos = {}
        for i, (e, eHash) in enumerate(zip(edges, eHashs)):
            for j, vIdx in enumerate(e):
                ePos[(eHash, vIdx)] = numEdgeVerts*i + j
            if walled:
                ePos[(eHash, None)] = numEdgeVerts*i + 2
            boundaryVertMask += [self.boundaryVertMask[e].all()] * numEdgeVerts
        
        fHashs = cantorPiKV(np.int64(faces))
        fPos = {}
        for i, (fc, fHash) in enumerate(zip(faces, fHashs)):
            for j, vIdx in enumerate(fc):
                fPos[(fHash, (vIdx+1))] = numFaceVerts*i + j
                if walled:
                    fPos[(fHash, -(vIdx+1))] = numFaceVerts*i + j + 3
            if walled:
                fPos[(fHash, None)] = numFaceVerts*i + 6
            boundaryVertMask += [self.boundaryVertMask[fc].all()] * numFaceVerts

        newHexas = []
        for tIdx, vIdxs in tqdm(enumerate(tets), total = tets.shape[0], ascii=True, desc='materializing'):
            tEdges = tetToEdges(vIdxs)
            tEdgesHashs = cantorPiV(tEdges)
            eIdxs = []
            for edge, eHash in zip(tEdges, tEdgesHashs):
                eIdxs += [ePos[(eHash, vIdx)] for vIdx in edge]
                if walled:
                    eIdxs += [ePos[(eHash, None)]]
            eIdxs = np.int32(eIdxs) + len(verts)

            tFaces = tetToFaces(vIdxs)
            tFacesHashs = cantorPiKV(tFaces)

            fIdxs = []
            for face, fHash in zip(tFaces, tFacesHashs):
                fIdxs += [fPos[(fHash, (vIdx+1))] for vIdx in face]
                if walled:
                    fIdxs += [fPos[(fHash, -(vIdx+1))] for vIdx in face]
                    fIdxs += [fPos[(fHash, None)]]
            fIdxs = np.int32(fIdxs) + len(verts) + len(edges) * numEdgeVerts
            
            cIdxs = np.arange(numCellVerts) + tIdx*numCellVerts + len(verts) + len(edges)*numEdgeVerts + len(faces)*numFaceVerts

            # corners
            if walled:
                newHexas += [[vIdxs[0], eIdxs[0], eIdxs[6], eIdxs[3], fIdxs[7], fIdxs[14], fIdxs[0], cIdxs[0]]]
                newHexas += [[eIdxs[1], vIdxs[1], fIdxs[9], fIdxs[15], eIdxs[16], eIdxs[9], cIdxs[1], fIdxs[21]]]
                newHexas += [[eIdxs[4], fIdxs[16], fIdxs[1], vIdxs[2], cIdxs[2], eIdxs[10], eIdxs[12], fIdxs[23]]]
                newHexas += [[fIdxs[2], cIdxs[3], eIdxs[7], eIdxs[13], fIdxs[8], fIdxs[22], vIdxs[3], eIdxs[15]]]
            else:
                newHexas += [[vIdxs[0], eIdxs[0], eIdxs[4], eIdxs[2], fIdxs[3], fIdxs[6], fIdxs[0], cIdxs[0]]]
                newHexas += [[eIdxs[1], vIdxs[1], fIdxs[5], fIdxs[7], eIdxs[11], eIdxs[6], cIdxs[1], fIdxs[9]]]
                newHexas += [[eIdxs[3], fIdxs[8], fIdxs[1], vIdxs[2], cIdxs[2], eIdxs[7], eIdxs[8], fIdxs[11]]]
                newHexas += [[fIdxs[2], cIdxs[3], eIdxs[5], eIdxs[9], fIdxs[4], fIdxs[10], vIdxs[3], eIdxs[10]]]

            # edges
            if walled:
                newHexas += [[eIdxs[0], eIdxs[2], fIdxs[7], fIdxs[14], fIdxs[11], fIdxs[19], cIdxs[0], cIdxs[4]]]
                newHexas += [[eIdxs[2], eIdxs[1], fIdxs[11], fIdxs[19], fIdxs[9], fIdxs[15], cIdxs[4], cIdxs[1]]]
                
                newHexas += [[eIdxs[3], fIdxs[14], fIdxs[0], eIdxs[5], cIdxs[0], fIdxs[18], fIdxs[5], cIdxs[6]]]
                newHexas += [[eIdxs[5], fIdxs[18], fIdxs[5], eIdxs[4], cIdxs[6], fIdxs[16], fIdxs[1], cIdxs[2]]]

                newHexas += [[eIdxs[6], fIdxs[7], eIdxs[8], fIdxs[0], fIdxs[12], cIdxs[0], fIdxs[4], cIdxs[7]]]
                newHexas += [[eIdxs[8], fIdxs[12], eIdxs[7], fIdxs[4], fIdxs[8], cIdxs[7], fIdxs[2], cIdxs[3]]]

                newHexas += [[fIdxs[1], cIdxs[2], fIdxs[3], eIdxs[12], cIdxs[9], fIdxs[23], eIdxs[14], fIdxs[24]]]
                newHexas += [[fIdxs[3], cIdxs[9], fIdxs[2], eIdxs[14], cIdxs[3], fIdxs[24], eIdxs[13], fIdxs[22]]]
                
                newHexas += [[fIdxs[16], fIdxs[17], cIdxs[2], eIdxs[10], cIdxs[5], eIdxs[11], fIdxs[23], fIdxs[25]]]
                newHexas += [[fIdxs[17], fIdxs[15], cIdxs[5], eIdxs[11], cIdxs[1], eIdxs[9], fIdxs[25], fIdxs[21]]]
                
                newHexas += [[fIdxs[9], eIdxs[16], fIdxs[10], cIdxs[1], eIdxs[17], fIdxs[21], cIdxs[8], fIdxs[26]]]
                newHexas += [[fIdxs[10], eIdxs[17], fIdxs[8], cIdxs[8], eIdxs[15], fIdxs[26], cIdxs[3], fIdxs[22]]]
            else:
                newHexas += [[eIdxs[0], eIdxs[1], fIdxs[3], fIdxs[6], fIdxs[5], fIdxs[7], cIdxs[0], cIdxs[1]]]
                newHexas += [[eIdxs[2], fIdxs[6], fIdxs[0], eIdxs[3], cIdxs[0], fIdxs[8], fIdxs[1], cIdxs[2]]]
                newHexas += [[eIdxs[4], fIdxs[3], eIdxs[5], fIdxs[0], fIdxs[4], cIdxs[0], fIdxs[2], cIdxs[3]]]
                newHexas += [[fIdxs[1], cIdxs[2], fIdxs[2], eIdxs[8], cIdxs[3], fIdxs[11], eIdxs[9], fIdxs[10]]]
                newHexas += [[fIdxs[8], fIdxs[7], cIdxs[2], eIdxs[7], cIdxs[1], eIdxs[6], fIdxs[11], fIdxs[9]]]
                newHexas += [[fIdxs[5], eIdxs[11], fIdxs[4], cIdxs[1], eIdxs[10], fIdxs[9], cIdxs[3], fIdxs[10]]]
            
            # faces
            if walled:
                newHexas += [[fIdxs[6], cIdxs[11], fIdxs[5], fIdxs[4], cIdxs[6], cIdxs[7], fIdxs[0], cIdxs[0]]]
                newHexas += [[fIdxs[6], cIdxs[11], fIdxs[4], fIdxs[3], cIdxs[7], cIdxs[9], fIdxs[2], cIdxs[3]]]
                newHexas += [[fIdxs[6], cIdxs[11], fIdxs[3], fIdxs[5], cIdxs[9], cIdxs[6], fIdxs[1], cIdxs[2]]]

                newHexas += [[fIdxs[13], cIdxs[12], fIdxs[12], fIdxs[11], cIdxs[7], cIdxs[4], fIdxs[7], cIdxs[0]]]
                newHexas += [[fIdxs[13], cIdxs[12], fIdxs[11], fIdxs[10], cIdxs[4], cIdxs[8], fIdxs[9], cIdxs[1]]]
                newHexas += [[fIdxs[13], cIdxs[12], fIdxs[10], fIdxs[12], cIdxs[8], cIdxs[7], fIdxs[8], cIdxs[3]]]

                newHexas += [[fIdxs[20], cIdxs[13], fIdxs[19], fIdxs[18], cIdxs[4], cIdxs[6], fIdxs[14], cIdxs[0]]]
                newHexas += [[fIdxs[20], cIdxs[13], fIdxs[18], fIdxs[17], cIdxs[6], cIdxs[5], fIdxs[16], cIdxs[2]]]
                newHexas += [[fIdxs[20], cIdxs[13], fIdxs[17], fIdxs[19], cIdxs[5], cIdxs[4], fIdxs[15], cIdxs[1]]]

                newHexas += [[fIdxs[27], cIdxs[10], fIdxs[26], fIdxs[25], cIdxs[8], cIdxs[5], fIdxs[21], cIdxs[1]]]
                newHexas += [[fIdxs[27], cIdxs[10], fIdxs[25], fIdxs[24], cIdxs[5], cIdxs[9], fIdxs[23], cIdxs[2]]]
                newHexas += [[fIdxs[27], cIdxs[10], fIdxs[24], fIdxs[26], cIdxs[9], cIdxs[8], fIdxs[22], cIdxs[3]]]

            boundaryVertMask += [False] * numCellVerts

        self.ts = np.int32(newHexas)[:,::-1]
        self.boundaryVertMask = np.concatenate([self.boundaryVertMask, boundaryVertMask])

        if withFEM:
            self.initMaterializedFEM()

    def materializeHex(self, alpha = 0.5, walled = False, withVerts = True, withFEM = True):

        def hexaToEdges(hexa):
            return np.int64(np.transpose([hexa[[0,0,0,1,2,1,3,2,3,4,5,6]], hexa[[1,2,3,4,4,5,5,6,6,7,7,7]]]))

        def hexaToFaces(hexa):
            return np.int64(hexa[[[0,2,6,3], [0,3,5,1], [0,1,4,2], [6,7,5,3], [2,4,7,6], [1,5,7,4]]])

        verts = self.meshInit.doflocs.T
        edges = self.meshInit.edges.T
        faces = self.meshInit.facets.T
        hexas = self.meshInit.t.T     

        hexaTets = np.vstack([hexaToTetras(hexa) for hexa in hexOrderDg2BT(hexas)])
        self.volInit = computeTetraVolumes(verts[hexaTets], False).sum()

        if withVerts:
            if walled:
                tau = (1 - (1 - alpha)**(1/3)) / 2
            else:
                tau = np.arcsin(alpha*2-1)/(2*np.pi)+1/4
            self.verts = materializeHexVerts(None, tau, verts, edges, faces, hexas, self.mType, walled)
        
        boundaryVertMask = []

        eHashs = cantorPiV(np.int64(edges))
        ePos = {}
        for i, (e, eHash) in enumerate(zip(edges, eHashs)):
            for j, vIdx in enumerate(e):
                ePos[(eHash, vIdx)] = 2*i + j
            boundaryVertMask += [self.boundaryVertMask[e].all()] * 2
        
        fHashs = cantorPiKV(np.int64(faces))
        fPos = {}
        for i, (fc, fHash) in enumerate(zip(faces, fHashs)):
            for j, vIdx in enumerate(fc):
                fPos[(fHash, vIdx)] = 4*i + j
            boundaryVertMask += [self.boundaryVertMask[fc].all()] * 4

        newHexas = []
        for hIdx, vIdxs in tqdm(enumerate(hexas), total = hexas.shape[0], ascii=True, desc='materializing'):
            hEdges = hexaToEdges(vIdxs)
            hEdgesHashs = cantorPiV(hEdges)
            eIdxs = []
            for edge, eHash in zip(hEdges, hEdgesHashs):
                eIdxs += [ePos[(eHash, vIdx)] for vIdx in edge]
            eIdxs = np.int32(eIdxs) + len(verts)

            hFaces = hexaToFaces(vIdxs)
            hFacesHashs = cantorPiKV(hFaces)
            fIdxs = []
            for face, fHash in zip(hFaces, hFacesHashs):
                fIdxs += [fPos[(fHash, vIdx)] for vIdx in face]
            fIdxs = np.int32(fIdxs) + len(verts) + len(edges)*2
            
            cIdxs = np.arange(8) + hIdx*8 + len(verts) + len(edges)*2 + len(faces)*4

            # corners
            newHexas += [[vIdxs[0], eIdxs[0], eIdxs[2], eIdxs[4], fIdxs[8], fIdxs[4], fIdxs[0], cIdxs[0]]]
            newHexas += [[eIdxs[1], vIdxs[1], fIdxs[9], fIdxs[7], eIdxs[6], eIdxs[10], cIdxs[1], fIdxs[20]]]
            newHexas += [[eIdxs[3], fIdxs[11], vIdxs[2], fIdxs[1], eIdxs[8], cIdxs[2], eIdxs[14], fIdxs[16]]]
            newHexas += [[eIdxs[5], fIdxs[5], fIdxs[3], vIdxs[3], cIdxs[3], eIdxs[12], eIdxs[16], fIdxs[15]]]

            newHexas += [[fIdxs[10], eIdxs[7], eIdxs[9], cIdxs[4], vIdxs[4], fIdxs[23], fIdxs[17], eIdxs[18]]]
            newHexas += [[fIdxs[6], eIdxs[11], cIdxs[5], eIdxs[13], fIdxs[21], vIdxs[5], fIdxs[14], eIdxs[20]]]
            newHexas += [[fIdxs[2], cIdxs[6], eIdxs[15], eIdxs[17], fIdxs[19], fIdxs[12], vIdxs[6], eIdxs[22]]]
            newHexas += [[cIdxs[7], fIdxs[22], fIdxs[18], fIdxs[13], eIdxs[19], eIdxs[21], eIdxs[23], vIdxs[7]]]

            # edges
            newHexas += [[eIdxs[4], fIdxs[4], fIdxs[0], eIdxs[5], cIdxs[0], fIdxs[5], fIdxs[3], cIdxs[3]]]
            newHexas += [[eIdxs[2], fIdxs[8], eIdxs[3], fIdxs[0], fIdxs[11], cIdxs[0], fIdxs[1], cIdxs[2]]]
            newHexas += [[fIdxs[1], cIdxs[2], eIdxs[14], fIdxs[2], fIdxs[16], cIdxs[6], eIdxs[15], fIdxs[19]]]
            newHexas += [[fIdxs[3], cIdxs[3], fIdxs[2], eIdxs[16], cIdxs[6], fIdxs[15], eIdxs[17], fIdxs[12]]]

            newHexas += [[eIdxs[0], eIdxs[1], fIdxs[8], fIdxs[4], fIdxs[9], fIdxs[7], cIdxs[0], cIdxs[1]]]
            newHexas += [[fIdxs[11], fIdxs[10], eIdxs[8], cIdxs[2], eIdxs[9], cIdxs[4], fIdxs[16], fIdxs[17]]]
            newHexas += [[fIdxs[5], fIdxs[6], cIdxs[3], eIdxs[12], cIdxs[5], eIdxs[13], fIdxs[15], fIdxs[14]]]
            newHexas += [[cIdxs[6], cIdxs[7], fIdxs[19], fIdxs[12], fIdxs[18], fIdxs[13], eIdxs[22], eIdxs[23]]]

            newHexas += [[fIdxs[7], eIdxs[10], cIdxs[1], fIdxs[6], fIdxs[20], eIdxs[11], cIdxs[5], fIdxs[21]]]
            newHexas += [[fIdxs[9], eIdxs[6], fIdxs[10], cIdxs[1], eIdxs[7], fIdxs[20], cIdxs[4], fIdxs[23]]]
            newHexas += [[cIdxs[4], fIdxs[23], fIdxs[17], cIdxs[7], eIdxs[18], fIdxs[22], fIdxs[18], eIdxs[19]]]
            newHexas += [[cIdxs[5], fIdxs[21], cIdxs[7], fIdxs[14], fIdxs[22], eIdxs[20], fIdxs[13], eIdxs[21]]]

            # faces
            if walled:
                newHexas += [[fIdxs[0], cIdxs[0], fIdxs[1], fIdxs[3], cIdxs[2], cIdxs[3], fIdxs[2], cIdxs[6]]]
                newHexas += [[fIdxs[8], fIdxs[9], fIdxs[11], cIdxs[0], fIdxs[10], cIdxs[1], cIdxs[2], cIdxs[4]]]
                newHexas += [[fIdxs[4], fIdxs[7], cIdxs[0], fIdxs[5], cIdxs[1], fIdxs[6], cIdxs[3], cIdxs[5]]]
                newHexas += [[cIdxs[2], cIdxs[4], fIdxs[16], cIdxs[6], fIdxs[17], cIdxs[7], fIdxs[19], fIdxs[18]]]
                newHexas += [[cIdxs[3], cIdxs[5], cIdxs[6], fIdxs[15], cIdxs[7], fIdxs[14], fIdxs[12], fIdxs[13]]]
                newHexas += [[cIdxs[1], fIdxs[20], cIdxs[4], cIdxs[5], fIdxs[23], fIdxs[21], cIdxs[7], fIdxs[22]]]

            # inner block
            #newHexas += [[cIdxs[0], cIdxs[1], cIdxs[2], cIdxs[3], cIdxs[4], cIdxs[5], cIdxs[6], cIdxs[7]]]

            boundaryVertMask += [False] * 8

        self.ts = np.int32(newHexas)
        self.boundaryVertMask = np.concatenate([self.boundaryVertMask, boundaryVertMask])

        if withFEM:
            self.initMaterializedFEM()

    def setupBoundaryMasks(self):
        self.boundaryVertMask = np.zeros(self.nVerts, dtype=np.bool_)
        if self.mType in ['tri', 'quad']:
            es = facesToEdges(self.ts if hasattr(self, 'ts') else self.meshInit.t.T, False)
            unq, inv, cnt = np.unique(cantorPiKV(es), return_inverse = True, return_counts = True)
            self.boundaryEdgeMask = cnt[inv] == 1
            self.boundaryVertMask[np.unique(es[self.boundaryEdgeMask].ravel())] = True
        if self.mType in ['hex', 'tet']:
            ts = self.ts if hasattr(self, 'ts') else self.meshInit.t.T
            fcs = tetrasToFaces(ts) if self.mType == 'tet' else hexasToFaces(ts)
            unq, inv, cnt = np.unique(cantorPiKV(fcs), return_inverse = True, return_counts = True)
            self.boundaryFaceMask = cnt[inv] == 1
            self.boundaryVertMask[self.meshInit.boundary_nodes()] = True

    def computeCompliance(self, complianceOnly = False):

        fVecs = np.zeros((self.nVerts, self.nDim), np.float32)
        if not hasattr(self, 'forceVecs'):
            vs = self.meshInit.doflocs.T
            
            funIdx = 0
            fixMsk = self.boundaryVertMask * 0
            for fixFun in self.fixFuns:
                msk = fixFun(vs, funIdx, self.mel/1000)
                fixMsk = np.bitwise_or(np.bitwise_and(msk, self.boundaryVertMask), fixMsk)
                funIdx += 1
            self.vIdxsFix = np.where(fixMsk)[0].tolist()

            flxMsk = self.boundaryVertMask * 0
            flxIdxs = []
            for fIdx, flxFun in enumerate(self.flxFuns):
                msk = flxFun(vs, funIdx, self.mel/1000)
                bMsk = np.bitwise_and(msk, self.boundaryVertMask)
                flxMsk = np.bitwise_or(bMsk, flxMsk)
                fvIdxs = np.where(bMsk)[0]
                flxIdxs.append(fvIdxs)

                w = self.bWeights[fvIdxs] #* 0 + 1
                fVecs[fvIdxs] = self.frcVecs[fIdx] * w.reshape(-1,1) / w.sum()
                
                funIdx += 1
            numVertsPerForce = np.int32(list(map(len, flxIdxs)))
            self.vIdxsFlx = np.where(flxMsk)[0].tolist()

        else:
            fVecs[self.vIdxsFlx] = self.forceVecs
            
        self.f = fVecs.ravel()
        
        print('#fix: %d, #flx: %d'%(len(self.vIdxsFix), len(self.vIdxsFlx)))

        if False:
            msk = np.ones((self.nVerts, self.nDim), np.bool_)
            msk[self.vIdxsFix] = False
            self.I = np.where(msk.ravel())[0]
            self.cnds = fem.condense(self.K, b = self.f, I = self.I)
        else: # same but a bit faster
            self.D = (np.tile(np.int32(self.vIdxsFix)*self.nDim,[self.nDim,1]).T + np.arange(self.nDim)).ravel()
            self.cnds = fem.condense(self.K, b = self.f, D = self.D)
        
        print('solving...')
        if pypardisoFound:
            self.u = fem.solve(*self.cnds, solver = solver_pypardiso)
        else:
            self.u = fem.solve(*self.cnds)
        print('done')

        self.compliance = np.dot(self.u, self.K.dot(self.u))/2
        self.dCols = norm(self.u.reshape(-1,self.nDim))

        if complianceOnly:
            return self.compliance

        # deformed mesh and stresses
        self.meshDeformed = self.meshInit.translated(self.u.reshape(-1,self.nDim).T)

        self.dg1 = self.basis.with_element(self.eType)
        self.u1 = self.basis.interpolate(self.u)

        C_sym_grad = C(sym_grad(self.u1))
        self.C = np.transpose(C_sym_grad, axes=[2,3,0,1])

        if pypardisoFound:
            s00 = dg1Project(self.dg1, C_sym_grad[0,0])
            s01 = dg1Project(self.dg1, C_sym_grad[0,1])
            s11 = dg1Project(self.dg1, C_sym_grad[1,1])
            if self.mType in ['tet', 'hex']:
                s02 = dg1Project(self.dg1, C_sym_grad[0,2])
                s12 = dg1Project(self.dg1, C_sym_grad[1,2])
                s22 = dg1Project(self.dg1, C_sym_grad[2,2])
        else:
            s00 = self.dg1.project(C_sym_grad[0,0])
            s01 = self.dg1.project(C_sym_grad[0,1])
            s11 = self.dg1.project(C_sym_grad[1,1])
            if self.mType in ['tet', 'hex']:
                s02 = self.dg1.project(C_sym_grad[0,2])
                s12 = self.dg1.project(C_sym_grad[1,2])
                s22 = self.dg1.project(C_sym_grad[2,2])

        self.sMats = np.zeros((self.nVerts,self.nDim,self.nDim), np.float32)
        cts = np.zeros(len(self.sMats))
        for tIdx, t in enumerate(self.meshInit.t.T):
            self.sMats[t] += self.C[tIdx]
            cts[t] += 1

        self.sMats /= cts.reshape(-1,1,1)
        dets = simpleDets(self.sMats)
        self.sMats *= simpleSign(dets.reshape(-1,1,1))

        self.pStress = np.zeros_like(self.sMats)
        self.pStressE = np.zeros((len(self.pStress), self.nDim), np.float32)
        for i, sMat in enumerate(self.sMats):
            self.pStress[i], self.pStressE[i] = computePrincipalStress(sMat)

        if self.mType in ['tet', 'hex']:
            self.vmStress = np.sqrt(((s00-s11)**2 + (s11-s22)**2 + (s22-s00)**2 + 6*(s01**2+s12**2+s02**2))/2)
        else:
            self.vmStress = np.sqrt(s00**2 + s11**2 - s00*s11 + 3*s01**2)

        return self.compliance

    def evalScaledJacobians(self, save = False):
        verts = self.verts if hasattr(self, 'verts') else self.meshInit.doflocs.T
        cells = self.ts if hasattr(self, 'ts') else self.meshInit.t.T
        self.SJs = computeJacobians(verts[cells], True)
        if save:
            mSJs = self.SJs.min(axis=1)
            mSJs.sort()
            np.savetxt(self.getFileName() + '.msjs', np.transpose([np.linspace(0,1,len(mSJs)), mSJs]), fmt='%.6e')
        return self.SJs

    def showPs(self): # somehow only works if called before any use of mayavi
        if not polyscopeFound or self.nDim < 3:
            print('Either polyscope is not installed or model is 2D.')
            return

        vsInit = self.verts if hasattr(self, 'verts') else self.meshInit.doflocs.T
        if self.mType == 'tet':
            tetsInit = self.ts if hasattr(self, 'ts') else self.meshInit.t.T
            psMeshInit = ps.register_volume_mesh("input_tetInit", vsInit, tets = tetsInit, enabled=True)
        else:
            hexasInit = self.meshInit.t.T[:,[0,3,6,2,1,5,7,4]]
            psMeshInit = ps.register_volume_mesh("input_hexInit", vsInit, hexes = hexasInit, enabled=True)

        if hasattr(self, 'vmStress'):
            psMeshInit.add_scalar_quantity("vMises", self.vmStress, enabled=True)
            psMeshInit.add_scalar_quantity("icdf", icdf(self.vmStress), enabled=False)
        if hasattr(self, 'vStress'):
            psMeshInit.add_scalar_quantity("vStress", self.vStress, enabled=False)
        if hasattr(self, 'SJs'):
            hs = self.ts if hasattr(self, 'ts') else self.meshInit.t.T
            mSJs = np.ones(len(vsInit))
            np.minimum.at(mSJs, hs.ravel(), self.SJs.ravel())
            psMeshInit.add_scalar_quantity("mSJs", mSJs, enabled=False, cmap = 'spectral')
            
        if hasattr(self, 'meshDeformed'):
            vsTrans = self.meshDeformed.doflocs.T
            if self.mType == 'tet':
                psTrans = ps.register_volume_mesh("result_tetDeformed", vsTrans, tets = tetsInit)                
            else:
                psTrans = ps.register_volume_mesh("result_hexDeformed", vsTrans, hexes = hexasInit)

            psTrans.add_scalar_quantity("vMovement", norm(self.u.reshape(-1,3)))
            psTrans.add_scalar_quantity("vMises", self.vmStress, enabled=True)
            psTrans.add_scalar_quantity("icdf", icdf(self.vmStress), enabled=False)
            psTrans.add_scalar_quantity("pSe0", normZeroToOne(self.pStressE[:,0]), enabled=False)
            psTrans.add_scalar_quantity("pSe1", normZeroToOne(self.pStressE[:,1]), enabled=False)
            psTrans.add_scalar_quantity("pSe2", normZeroToOne(self.pStressE[:,2]), enabled=False)

        ps.show()
        
    def show(self, showField = True, showVerts = False, showVectors = False, showDeformed = True, showLabels = False):

        if self.mType in ['tri', 'quad']:
            elementsInit = self.ts if hasattr(self, 'ts') else self.meshInit.t.T
            es = facesToEdges(elementsInit)
        elif self.mType == 'tet':
            elementsInit = self.ts if hasattr(self, 'ts') else self.meshInit.t.T
            es = tetsToEdges(elementsInit)
        elif self.mType == 'hex':
            elementsInit = self.meshInit.t.T[:,[0,3,6,2,1,5,7,4]]
            es = hexasToEdges(elementsInit)
        eTris = toEdgeTris(es)

        verts = self.verts if hasattr(self, 'verts') else self.meshInit.doflocs.T
        vsInit = pad2Dto3D(verts) if self.nDim == 2 else verts
        x,y,z = vsInit.T
        scals = icdf(self.vmStress) ** 0.5 if hasattr(self, 'vmStress') else np.zeros_like(x)
           
        # init wireframe
        if self.nDim == 2:
            self.mPlot = mlab.triangular_mesh(x, y, z, facesToTris(elementsInit), representation = 'surface', scalars = scals)
            self.mPlot = mlab.triangular_mesh(x, y, z, eTris, representation = 'mesh', tube_radius=self.sf*0.1, color = (0,0,0))
        else:
            self.mPlot = mlab.triangular_mesh(x, y, z, eTris, representation = 'mesh', tube_radius=self.sf*2, scalars = scals)

        # deformed wireframe
        if showDeformed and hasattr(self, 'meshDeformed'):
            vsTrans = self.meshDeformed.doflocs.T if self.mType in ['tet', 'hex'] else pad2Dto3D(self.meshDeformed.doflocs.T)
            xt,yt,zt = vsTrans.T
            if self.nDim == 2:
                self.dPlot = mlab.triangular_mesh(xt, yt, zt, facesToTris(elementsInit), representation = 'surface', scalars = scals)
            else:
                self.dPlot = mlab.triangular_mesh(xt, yt, zt, eTris, representation = 'mesh', tube_radius = self.sf*2, scalars = scals)
            
        # verts
        if showVerts:
            s = np.ones_like(x)
            self.vPlot = mlab.quiver3d(x, y, z, s, s, s, scalars = scals, scale_factor = 0.05, mode='sphere')
            self.vPlot.glyph.color_mode = 'color_by_scalar'
            self.vPlot.glyph.glyph_source.glyph_position = 'center'

        # labels
        if showLabels: # very slow, impractical for large models
            mlab.gcf().scene.disable_render = True # should speedup rendering but has no effect somehow
            for vIdx, v in enumerate(vsInit):
                mlab.text3d(v[0], v[1], v[2], str(vIdx), scale = (self.sf*10,self.sf*10,self.sf*10))
            mlab.gcf().scene.disable_render = False

        # deformation vectors
        if showVectors and hasattr(self, 'u'):
            uVecs = pad2Dto3D(self.u.reshape(-1,self.nDim))
            u,v,w = uVecs.T
            s = norm(uVecs)
            self.uPlot = mlab.quiver3d(x, y, z, u, v, w, scalars = s, scale_factor = 1, mode='arrow')

        # stress field
        if showField and hasattr(self, 'pStress'):
            x,y,z = np.repeat(vsInit.T, self.nDim, axis=1)
            pStressScale = 1#np.clip(self.pStressE / np.percentile(self.pStressE, 99, axis=0), 0, 1).reshape(-1,3,1) ** 0.5
            pStress = np.vstack(self.pStress * pStressScale)
            u,v,w = pStress.T if self.mType in ['tet', 'hex'] else pad2Dto3D(pStress).T
            scals = np.tile(np.arange(self.nDim)[::-1], len(vsInit))
            self.sPlot = mlab.quiver3d(x, y, z, u, v, w, scalars = scals)
            self.sPlot.glyph.color_mode = 'color_by_scalar'
            self.sPlot.glyph.glyph_source.glyph_source.glyph_type = 'dash'
            self.sPlot.glyph.glyph_source.glyph_position = 'center'

    def showForces(self, showLabels = False):

        if self.mType in ['tri', 'quad']:
            elementsInit = self.meshInit.t.T
            es = facesToEdges(elementsInit)        
        elif self.mType == 'tet':
            elementsInit = self.meshInit.t.T
            es = tetsToEdges(elementsInit)
        elif self.mType == 'hex':
            elementsInit = self.meshInit.t.T[:,[0,3,6,2,1,5,7,4]]
            es = hexasToEdges(elementsInit)
        eTris = toEdgeTris(es)

        vsInit = pad2Dto3D(self.meshInit.doflocs.T) if self.nDim == 2 else self.meshInit.doflocs.T
        x,y,z = vsInit.T

        # edges
        self.ePlot = mlab.triangular_mesh(x, y, z, eTris, representation = 'mesh', tube_radius=self.sf/5, color=(0,0,0))

        # labels
        if showLabels: # very slow, impractical for large models
            mlab.gcf().scene.disable_render = True # should speedup rendering but has no effect somehow
            for vIdx, v in enumerate(vsInit):
                if self.boundaryVertMask[vIdx]:
                    mlab.text3d(v[0], v[1], v[2], str(vIdx), scale = (self.sf*2.5,self.sf*2.5,self.sf*2.5))
            mlab.gcf().scene.disable_render = False

        # vertices
        scals = np.zeros_like(x)
        scals[self.vIdxsFix] = 1
        scals[self.vIdxsFlx] = 2
        print('nFix: %d, nFlex: %d'%(len(self.vIdxsFix), len(self.vIdxsFlx)))
        s = np.ones_like(x)
        self.vPlot = mlab.quiver3d(x, y, z, s, s, s, scalars = scals, scale_factor = self.sf*2, mode='sphere')
        self.vPlot.glyph.color_mode = 'color_by_scalar'
        self.vPlot.glyph.glyph_source.glyph_position = 'center'

        # applied forces
        if hasattr(self, 'f'):
            x,y,z = vsInit[self.vIdxsFlx].T
            fVecs = pad2Dto3D(self.f.reshape(-1,self.nDim)) if self.nDim == 2 else self.f.reshape(-1,self.nDim)
            u,v,w = fVecs[self.vIdxsFlx].T
            self.fPlot = mlab.quiver3d(x, y, z, u, v, w, scale_factor = self.sf*50, mode='arrow')
            self.fPlot.glyph.glyph_source.glyph_position = 'tail'

    def exportToStress(self, fileName = None):
        fileName = self.getFileName(fileName).split('_')[0] + '.stress'
        verts = self.verts if hasattr(self, 'verts') else self.meshInit.doflocs.T
        cells = self.ts if hasattr(self, 'ts') else self.meshInit.t.T
        fixedIdxs = self.vIdxsFix
        forceIdxs = self.vIdxsFlx
        forceVecs = self.f.reshape(-1, self.nDim)[self.vIdxsFlx]

        sMats = np.transpose(self.sMats, axes=[1,2,0])
        stress = sMats[[0,1,0],[0,1,1],:].T if self.nDim == 2 else sMats[[0,1,2,1,0,0],[0,1,2,2,2,1],:].T
        
        writeStressFile(fileName, verts, cells, forceIdxs, forceVecs, fixedIdxs, stress)

    def exportToMesh(self, fileName = None):
        fileName = self.getFileName(fileName) + ('.mesh' if self.nDim == 3 else '.obj')
        verts = self.verts if hasattr(self, 'verts') else self.meshInit.doflocs.T
        cells = self.ts if hasattr(self, 'ts') else self.meshInit.t.T

        if self.nDim == 2:
            writeObjFile(fileName, pad2Dto3D(verts), cells)
        elif self.mType == 'hex':
            writeMeshFile(fileName, verts, hexOrderDg2BT(cells[:,::-1]))

    def getFileName(self, fileName = None):
        if fileName is None:
            fileName = 'data/' + os.path.basename(self.inputFile).split('.')[0]
            fileName += '_' + self.mTypeInit
            if self.materialized:
                fileName += '_' + ('wall' if self.materialized[1] else 'beam')
                fileName += ('_%d'%(self.materialized[0]*100)) if self.materialized[0] > 0 else '_tau'
        return fileName            


if __name__ == "__main__":


    #inFile = 'data/bar2D.frc'
    #inFile = 'data/spot2D.frc'
    #inFile = 'data/bunny2D.frc'

    #inFile = 'data/cube.frc'
    inFile = 'data/femur.frc'
    #inFile = 'data/fertility.frc'
    #inFile = 'data/kitten.frc'
    #inFile = 'data/spot.frc'
    #inFile = 'data/venus.frc'

    #inFile = 'data/JEB.frc'
    #inFile = 'data/buddhaDown.frc'
    #inFile = 'data/buddhaBack.frc'
    #inFile = 'data/buddhaBelly.frc'

    
    fObj = FemObject(inFile, False)
    #fObj = FemObject(inFile, True)

    c0 = fObj.computeCompliance()
    print('c0:', c0)
    fObj.exportToStress()
    """
    fObj.materialize(0.5, False)
    #fObj.materializeOpti(0.5, False, weights = icdf(fObj.vmStress)**2)
    c1 = fObj.computeCompliance()
    print('c1:', c1)
    print('c:', c0, c1, c1/c0)
    print('v:', fObj.volInit, fObj.vol, fObj.vol/fObj.volInit)
    """
    #SJs = fObj.evalScaledJacobians()
    #print('msj:', fObj.SJs.min(), fObj.SJs.max(), SJs.mean())

    fObj.showPs()
    fObj.show(False, showDeformed = False)
    #fObj.showForces()


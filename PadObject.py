from femUtil import *

class PadObject:
    
    def __init__(self, tVerts, tris, qVerts, quads, pWeight = 0.1):
        self.tVerts, self.tris = tVerts, tris
        self.qVerts, self.quads = qVerts, quads
        self.nDim = tVerts.shape[1]
        self.qVerts = self.qVerts[:,:self.nDim]
        self.pWeight = pWeight

    @classmethod
    def getDeformedVerts(cls, tVerts, tris, qVerts, quads, pWeight = 0.1, projectedOnly = False):
        pObj = cls(tVerts, tris, qVerts, quads, pWeight)

        for i in range(2):
            pObj.computeNormals()
            pObj.computeProjected()
            if projectedOnly or i:
                return pObj.pVerts
            pObj.computeNeighborsAndWeights()
            pObj.computeDeformation(keepSteps=False)
            pObj.qVerts = pObj.dVerts
        
        return pObj.dVerts

    def computeNormals(self):
        if self.nDim == 2:
            qNormals = []
            for qIdx in range(len(self.qVerts)):
                nIdxs = np.unique(self.quads[(self.quads == qIdx).any(axis=1)])
                nIdxs = nIdxs[nIdxs != qIdx]
                qeVecs = self.qVerts[nIdxs] - self.qVerts[[qIdx, qIdx]]
                qeVecs = np.dot(qeVecs * [[1],[-1]], Mr2D(np.pi/2))
                qNormals.append(qeVecs.mean(axis=0))
            self.qNormals = normVec(np.float32(qNormals))
        else:
            if iglFound:
                self.qNormals = igl.per_vertex_normals(self.qVerts, quadsToTris(np.vstack([self.quads, np.roll(self.quads, 1, axis=1)])))
            elif trimeshFound:
                self.qtm = trimesh.Trimesh(self.qVerts, self.quads)
                self.qNormals = self.qtm.vertex_normals.copy()
            else: # fallback
                self.qNormals = np.zeros_like(self.qVerts)
                for vIdx in range(len(self.qVerts)):
                    qs = self.quads[np.any(self.quads == vIdx, axis=1)]
                    r = np.where(qs == vIdx)[1]
                    ts = rollRows(qs, -r)

                    us = self.qVerts[ts[:,1]] - self.qVerts[ts[:,0]]
                    vs = self.qVerts[ts[:,3]] - self.qVerts[ts[:,0]]
                    cs = cross(us, vs, False)

                    eL = self.qVerts[ts[:,0]] - self.qVerts[ts[:,1]]
                    eR = self.qVerts[ts[:,0]] - self.qVerts[ts[:,3]]
                    rL = self.qVerts[ts[:,3]] - self.qVerts[ts[:,1]]
                    rR = self.qVerts[ts[:,1]] - self.qVerts[ts[:,3]]
                    cL = inner1d(eL, rL) / norm(cross(eL, rL, False))
                    cR = inner1d(eR, rR) / norm(cross(eR, rR, False))
                    ws = (cL+cR)/2
    
                    self.qNormals[vIdx] = np.dot(ws/ws.sum(), normVec(cs))
                self.qNormals = normVec(self.qNormals)

    def computeProjected(self):
        if self.nDim == 2:

            self.pVerts = self.qVerts.copy()
            for i, pIdx in enumerate(np.unique(self.quads.ravel())):
                pInner = intersectEdgesWithRay2D(self.tVerts[self.tris], self.qVerts[pIdx] - self.qNormals[i] * eps, self.qNormals[i])
                pOuter = intersectEdgesWithRay2D(self.tVerts[self.tris], self.qVerts[pIdx] + self.qNormals[i] * eps, -self.qNormals[i])
                if norm(self.qVerts[pIdx] - pInner) < norm(self.qVerts[pIdx] - pOuter):
                    self.pVerts[pIdx] = pInner
                else:
                    self.pVerts[pIdx] = pOuter
        else:
            if embreeFound:
                if not hasattr(self, 'scene'):
                    self.embreeDevice = rtc.EmbreeDevice()
                    self.scene = rtcs.EmbreeScene(self.embreeDevice)
                    mesh = TriangleMesh(self.scene, self.tVerts[self.tris])

                qVerts = np.float32(np.tile(self.qVerts, [2,1]))
                qNormals = np.float32(np.vstack([self.qNormals, -self.qNormals]))
                dsts = self.scene.run(qVerts - qNormals * eps, qNormals, query='DISTANCE').reshape(2,-1)
                dists = np.min(dsts, axis=0) * (1-np.argmin(dsts, axis=0)*2)

            elif trimeshFound: # something is wrong here
                if not hasattr(self, 'tm'):
                    self.tm = trimesh.Trimesh(self.tVerts, self.tris)

                dists = np.zeros(len(self.qVerts), np.float32)
                triIdxs, rayIdxs, hitPoints = self.tm.ray.intersects_id(self.qVerts, -self.qNormals, return_locations=True, multiple_hits=False)
                dists[rayIdxs] = norm(self.qVerts[rayIdxs] - hitPoints)

                dsts = np.zeros(len(self.qVerts), np.float32)
                triIdxs, rayIdxz, hitPoints = self.tm.ray.intersects_id(self.qVerts, self.qNormals, return_locations=True, multiple_hits=False)
                dsts[rayIdxz] = norm(self.qVerts[rayIdxz] - hitPoints)

                #print(dsts.min(), dsts.max(), dsts.mean(), dsts.sum())
                #print(dists.min(), dists.max(), dists.mean(), dists.sum())
                rIdxs = np.unique(np.concatenate([rayIdxs, rayIdxz]))
    
                m = dsts[rIdxs] < dists[rIdxs]
                if np.any(m):
                    dists[rIdxs[m]] = -dsts[rIdxs[m]]

            else:
                print('install either embreex, pyembree or trimesh')
            
            qEdges = facesToEdges(self.quads)
            mel = norm(self.qVerts[qEdges[:,0]] - self.qVerts[qEdges[:,1]]).mean()
            dists = np.clip(dists, -mel, mel)

            self.pVerts = self.qVerts + self.qNormals * dists.reshape(-1,1)

    def computeNeighborsAndWeights(self):
        self.qNeighbors = np.zeros((len(self.qVerts), 6), np.int32)
        self.qWeights = np.zeros_like(self.qNeighbors, np.float32)

        n = len(self.qVerts)
        self.L = sparse.dok_matrix((n,n), dtype=float)
        E = sparse.dok_matrix((n,n), dtype=float)
        for vIdx in range(len(self.qVerts)):
            qs = self.quads[np.any(self.quads == vIdx, axis=1)]
            r = np.where(qs == vIdx)[1]
            ts = rollRows(qs, -r)

            neighborIdxs = np.unique(ts[:,[1,3]].ravel())
            self.qNeighbors[vIdx] = vIdx
            self.qNeighbors[vIdx,:len(neighborIdxs)] = neighborIdxs
            self.qWeights[vIdx,:len(neighborIdxs)] = 1/len(neighborIdxs)
            for nIdx in neighborIdxs:
                self.L[vIdx,nIdx] = 1/len(neighborIdxs)
            self.L[vIdx,vIdx] = -1
            E[vIdx,vIdx] = self.pWeight
        self.L *= -1

        self.solver = spla.factorized((self.L.T @ self.L + E).tocsc())
        self.truHoods = (self.qVerts[self.qNeighbors] - self.qVerts.reshape(-1,1,3)) * self.qWeights[:,:,None]

    def computeDeformation(self, num_iters = 10, keepSteps = True):
        Rs = np.float32([np.eye(3)] * len(self.qVerts))

        self.dVertss = [self.qVerts.copy()]
        for i in range(num_iters):
            if i:
                Rs = self.estimateRotations(self.dVerts)
            R = (np.take(Rs, self.qNeighbors, axis=0) + Rs[:,None]) / 2
            rhs = np.sum((R @ self.truHoods[...,None]).squeeze(), axis=1)
            self.dVerts = self.solver(self.L @ rhs + self.pVerts) * self.pWeight

            mvmt = norm(self.dVertss[-1] - self.dVerts).sum()
            if keepSteps:
                self.dVertss.append(self.dVerts.copy())
            else:
                self.dVertss[-1] = self.dVerts.copy()
                
            if mvmt < eps:
                break
            
        return self.dVerts

    def estimateRotations(self, dVerts):
        rotHoods = (dVerts[self.qNeighbors] - dVerts.reshape(-1,1,3))
        U,s,Vt = np.linalg.svd(rotHoods.transpose((0,2,1)) @ self.truHoods)
        Vt[:,-1,:] *= simpleDets3x3(U @ Vt)[:,None]
        return U @ Vt

    def show(self):
        mlab.figure()

        if not hasattr(self, 'dVerts'):
            self.dVerts = self.qVerts.copy()

        self.sf = norm(self.tVerts.max(axis=0)-self.tVerts.min(axis=0)) / 1000

        tVerts = self.tVerts
        pVerts = self.pVerts
        qVerts = self.qVerts
        dVerts = self.dVerts
        qNormals = self.qNormals
        if self.nDim == 2:
            tVerts, pVerts, qVerts, dVerts, qNormals = map(pad2Dto3D, [tVerts, pVerts, qVerts, dVerts, qNormals])
            
        x,y,z = tVerts.T
        eTris = toEdgeTris(facesToEdges(self.tris))
        tPlot = mlab.triangular_mesh(x, y, z, eTris, representation = 'mesh', color = (0,0,0), tube_radius = self.sf * 1.25)

        x,y,z = qVerts.T
        eTris = toEdgeTris(facesToEdges(self.quads))
        qPlot = mlab.triangular_mesh(x, y, z, eTris, representation = 'mesh',  color = (0.125,0.5,1), tube_radius = self.sf * 2.5)

        x,y,z = pVerts.T
        pPlot = mlab.triangular_mesh(x, y, z, eTris, representation = 'mesh',  color = (0,1,0), tube_radius = self.sf * 2.5)

        x,y,z = qVerts.T
        u,v,w = qNormals.T
        nPlot = mlab.quiver3d(x, y, z, u, v, w, color = (1,0,0), scale_factor=0.25)

if sys.platform == 'win32': # some TVTK issue on linux?
    
    class PadVisual(HasTraits):
        p = Range(0, 10, 0)
        scene = Instance(MlabSceneModel, ())
        meshPlot = Instance(PipelineBase)
        view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene), show_label=False), Group('p'), resizable=True)

        def show(self, pObj):
            self.pObj = pObj
            self.eTris = toEdgeTris(facesToEdges(self.pObj.quads))
            self.configure_traits()        

        @on_trait_change('p, scene.activated')
        def update_plot(self):
            p = min(len(self.pObj.dVertss)-1, self.p)
            dVerts = self.pObj.dVertss[p]

            if not hasattr(self, 'ePlot'):
                x,y,z = np.vstack([self.pObj.qVerts, self.pObj.pVerts]).T
                eTris = toEdgeTris(np.transpose([np.arange(len(self.pObj.qVerts)), np.arange(len(self.pObj.qVerts))+len(self.pObj.qVerts)]))
                self.ePlot = mlab.triangular_mesh(x, y, z, eTris, representation = 'mesh',  color = (0,1,1), tube_radius=0.005)

                x,y,z = self.pObj.qVerts.T
                self.qPlot = mlab.triangular_mesh(x, y, z, self.eTris, representation = 'mesh',  color = (0.125,0.5,1), tube_radius = self.pObj.sf * 2.5)

                x,y,z = self.pObj.pVerts.T
                self.pPlot = mlab.triangular_mesh(x, y, z, self.eTris, representation = 'mesh',  color = (0,1,0), tube_radius = self.pObj.sf * 2.5)

                x,y,z = dVerts.T
                self.dPlot = mlab.triangular_mesh(x, y, z, self.eTris, representation = 'mesh',  color = (1,0,0), tube_radius = self.pObj.sf * 5)
            else:
                self.dPlot.mlab_source.points = dVerts


if __name__ == "__main__":

    # first call cObj.exportHullsToObj() on a CubeObject   
    tVerts, tris = loadObjFile('data/femurHullTris.obj')
    qVerts, quads = loadObjFile('data/femurHullQuads.obj')
    pObj = PadObject(tVerts, tris, qVerts, quads)
    pObj.computeNormals()
    pObj.computeProjected()
    pObj.computeNeighborsAndWeights()
    pObj.computeDeformation()

    pObj.show()

    pVis = PadVisual()
    pVis.show(pObj)


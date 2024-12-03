try:
    from drbutil import *
except ImportError:
    drbutilFound = False
else:
    drbutilFound = True

try:
    import polyscope as ps
except ImportError:
    polyscopeFound = False
else:
    polyscopeFound = True
    ps.set_up_dir('z_up')
    ps.init()

try:
    import igl
except ImportError:
    iglFound = False
else:
    iglFound = True

try:
    import cupy as cp
except ImportError:
    cupyFound = False
else:
    cupyFound = True

try:
    import trimesh
except ImportError:
    trimeshFound = False
else:
    trimeshFound = True

try:
    from embreex import rtcore as rtc
    from embreex import rtcore_scene as rtcs
    from embreex.mesh_construction import TriangleMesh
except ImportError:
    try:
        from pyembree import rtcore as rtc
        from pyembree import rtcore_scene as rtcs
        from pyembree.mesh_construction import TriangleMesh
    except ImportError:
        embreeFound = False
    else:
        embreeFound = True
else:
    embreeFound = True

try:
    import scipy
    from scipy.interpolate import griddata
except ImportError:
    scipyFound = False
else:
    scipyFound = True

try:
    import scipy.sparse as sparse
    import scipy.sparse.linalg as spla
except ImportError:
    sciSparseFound = False
else:
    sciSparseFound = True

try:
    import skfem as fem
    from skfem.models.elasticity import linear_elasticity, lame_parameters, linear_stress
    from skfem.helpers import dot, ddot, sym_grad, jump, mul
    from skfem.assembly import BilinearForm
except ImportError:
    skfemFound = False
else:
    skfemFound = True


# fast sparse factorization and solving
try:
    import pypardiso
except ImportError:
    pypardisoFound = False
else:
    pypardisoFound = True
    print('using pypardiso')
    def solver_pypardiso(A, b, **kwargs):
        # mkl stores factorized (huge) matrix in undeletable memory
        return pypardiso.spsolve(A, b, factorize = len(b) < 1000000, **kwargs)

    def dg1Project(dg1, Csg):
        M, f = dg1._projection(Csg)
        if dg1.tind is not None:
            return fem.solve(*fem.condense(M, f, I=dg1.get_dofs(elements=dg1.tind)), solver = solver_pypardiso)
        return fem.solve(M, f, solver = solver_pypardiso)


if sys.platform != 'win32' or '-mp' in sys.argv:
    print('mpCPU:', cpuCount)
    def linear_elasticity(Lambda=1., Mu=1.):
        """Weak form of the linear elasticity operator."""

        C = linear_stress(Lambda, Mu)

        @BilinearForm(nthreads=cpuCount)
        def weakform(u, v, w):
            return ddot(C(sym_grad(u)), sym_grad(v))

        return weakform

"""
def sparseCond(M):
    ew1, ev = sparse.linalg.eigsh(M, which='LM')
    ew2, ev = sparse.linalg.eigsh(M, sigma=eps)
    return np.abs(ew1).max() / np.abs(ew2).min()
"""

def lassoShrink(x, k):
    return np.maximum(x - k, 0.0) - np.maximum(-x - k, 0.0)

def sampleStar(cIdx, nCells, vts):
    sPts = []

    for nCell in nCells:
        qes = faceToEdges(nCell) if len(nCell) == 4 else hexaToEdges(nCell)
        nIdxs = np.unique(ringNeighborElements(qes, [cIdx]).ravel())
        if vts.shape[1] == 2:
            sPts.append(generateTriGridSamples(vts[nIdxs], 13, False))
        else:
            sPts.append(generateTetGridSamples(vts[nIdxs], 8, False))
    return np.vstack(sPts)

def computeOptiPoint(cIdx, cells, vts, useBB = False):
    nCells = ringNeighborElements(cells, [cIdx])
    if useBB:
        pts = vts[nCells.ravel()]
        bbMin = pts.min(axis=0)
        bbMax = pts.max(axis=0)
        if pts.shape[1] < 3:
            bbVerts = quadVerts * (bbMax-bbMin)/2 + (bbMax+bbMin)/2
            sPts = generateQuadGridSamples(bbVerts, 8, False)
        else:
            bbVerts = cubeVerts * (bbMax-bbMin)/2 + (bbMax+bbMin)/2
            sPts = generateHexGridSamples(bbVerts, 8, False)
    else:
        sPts = sampleStar(cIdx, nCells, vts)

    tVts = vts.copy()
    mSJs = []
    for sPt in sPts:
        tVts[cIdx] = sPt
        mSJs.append(computeJacobians(tVts[nCells], True).min())

    if useBB:
        x,y,z = sPts.T
        s = z*0+1
        scals = normZeroToOne(mSJs)**0.5
        sPlot = mlab.quiver3d(x, y, z, s, s, s, scalars = scals, scale_factor = 0.0025, mode='sphere')
        sPlot.glyph.color_mode = 'color_by_scalar'
        sPlot.glyph.glyph_source.glyph_position = 'center'
        sys.exit()

    return sPts[np.argmax(mSJs)]   

alphaMin = 0.05
tauMinMax = {'tri': [(1 - np.sqrt(1 - alphaMin)) / 3, 1/3],
             'quad': [(1 - np.sqrt(1 - alphaMin)) / 2, 1/2],
             'tet': [0.01, 1/5],
             'tetw': [0.01, 1/5],
             'hex': [np.arccos(1-2*alphaMin)/(2*np.pi), 1/2],
             'hexw': [(1 - (1-alphaMin)**(1/3))/2, 1/2]}

def bandedTau(p, tau, mType, walled = False):
    tMin, tMax = tauMinMax[mType + 'w' * walled]
    return (tau * (1-abs(p)) + max(p,0)) * (tMax-tMin) + tMin

def materializeFaceVerts(p, tau, verts, edges, faces, mType, averagedTaus = False):
    taus = np.ones(len(verts)) * tau if p is None else bandedTau(p, tau, mType)

    eVerts = []
    for edge in edges:
        t0,t1 = [taus[edge].mean()]*2 if averagedTaus else taus[edge]
        eWs = [[1-t0, t0], [t1, 1-t1]]
        eVerts.append(np.dot(eWs, verts[edge]))

    fVerts = []
    for face in faces:
        if mType == 'tri':
            t0,t1,t2 = [taus[face].mean()]*3 if averagedTaus else taus[face]
            fWs = [[1-2*t0,t0,t0], [t1,1-2*t1,t1], [t2,t2,1-2*t2]]
        elif mType == 'quad':
            a = lambda t: t**2-2*t+1
            b = lambda t: t-t**2
            c = lambda t: t**2
            t0,t1,t2,t3 = [taus[face].mean()]*4 if averagedTaus else taus[face]
            fWs = [[a(t0), b(t0), c(t0), b(t0)],
                   [b(t1), a(t1), b(t1), c(t1)],
                   [c(t2), b(t2), a(t2), b(t2)],
                   [b(t3), c(t3), b(t3), a(t3)]]
        else:
            t = taus[face].mean()
            fW = [t, 1-2*t, t] + [0] * (len(face)-3)
            fWs = [np.roll(fW, s-1) for s in range(len(face))]            
            
        fVerts.append(np.dot(fWs, verts[face]))

    return np.vstack([verts, np.vstack(eVerts), np.vstack(fVerts)])

def materializeTetVerts(p, tau, verts, edges, tris, tets, mType, walled, averagedTaus = True):
    taus = np.ones(len(verts)) * tau if p is None else bandedTau(p, tau, mType, walled)

    eVerts = []
    for edge in edges:
        t0,t1 = [taus[edge].mean()]*2 if averagedTaus else taus[edge]
        eWs = [[1-t0, t0], [t1, 1-t1]]
        if walled:
            eWs += [[(1-t0+t1)/2,(1-t1+t0)/2]]
        eVerts.append(np.dot(eWs, verts[edge]))

    fVerts = []
    for tri in tris:
        t0,t1,t2 = [taus[tri].mean()]*3 if averagedTaus else taus[tri]
        fWs = [[1-2*t0,t0,t0], [t1,1-2*t1,t1], [t2,t2,1-2*t2]]
        if walled:
            fWs += [[(t1+t2)/2,(t2+1-2*t1)/2,(t1+1-2*t2)/2],
                    [(t2+1-2*t0)/2,(t2+t0)/2,(t0+1-2*t2)/2],
                    [(t1+1-2*t0)/2,(t0+1-2*t1)/2,(t0+t1)/2],
                    [(1-2*t0+t1+t2)/3,(t0+1-2*t1+t2)/3,(t0+t1+1-2*t2)/3]]
        fVerts.append(np.dot(fWs, verts[tri]))

    cVerts = []
    for tet in tets:
        t0,t1,t2,t3 = [taus[tet].mean()]*4 if averagedTaus else taus[tet]
        cWs = [[1-3*t0,t0,t0,t0], [t1,1-3*t1,t1,t1], [t2,t2,1-3*t2,t2], [t3,t3,t3,1-3*t3]]
        if walled:
            cWs += [[(1-3*t0+t1)/2,(1-3*t1+t0)/2,(t0+t1)/2,(t0+t1)/2],
                    [(t1+t2)/2,(1-3*t1+t2)/2,(1-3*t2+t1)/2,(t1+t2)/2],
                    [(1-3*t0+t2)/2,(t0+t2)/2,(1-3*t2+t0)/2,(t0+t2)/2],
                    [(1-3*t0+t3)/2,(t0+t3)/2,(t0+t3)/2,(1-3*t3+t0)/2],
                    [(t1+t3)/2,(1-3*t1+t3)/2,(t1+t3)/2,(1-3*t3+t1)/2],
                    [(t2+t3)/2,(t2+t3)/2,(1-3*t2+t3)/2,(1-3*t3+t2)/2],
                    [(t1+t2+t3)/3,(1-3*t1+t2+t3)/3,(t1+1-3*t2+t3)/3,(t1+t2+1-3*t3)/3],
                    [(1-3*t0+t2+t3)/3,(t0+t2+t3)/3,(t0+1-3*t2+t3)/3,(t0+t2+1-3*t3)/3],
                    [(1-3*t0+t1+t3)/3,(t0+1-3*t1+t3)/3,(t0+t1+t3)/3,(t0+t1+1-3*t3)/3],
                    [(1-3*t0+t1+t2)/3,(t0+1-3*t1+t2)/3,(t0+t1+1-3*t2)/3,(t0+t1+t2)/3]]
        cVerts.append(np.dot(cWs, verts[tet]))

    return np.vstack([verts, np.vstack(eVerts), np.vstack(fVerts), np.vstack(cVerts)])

def materializeHexVerts(p, tau, verts, edges, quads, hexas, mType, walled, averagedTaus = True):    
    taus = np.ones(len(verts)) * tau if p is None else bandedTau(p, tau, mType, walled)

    eVerts = []
    for edge in edges:
        t0,t1 = [taus[edge].mean()]*2 if averagedTaus else taus[edge]
        eWs = [[1-t0, t0], [t1, 1-t1]]
        eVerts.append(np.dot(eWs, verts[edge]))

    fVerts = []
    for quad in quads:
        a = lambda t: t**2-2*t+1
        b = lambda t: t-t**2
        c = lambda t: t**2
        t0,t1,t2,t3 = [taus[quad].mean()]*4 if averagedTaus else taus[quad]
        fWs = [[a(t0), b(t0), c(t0), b(t0)],
               [b(t1), a(t1), b(t1), c(t1)],
               [c(t2), b(t2), a(t2), b(t2)],
               [b(t3), c(t3), b(t3), a(t3)]]
        fVerts.append(np.dot(fWs, verts[quad]))

    cVerts = []
    for hexa in hexas:
        t0,t1,t2,t3,t4,t5,t6,t7 = [taus[hexa].mean()]*8 if averagedTaus else taus[hexa]
        a = lambda t: -(t-1)**3
        b = lambda t: (t-1)**2 * t
        c = lambda t: t**2-t**3
        d = lambda t: t**3
        cWs = [[a(t0),b(t0),b(t0),b(t0),c(t0),c(t0),c(t0),d(t0)],
               [b(t1),a(t1),c(t1),c(t1),b(t1),b(t1),d(t1),c(t1)],
               [b(t2),c(t2),a(t2),c(t2),b(t2),d(t2),b(t2),c(t2)],
               [b(t3),c(t3),c(t3),a(t3),d(t3),b(t3),b(t3),c(t3)],
               [c(t4),b(t4),b(t4),d(t4),a(t4),c(t4),c(t4),b(t4)],
               [c(t5),b(t5),d(t5),b(t5),c(t5),a(t5),c(t5),b(t5)],
               [c(t6),d(t6),b(t6),b(t6),c(t6),c(t6),a(t6),b(t6)],
               [d(t7),c(t7),c(t7),c(t7),b(t7),b(t7),b(t7),a(t7)]]
        cVerts.append(np.dot(cWs, verts[hexa]))
    return np.vstack([verts, np.vstack(eVerts), np.vstack(fVerts), np.vstack(cVerts)])    

def optimizeTaus(vTarget, taus, verts, cellsInit, cellsMat, mType, walled = False):
    pInit = 0

    def showProgress(vTarget, vCurr, vDelta, p):
        print('vt: %0.6f, vc: %0.6f, vd: %0.6f, p: %0.6f'%(vTarget, vCurr, vDelta, p))

    if mType in ['tri', 'quad']:
        edges, faces = cellsInit
        quadTris = facesToTris(cellsMat)
        def f(p, *args):
            vTarget, taus, verts = args[0]
            verts = materializeFaceVerts(p, taus, verts, edges, faces, mType)
            area = computeTriangleAreas(verts[quadTris], False).sum()
            err = abs(vTarget - area)**2
            showProgress(vTarget, area, err, p[0])
            return err

    elif mType == 'tet':
        edges, faces, cells = cellsInit
        hexaTets = np.vstack([hexaToTetras(hexa) for hexa in hexOrderDg2BT(cellsMat)])
        def f(p, *args):
            vTarget, taus, verts = args[0]
            verts = materializeTetVerts(p, taus, verts, edges, faces, cells, mType, walled)
            vol = computeTetraVolumes(verts[hexaTets], False).sum()
            err = abs(vTarget - vol)**2
            showProgress(vTarget, vol, err, p[0])
            return err

    elif mType == 'hex':
        edges, faces, cells = cellsInit
        hexaTets = np.vstack([hexaToTetras(hexa) for hexa in hexOrderDg2BT(cellsMat)])
        def f(p, *args):            
            vTarget, taus, verts = args[0]
            verts = materializeHexVerts(p, taus, verts, edges, faces, cells, mType, walled)
            vol = computeTetraVolumes(verts[hexaTets], False).sum()
            err = abs(vTarget - vol)**2
            showProgress(vTarget, vol, err, p[0])
            return err

    # use L-BFGS-B or Nelder-Mead
    print('optimizing tau*')
    p = scipy.optimize.minimize(f, pInit, method="L-BFGS-B", bounds = [(-1,1)], args = [vTarget, taus, verts], options={"maxiter": 100}).x
    
    return p

def computeBarycentricWeights(verts, tVerts, nDim, innerOnly = False, hullTolerance = False):

    if iglFound and not cupyFound:
        tVertsT = [np.float32(tVerts[:,i,:]) for i in range(nDim+1)]
        tVertsT = list(map(pad2Dto3D, tVertsT)) if nDim == 2 else tVertsT
            
    if nDim == 2:
        tVols = computeTriangleAreas(tVerts)
        subTVerts = tVerts[:,[[1,2],[2,0],[0,1],]].reshape(-1, nDim, nDim)
            
    else:
        if iglFound and innerOnly and not hullTolerance:
            hullVerts, hullTris = innerOnly
            windingNumbers = np.abs(igl.fast_winding_number_for_meshes(np.array(hullVerts, np.float64, order='F'), hullTris, verts))
            
        tVols = computeTetraVolumes(tVerts, False)
        subTVerts = tVerts[:,[[1,2,3],[0,3,2],[0,1,3],[0,2,1]]].reshape(-1, nDim, nDim)

    if cupyFound: # cupy to much overhead for 2D ? 
        tVols = cp.array(tVols)
        subTVerts = cp.array(subTVerts)
        verts = cp.array(verts)
        tVerts = cp.array(tVerts)

    tIdxs = np.zeros(len(verts), np.int32)-1
    bWeights = np.zeros((len(verts), nDim+1), np.float32)
    for vIdx, vert in tqdm(enumerate(verts), total = len(verts), ascii=True, desc='baryWeights'):
            
        if nDim == 2:
            if not iglFound or cupyFound:
                subTriVols = simpleDets2x2(subTVerts - vert.reshape(1,2)) / 2.0
                bCoords = subTriVols.reshape(-1,3) / tVols.reshape(-1,1)
            else:
                ps = pad2Dto3D(np.float32(np.repeat([vert], len(tVerts), axis=0)))
                bCoords = igl.barycentric_coordinates_tri(ps, tVertsT[0], tVertsT[1], tVertsT[2])
        else:
            if iglFound and innerOnly and not hullTolerance and windingNumbers[vIdx] < 0.5:
                continue

            if not iglFound or cupyFound:
                subTetVols = simpleDets3x3(subTVerts - vert.reshape(1,3)) / 6.0
                bCoords = subTetVols.reshape(-1,4) / tVols.reshape(-1,1)
            else:
                ps = np.float32(np.repeat([vert], len(tVerts), axis=0))
                bCoords = igl.barycentric_coordinates_tet(ps, tVertsT[0], tVertsT[1], tVertsT[2], tVertsT[3])

        if innerOnly or innerOnly is None:
            if hullTolerance:
                bCoordsNorm = cp.linalg.norm(bCoords, axis=1) if cupyFound else norm(bCoords)
                validMask = bCoordsNorm < 2
            else:
                validMask = (bCoords >= 0).all(axis=1)
            
            if cupyFound:
                validMask = validMask.get()
           
            if validMask.any():
                if hullTolerance:
                    tIdx = bCoordsNorm.argmin()
                    bWs = cp.clip(bCoords[tIdx], 0, 1) if cupyFound else np.clip(bCoords[tIdx], 0, 1)
                    pVert = cp.dot(bWs / bWs.sum(), tVerts[tIdx]) if cupyFound else np.dot(bWs / bWs.sum(), tVerts[tIdx])
                    if (cp.linalg.norm(vert - pVert) if cupyFound else norm(vert - pVert)) > hullTolerance:
                        continue
                else:
                    tIdx = np.where(validMask)[0][0]
                tIdxs[vIdx] = tIdx
                bWeights[vIdx] = bCoords[tIdx].get() if cupyFound else bCoords[tIdx]
        else:
            tIdx = (cp.linalg.norm(bCoords, axis=1) if cupyFound else norm(bCoords)).argmin()
            tIdxs[vIdx] = tIdx
            bWeights[vIdx] = bCoords[tIdx].get() if cupyFound else bCoords[tIdx]

    return bWeights, tIdxs

# FreeTO numerical specification (MATLAB → 0-based numpy)

Source of truth: the original FreeTO MATLAB sources (`*.m`, not shipped). All indices below are **0-based** unless marked `(1b)`.
Array shapes are given in MATLAB order `(nely, nelx, nelz)` = `(rows, cols, pages)`; "flatten" always
means **column-major** (`order='F'`), which is what MATLAB `(:)` / `reshape` / `find` do.

Notation: `h = ssz` (element edge, mm). `ny = nely+1, nx = nelx+1, nz = nelz+1`. `N = ny*nx*nz` nodes,
`nele = nely*nelx*nelz`, `ndof = 3N`.

---------------------------------------------------------------------------------------------------
## 1. geomeshini — grid construction

### 1.1 Axis selection (quirk: uses |coordinates|, not extents)
```
stt  = abs([xmin xmax ymin ymax zmin zmax])      # bounding box of the DOMAIN STL vertices
stff = first index (1b) where stt == max(stt)
stff ∈ {1,2} → MeshControl axis = x ; {3,4} → y ; {5,6} → z
```
So a domain occupying x∈[100,110], y∈[0,50] gets `MeshControl` nodes along **x** (|110| is largest),
giving 50/10·MC points along y. Replicate faithfully (it is what the README examples were tuned on).
Ties: `find` returns the lowest index → priority x > y > z.

### 1.2 Node coordinates
Let the chosen axis be `p` (say x), the other two `q, r`. With `MC = MeshControl`:
```
p_old   = linspace(pmin, pmax, MC);  ssz_old = (p_old[1]-p_old[0]) * 1e-2        # 1 % of raw spacing
p       = linspace(pmin + ssz_old, pmax - ssz_old, MC)                            # shrunk 1 % at each end
h = ssz = p[1] - p[0]  = (pmax - pmin - 2*ssz_old)/(MC-1)                          # note: NOT (pmax-pmin)/(MC-1)
a       = ssz/2000                                                                # "half element size in m"
q_old   = colon(qmin, h, qmax);   diff_q = qmax - q_old[-1]          (0 ≤ diff_q < h)
q       = colon(qmin + diff_q/2, h, qmax)                            # centred in [qmin,qmax], NOT shrunk
gp_p = MC, gp_q = len(q), gp_r = len(r)                              # node counts
```
MATLAB `linspace(d1,d2,n)` evaluates `d1 + ((0:n-1).*(d2-d1))./(n-1)` (multiply first, then divide) and then
forces both endpoints; numpy does `arange(n)*((d2-d1)/(n-1)) + d1`, which can differ by 1 ulp at interior
points. `ssz = x[1]-x[0]` is identical either way; for ulp-faithful node coordinates use
`x = d1 + (np.arange(n)*(d2-d1))/(n-1); x[0]=d1; x[-1]=d2`.

**MATLAB colon `a:s:b` count.** MATLAB/Octave do **not** use a naive `floor((b-a)/s)+1`; they use a
tolerant floor (Octave: `tfloor(x, 3*eps)`; MATLAB is equivalent in practice): the count is
`n = floor(r + tol) + 1`, `r = (b-a)/s`, `tol ≈ 3*eps*max(1,|r|)`, and the last element is clamped to `≤ b`.
Recommended robust equivalent (verified: `0:0.1:0.3` → 4 pts, `0:0.1:1` → 11):
```python
def mcolon(a, s, b):
    r = (b - a) / s
    n = int(np.floor(r + 3*np.finfo(float).eps*max(1.0, abs(r)))) + 1
    v = a + s*np.arange(n)
    v[-1] = min(v[-1], b)
    return v
```
Because `q` is derived from `q_old` with `diff_q/2 < h/2`, `len(q) == len(q_old)` always (the tolerance only
matters when `(qmax-qmin)/h` is within ~1e-15 of an integer; then `diff_q≈0` and nodes lie **exactly on the
domain faces**, see §8 for what intriangulation does with those).

### 1.3 Element centres
```
x_cen = (x + h/2)[:-1]  (len nelx = gp_x-1), same for y_cen, z_cen           # derived from x, not midpoints
nelx = gp_x-1, nely = gp_y-1, nelz = gp_z-1
```

### 1.4 meshgrid, flatten, flip
`[x1,y1,z1] = meshgrid(x,y,z)` gives arrays of shape `(gp_y, gp_x, gp_z)` with
`x1[j,i,k] = x[i]`, `y1[j,i,k] = y[j]`, `z1[j,i,k] = z[k]`. `[x1(:) y1(:) z1(:)]` is the column-major
flatten: point `L = j + gp_y*i + gp_y*gp_x*k` (0b) is `(x[i], y[j], z[k])`. **Every** intriangulation result
is then `flip(reshape(out, gp_y, gp_x, gp_z), 1)` → row index reversed.

**Grid frame (after the flip) — the frame in which everything downstream (node ids, DOFs, elements, xg,
top, eleden, symmetry) lives:**

| entity | array index (0b) | physical position |
|---|---|---|
| node `(j,i,k)` in `(ny,nx,nz)` | `n = j + ny*i + ny*nx*k` | `x = x[i]`, `y = y[ny-1-j]`, `z = z[k]` |
| DOFs of node n | `3n, 3n+1, 3n+2` = ux, uy, uz | (1b: `3n-2,3n-1,3n` for 1b node n) |
| element `(j,i,k)` in `(nely,nelx,nelz)` | `e = j + nely*i + nely*nelx*k` | centre `x_cen[i]`, `y_cen[nely-1-j]`, `z_cen[k]` |
| fine grid point `(J,I,K)` (ngrid=4) | shape `(4nely+1, 4nelx+1, 4nelz+1)` | `x[0]+I·h/4`, `y[ny-1]−J·h/4`, `z[0]+K·h/4` |

i.e. **row index increases with decreasing y**; columns ↔ x ascending; pages ↔ z ascending. Node `(j,i,k)`
is the corner of element `(j,i,k)` with the *largest* y and smallest x, z. The y-flip is what makes the
top3d `lk_H8` (right-handed, counter-clockwise node order) consistent with physical coordinates (§2).

Python recommendation: compute the membership masks directly on the flipped grid by building the physical
coordinate arrays in the grid frame (`Y[j] = y[ny-1-j]`), so no explicit flip is needed; conversion to the
contract's `(x,y,z)`-ordered `FieldSnapshot` is `np.transpose(A[::-1,:,:], (1,0,2))` with
`origin = (x[0], y[0], z[0])`, `spacing = (h/4,)*3` for fine-grid arrays (or `h` for element arrays).

### 1.5 Outputs of geomeshini
* `out_p` (unused downstream), `oute` = domain membership of element centres (`-1 → 0`), shape `(nely,nelx,nelz)`.
* `sup_all, sup_x, sup_y, sup_z` = 1b node ids (`find(out>0)`, so `-1` counts as outside) — 0b: `n` as in the table.
* `Fn` = sparse `(N, nforcefiles)`; column i holds the 1b node ids inside force STL i (padded with 0).
* `outeM` = "must keep" element mask (§1.6), `nele, ndof, stp1` (domain vertices), `a`, `del_x/y/z` (domain extents).

### 1.6 domainstokeep
Builds the list of STL regions to keep: always `keepdom`; if `keep_BC=='yes'` also `fixed` and **all** force
STLs; plus `xfixed/yfixed/zfixed` when the corresponding `keep_BCx/y/z=='yes'`. For each region (in order
fixed, [x,y,z fixed], force1..forceN, keepdom) it accumulates `outeM += intri(region, element centres)`
then clamps `<0→0, >1→1` **after each addition**. Quirk: a `-1` (undecided) from a later region cancels a
`1` from an earlier one. Python: treat each region's result as boolean (`==1`) and OR them (fix; document;
identical unless the inside test is undecided). Regions are tested against **all** element centres, not only
those inside the domain → `MusD` may contain passive elements (see §6.4).

---------------------------------------------------------------------------------------------------
## 2. domainprep — connectivity and index sets

`nodenr[j,i,k] = 1 + j + ny*i + ny*nx*k` (1b). `nodenrs1 = nodenr-1` cropped to `(nely,nelx,nelz)` = 0b id
`n0` of the element's node `(j,i,k)`. For element `e` (flattened column-major over `(nely,nelx,nelz)`):

**edofMat1 (0b node ids)** — 8 local nodes, column `a` ↔ node offset:

| a | offset from n0 | array node | physical (relative to element) |
|---|---|---|---|
| 0 | `+1`          | `(j+1, i,   k)`   | (x0, y0, z0)  low-y corner |
| 1 | `+1+ny`       | `(j+1, i+1, k)`   | (x1, y0, z0) |
| 2 | `+ny`         | `(j,   i+1, k)`   | (x1, y1, z0) |
| 3 | `0`           | `(j,   i,   k)`   | (x0, y1, z0) |
| 4 | `+1+ny*nx`    | `(j+1, i,   k+1)` | (x0, y0, z1) |
| 5 | `+1+ny+ny*nx` | `(j+1, i+1, k+1)` | (x1, y0, z1) |
| 6 | `+ny+ny*nx`   | `(j,   i+1, k+1)` | (x1, y1, z1) |
| 7 | `+ny*nx`      | `(j,   i,   k+1)` | (x0, y1, z1) |

(x0<x1, y0<y1, z0<z1.) This is exactly the top3d/`lk_H8` hexahedron: nodes 1-4 counter-clockwise on the
z0 face starting at the origin corner, 5-8 the same on z1. Verified: with these unit-cube coordinates,
`u_x = x` gives `uᵀ KE u = C11 = (1-ν)/((1+ν)(1-2ν))` (1.34615 at ν=0.3), and rigid rotations are in the null
space. **Do not reorder**; a left-handed ordering would silently corrupt shear terms.

**edofMat (0b DOF ids)**: `edofMat[e, 3a+c] = 3*edofMat1[e,a] + c`, c∈{0,1,2} = (ux,uy,uz). (MATLAB's
explicit offset list `[4 5 6 4+3ny ...]` is precisely this.) `KE` row/col `3a+c` ↔ same.

Index sets:
```
ndn   = elements with oute == 0            (passive; oute<0 already zeroed)
ele   = sorted active element ids (setdiff) ; nnele = len(ele)
edofMatn = edofMat[ele, :]                  (nnele x 24, in ele order)
n_vec = np.unique(edofMatn)                 (sorted active DOFs) = "alldofs" in SIMP.m
vol1  = vol*nnele/nele                      (returned but UNUSED by SIMP/SEMDOT)
MusD  = element ids with outeM == 1         (forced density 1; may include passive elements)
```
Everything optimisation-side (vx, vxPhys, dc, dv, H, Hs) is a length-`nnele` vector in `ele` order.

---------------------------------------------------------------------------------------------------
## 3. Loads and supports

### 3.1 forcevec(Fn, ndof, Fmagx, Fmagy, Fmagz, loadtype)
```
nf = max(len(Fmagx), len(Fmagy), len(Fmagz))          # scalars have len 1
Fx = Fy = Fz = zeros(ndof, nf)
for i in 0..nf-1:
    frn = nonzero entries of Fn[:, i]   (1b node ids of force file i)  -> ERROR in MATLAB if i >= n_forcefiles
    if loadtype == 'point': frn = sort(frn); frn = [ frn[(len(frn)-1)//2] ]      # MATLAB frn(round(len/2)), 1b
    n = len(frn)                                                                     # (empty frn -> MATLAB index error)
    dofs_x = 3*frn0, dofs_y = 3*frn0+1, dofs_z = 3*frn0+2      (frn0 = frn-1, 0b)
    if len(Fmagx) > 1: Fx[dofs_x, i] = Fmagx[i]/n     else: Fx[dofs_x, 0] = Fmagx/n        # <-- column 0 !
    (same for y, z)
F = Fx+Fy+Fz;  drop columns that are entirely zero;  sparse
```
Semantics/quirks:
* `'point'` picks the **median node id** of the sorted id list (0b index `(n-1)//2`), not the geometric
  centre; then `n = 1`.
* **Scalar (length-1) Fmag component is always written into column 0**, for *every* load region i (each
  region's nodes get `Fmag/n_i`; overlapping nodes are overwritten, not summed). Non-scalar components go to
  column i. E.g. `Fmagx=1000, Fmagy=[0,-500]` with 2 regions → column 0: x-load on region 1 ∪ region 2;
  column 1: y-load on region 2.
* Load case `i ≥ n_forcefiles` → MATLAB errors. Extra force files beyond nf are ignored for loading but still
  used by `domainstokeep`. Python: `validate()` should require `nf <= len(forces)` (recommend `==`).
* All-zero columns are removed, so `F.shape[1]` may be `< nf`. MATLAB then does `U = zeros(ndof,nf)` and
  `U(freedofs,:) = K\F(freedofs,:)` which **errors** on size mismatch. Python: allocate `U` with `F.shape[1]`
  columns (fix).
* Point-load magnitudes stay `Fmag` (n=1); distributed loads are `Fmag/n` per node → total = `Fmag`.

### 3.2 supportDOFs
`fixeddof = sorted unique( {3n,3n+1,3n+2 : n∈sup_all} ∪ {3n : n∈sup_x} ∪ {3n+1 : n∈sup_y} ∪ {3n+2 : n∈sup_z} )` (0b).
Support nodes need not belong to active elements.

### 3.3 Free DOFs
`freedofs = setdiff(n_vec, fixeddof)` (sorted). SIMP.m calls `n_vec` "alldofs". DOFs of nodes not touched
by any active element are excluded automatically (they would make K singular). `K` must be assembled with
explicit shape `(ndof, ndof)` (MATLAB's `sparse(iK,jK,sK)` infers the size from the max index, which is
fine there only because `K(freedofs,freedofs)` is sliced).

---------------------------------------------------------------------------------------------------
## 4. Filters

### 4.1 HHs3D (element density filter, top3d style)
```
c = ceil(rmin) - 1                                       # window half-width (1 for rmin=1.5, 1 for rmin=2, 2 for 2.5)
for every element e1=(j1,i1,k1), every e2=(j2,i2,k2) with |j1-j2|,|i1-i2|,|k1-k2| ≤ c (clipped to the grid):
    H_full[e1,e2] = max(0, rmin - sqrt((i1-i2)^2+(j1-j2)^2+(k1-k2)^2))      # index-space distance
H  = H_full[ele][:, ele]      (nnele x nnele, symmetric)
Hs = H.sum(axis=1)            (nnele,)  -- row sums of the RESTRICTED matrix (passive neighbours excluded)
```
Preallocation is `nele*(2c+1)^2` (a `^2` typo — MATLAB grows the arrays; for c=0 the trailing/unused
entries are `(1,1,0)` and `sparse()` drops explicit zeros, so **no numerical effect**). For rmin=1.5:
weights self 1.5, face 0.5, edge `1.5-√2 = 0.08579`, corner 0; interior row sum `5.529437251522859`.
Vectorised: loop over the `(2c+1)^3` offsets, build COO triplets with `np.meshgrid`-style index arrays and
boundary masks, `scipy.sparse.coo_matrix((nele,nele))`, then `.tocsr()[ele][:,ele]`.

### 4.2 HnHns3D (element → node averaging), rnmin = 1 hard-coded
Loop bounds are asymmetric (1b): node `(jn1,in1,kn1)`, elements
`jn2 ∈ [max(jn1-ceil(r),1), min(jn1+ceil(r)-1, nely)]` etc. For `r=1` this is exactly the (up to) 8 elements
sharing the node; in 0b: node `(j,i,k)` ↔ elements `(j',i',k')`, `j'∈{j-1,j}∩[0,nely-1]`, etc.
Weight: `max(0, 1 - dist(node, element centre))` with centre at index `+0.5` → dist = `√0.75` for all 8 →
**all weights equal `w = 1-√0.75 = 0.13397459621556135`**. `Hn` is `(N, nele)` (max row index N, max column
index nele are both hit), `Hns = Hn.sum(1) = w·count`, count ∈ {1,2,4,8}. Preallocation `N*16` entries,
unused ones `(1,1,0)` → dropped. Hence `xn = Hn v / Hns` = **plain mean of the adjacent element densities**.
Vectorised equivalent (element array `V` of shape `(nely,nelx,nelz)`):
```python
P = np.zeros((nely+2, nelx+2, nelz+2)); P[1:-1,1:-1,1:-1] = V
S = sum(P[a:a+ny, b:b+nx, c:c+nz] for a in (0,1) for b in (0,1) for c in (0,1))
C = same with ones                     # 8 interior, 4 faces, 2 edges, 1 corners
xn = S / C                             # (ny, nx, nz); ulp-level difference from (w*S)/(w*C)
```
(Multiply numerator and denominator by `w` if bit-faithfulness to the dumps is wanted.)

---------------------------------------------------------------------------------------------------
## 5. smoothedge3D(vxPhys_full, ...) → (vxPhys_full', xg, ls, top, tol)

Input `vxPhys_full`: length-`nele` vector (column-major over `(nely,nelx,nelz)`), already with `MusD=1`.

1. `xn = reshape(Hn v / Hns, (ny,nx,nz))` (§4.2), nodal densities in the grid frame.
2. `xg = interp3(nodex,nodey,nodez, xn, fnx,fny,fnz,'linear')` where `nodex[j,i,k]=i, nodey=j, nodez=k`
   and `fnx[J,I,K] = I/4, fny = J/4, fnz = K/4` (exact binary fractions). So **axes are preserved**:
   `xg[J,I,K] = trilinear(xn) at (j=J/4, i=I/4, k=K/4)`, shape `(4nely+1, 4nelx+1, 4nelz+1)`. Separable
   vectorised form per axis: `fine[4j+r] = ((4-r)*A[j] + r*A[j+1])/4`, r=0..3, plus `fine[4n] = A[n]`
   (`scipy.ndimage.zoom` is NOT equivalent; use the explicit weights or `RegularGridInterpolator`).
3. Bisection for the level `ls` (exactly **17** iterations: `l1=0,l2=1; while l2-l1 > 1e-5`):
   ```
   ls = (l1+l2)/2
   xgnew = max(0.001, (tanh(β ls) + tanh(β (xg-ls))) / (tanh(β ls) + tanh(β (1-ls))))
   if mean(xgnew over the whole fine grid) - sum(vxPhys_full)/nele > 0: l1 = ls else: l2 = ls
   ```
   Target and mean are over **all** elements/fine points (passive included). `xgnew`, `ls` after the loop
   are those of the last midpoint. `top = xg - ls` (solid where `top>0`), returned `ls` = this level.
4. Element conversion, **inclusive** 5×5×5 windows (overlapping at shared faces): element `(j,i,k)` ↔ fine
   `J∈[4j,4j+4], I∈[4i,4i+4], K∈[4k,4k+4]`:
   ```
   vxPhys_out[e] = sum(window)/(ngrid+1)^3 = sum/125          (NOT a trapezoid/volume integral)
   Terr += 1  if  min(window) > 0.001  and  max(window) < 1   (strictly "grey" element)
   tol = Terr/nnele                                            (Terr counts ALL elements incl. passive)
   ```
   Vectorised: `W = sliding_window_view(xgnew, (5,5,5))[::4, ::4, ::4]` → `(nely,nelx,nelz,5,5,5)`;
   `sum/min/max` over the last three axes; flatten `order='F'`. Summation order differs from the MATLAB
   loop only at ulp level. Note `xgnew == 1` exactly iff `xg == 1` exactly (then `max<1` fails), and
   `xgnew == 0.001` wherever the Heaviside value is ≤ 0.001 (all-void regions) — the floor is why passive
   regions never count as grey and why `fvol` (§6) is slightly above the true fraction.

---------------------------------------------------------------------------------------------------
## 6. SIMP / SEMDOT loop (identical control flow; only the material law differs)

Constants: `maxloop=500, tolx=0.003, tol_thresh=3e-3, ngrid=4, rnmin=1, beta=0.1, ER=0.05, move=0.1`,
`E0 = YoungsModulus`, `Emin_SIMP = 0.001` (**absolute**, i.e. ≈0 relative to 210e9),
`Emin_SEMDOT = 0.001*E0` (relative). `KE = (aa/0.5)*lk_H8(nu) = (h/1000)*lk_H8(nu)` — `h` mm → m, so
`K = Σ E_e·h[m]·KE_unit`, `U` in m, compliance `FᵀU` in N·m.

State: `vx, vxPhys` length `nnele` (ele order), init `vx = vxPhys = vol`; `loop=0; change=1; tol=1`.

```
while change > tolx and tol > tol_thresh and loop < maxloop:
    loop += 1
    # (1) FE — uses vxPhys = ele-restricted output of the PREVIOUS smoothedge (iteration 1: vol)
    SIMP:   Ee = Emin + vxPhys**penal * (E0-Emin)
    SEMDOT: Ee = vxPhys*E0 + (1-vxPhys)*(Emin*E0)                 # linear, no penalisation
    K = assemble(KE, Ee, edofMatn) ; K = (K+K')/2 ; U[free,:] = K[free,free] \ F[free,:]
    # (2) objective / sensitivities, summed over load cases
    ce_i = rowsum((U_i[edofMatn] @ KE) * U_i[edofMatn])           # per-element uᵀKEu
    c    = Σ_i Σ_e Ee·ce_i
    SIMP:   dc = -Σ_i penal * vxPhys**(penal-1) * (E0-Emin) * ce_i
    SEMDOT: dc = -Σ_i ((1-vxPhys)*Emin + vxPhys) * E0 * ce_i        # NOT the true derivative; replicate
    dv = ones(nnele) ; cc[loop] = c
    # (3) filter both (top88 "density filter" of the sensitivities)
    dc = H @ (dc/Hs) ; dv = H @ (dv/Hs)
    # (4) OC — base point is vxPhys (the smoothedge output), NOT vx
    l1=0; l2=1e9
    while (l2-l1)/(l1+l2) > 1e-3:
        lmid = (l1+l2)/2
        vxnew = max(0, max(vxPhys-move, min(1, min(vxPhys+move, vxPhys*sqrt(-dc/dv/lmid)))))
        if sum(vxnew) > vol*nnele: l1 = lmid else: l2 = lmid
    vxPhys = (H @ vxnew)/Hs
    # (5) to full element array, keep-regions forced solid
    full = zeros(nele); full[ele] = vxPhys; full[MusD] = 1
    # (6) smooth edge
    full, xg, lss, top, tol = smoothedge3D(full, ..., beta)
    # (7) convergence measure on DESIGN variables
    change = sum(|vxnew - vx|)/(vol*nnele) ; vx = vxnew
    if change <= tolx or tol <= tol_thresh or loop >= maxloop:
        full[MusD] = 1
        full, xg, lss, top, tol = smoothedge3D(full, ..., beta)     # applied a SECOND time on its own output
    # (8) bookkeeping / print
    fvol[loop] = sum(full)/nnele                                     # full incl. passive (≥0.001 each) → slightly > true
    print('It.:%5i Obj.:%11.3f Vol.:%7.3f ch.:%7.5f Topo.:%7.5f' % (loop, cc[loop], fvol[loop], change, tol))
    if beta < 2: beta += ER
    print('Parameter beta increased to %g.' % beta)                  # printed every iteration, even when not increased
    vxPhys_full = reshape(full, (nely,nelx,nelz))  (order='F') ; vxPhys = full[ele]
S = {comp: cc[-1], finalvol: fvol[-1], eleden: vxPhys_full, gridden: xg, elenum1: nnele, elenum2: nele}
```
Notes:
* Assembly triplets (`iK = kron(edofMatn,ones(24,1))'`, `jK = kron(edofMatn,ones(1,24))'`, `sK = KE(:)*Ee`):
  entry `q = e*576 + r*24 + s` is `(row=edofMatn[e,s], col=edofMatn[e,r], val=Ee[e]*KE[s,r])`, i.e.
  `K[edof[e,s], edof[e,r]] += Ee[e]*KE[s,r]` summed over duplicates (`scipy.sparse.coo_matrix` does this).
  `K = (K+Kᵀ)/2` only enforces symmetry; keep it (cheap) for faithfulness.
* `sqrt(-dc/dv/lmid)`: dc ≤ 0 always (ce ≥ 0), so no NaN unless K is singular.
* The convergence branch's second smoothedge changes `tol`; if the branch was entered only via
  `tol <= tol_thresh` and the second call yields `tol > tol_thresh` (with `change > tolx`, `loop < maxloop`)
  the loop **continues**. Replicate.
* `xg`, `top`, `lss` returned to FreeTO.m are from the last smoothedge call (the second one on the
  terminating iteration). `S.gridden` is pre-symmetry.
* `fvol` uses `full` after smoothedge — elements in `MusD` are no longer exactly 1 (they were 1 only on input).
* `MusD` elements that are passive (outside domain) become solid in `top/xg` but are dropped from `vxPhys` for FE.
* beta schedule with OC: printed values 0.15, 0.20, …; the 38th increment (iteration 38) yields
  `2.000000000000001` (float accumulation), after which `beta < 2` is false and it stays there. Accumulate
  exactly as MATLAB does (repeated `beta = beta + ER`, test `beta < 2`) — do **not** use `0.1 + k*ER`, which
  gives a different value at iteration 38 and a different smoothedge result from iteration 39 on.
* For the contract's `callback`, `field.top` = `top` converted to (x,y,z) order (§1.4); `compliance = c`,
  `volfrac = fvol`, `change`, `topo = tol`, `beta` = value **after** the increment (as printed).

### 6.1 MMA variant (README: request `mmasub` from Svanberg; Python must implement its own)
README recipe: comment the OC block (lines 79-84) and lines 135 & 137 (the `if beta<2` guard → **beta grows
unboundedly**, `beta += ER` every iteration), uncomment 44-56 and 86-99, and set
`tolx = tol_thresh = 1e-3`, `beta = ER = 0.5`. The commented code slots in exactly where OC was:
```
m=1; n=nnele; xmin=0; xmax=1; xold1=xold2=vx; low=upp=ones(n); a0=1; a=0; c_MMA=10000; d=0
xval = vx ; f0val = c ; df0dx = dc (filtered) ; fval = sum(vxPhys)/(vol*nnele) - 1 ; dfdx = dvᵀ/(vol*nnele)
   # vxPhys here = ele-restricted smoothedge output used in the FE step (OC block is removed)
vxnew = mmasub(m,n,loop,xval,xmin,xmax,xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,a0,a,c_MMA,d)  (also updates low,upp)
vxPhys = (H @ vxnew)/Hs ; xold2 = xold1 ; xold1 = vx          # vx = vxnew happens later at step (7)
```
Then steps (5)-(8) unchanged. The signature is Svanberg's 2007 `mmasub` (asyinit 0.5, asyincr 1.2, asydecr 0.7,
albefa 0.1, raa0 1e-5, epsimin 1e-7; `move` is 1.0 in the 2007 file and 0.5 in the widely used later
distributions — make it a parameter, default 0.5). No objective scaling is applied in MATLAB; expose an
optional `mma_scale_objective` (deviation, off by default) because `c` can be ~1e-2..1e2 N·m with E=210e9.

---------------------------------------------------------------------------------------------------
## 7. Post-processing

### 7.1 symmetry(xg, ls, ...) — array space
`xg` shape `(R, C, P) = (4nely+1, 4nelx+1, 4nelz+1)` (MATLAB names them `sx,sy,sz` = rows, cols, pages!).
Uniform rule for all three planes: **`'right'` → `[xg, flip(xg,axis)]` (mirror appended at higher index),
`'left'` → `[flip(xg,axis), xg]` (mirror prepended).** Axis: `'x-y'` → pages (axis 2, z), `'y-z'` → cols
(axis 1, x), `'z-x'` → rows (axis 0, y). Unrecognised direction strings: for x-y/y-z anything ≠'left' is
right; for z-x anything ≠'right' is left. Unrecognised plane string → MATLAB error (fnx undefined).
The mirrored axis length becomes `2·(4n+1) = 8n+2`: **the boundary slice is duplicated** (an extra fine slab
of thickness h/4 sits at the mirror plane). `nel_axis *= 2`, `d_axis *= 2`; `fnx/fny/fnz` are regenerated as
`I/4, J/4, K/4` over the new shape (max = `2n + 0.25` on the mirrored axis). `top = xg - ls` recomputed
(`ls` = `lss` from the optimiser). Up to three successive calls with the updated `nelx,nely,nelz,dx,dy,dz`.

### 7.2 symmetry — physical placement (given the y-flip)
Keeping the original block at its original coordinates and using the fine spacing `h/4`:

| plane | dir | array op | mirror copy lies at | mirror plane | new origin of the (x,y,z) array |
|---|---|---|---|---|---|
| x-y | right | append pages | **+z** side | `z = z[-1] + h/8` | unchanged |
| x-y | left  | prepend pages | **−z** side | `z = z[0] − h/8`  | `z0' = z[0] − (4nelz+1)·h/4` |
| y-z | right | append cols  | **+x** side | `x = x[-1] + h/8` | unchanged |
| y-z | left  | prepend cols | **−x** side | `x = x[0] − h/8`  | `x0' = x[0] − (4nelx+1)·h/4` |
| z-x | right | append rows  | **−y** side (rows ↔ −y!) | `y = y[0] − h/8` | `y0' = y[0] − (4nely+1)·h/4` |
| z-x | left  | prepend rows | **+y** side | `y = y[-1] + h/8` | unchanged |

(`x[0], x[-1]` are the first/last node coordinates.) The mirror plane sits h/8 outside the last node because
of the duplicated slice; the copy therefore starts h/4 beyond the last node. Python: apply the same array
ops on the grid-frame `top` (before the (x,y,z) conversion) and shift `origin` per the last column; or apply
on the (x,y,z)-ordered array with the axis map `{x-y: axis 2, y-z: axis 0, z-x: axis 1}` and the
**side** map (`x-y/right → high`, `y-z/right → high`, `z-x/right → low`). README examples: air bracket `x-y,right`
(copy at +z), quadcopter `y-z,right` (+x) then `z-x,left` (+y).

### 7.3 stlgen(top, fnx, fny, fnz, dx, dy, dz, name) — MATLAB output is NOT in original coordinates
```
st_max = max(dx,dy,dz)                    # domain extents (doubled on mirrored axes)
phi   = flip(top, axis 0)                 # un-flip: rows now ascend with y → right-handed geometry
gmax  = max(fnx.max(), fny.max(), fnz.max()) = max index extent in element units (incl. +0.25 if mirrored)
gx = fnx/gmax*st_max, gy = fny/gmax*st_max, gz = fnz/gmax*st_max     # uniform scale s = st_max/gmax per element unit
phi_stl = smooth3(phi)                    # 3x3x3 box mean, boundary padded by REPLICATION; the ±1 thresholding
                                          # lines before it are dead code (overwritten)
(f1,v1) = isosurface(gx,gy,gz,phi_stl,0) ; (f2,v2) = isocaps(gx,gy,gz,phi_stl,0)   # caps enclose phi>0 ('above')
faces = [f1; f2 + len(v1)] ; vertices = [v1; v2] ; binary STL, float32, face normals from triangulation
```
So MATLAB's STL has its corner node at the **origin (0,0,0)**, scale `s = st_max/gmax` per element instead
of `h` (`s/h = del_max/(gmax·h)`: 1+0.02/(MC−1) when the longest-extent axis is the MeshControl axis, up to
`1 + h/del` otherwise, and further off with symmetry's +0.25), and is **translated by −(x[0],y[0],z[0])**.
The `nm` handling forces a `.stl` extension (splits at the first '.', so `STLs\air_brack_TO.stl` works but a
path containing a '.' directory breaks — irrelevant in Python).

**Python recommendation**: place the surface in the *original* physical frame of the input domain:
vertex of fine index `(I,J,K)` (x,y,z order) → `origin + (I,J,K)·h/4` with `origin` from §7.2. Use
`skimage.measure.marching_cubes(phi_pad, 0, spacing=(h/4,)*3)` on the volume padded by one voxel of a large
negative value (e.g. `-1e6`; crossing lands `<1e-6` voxel outside the boundary → clamp vertex coordinates to
the unpadded box, giving isocaps-equivalent flat caps), subtract one voxel and add `origin`. Apply the same
3×3×3 replicate-padded box mean first when `smooth=True` (`scipy.ndimage.uniform_filter(size=3, mode='nearest')`).
Normals: skimage's default `gradient_direction='descent'` means "object value greater than exterior", which
matches material = `phi > 0`; still verify outward orientation on a sphere test (signed volume > 0) and flip
face winding if negative. Document that MATLAB's own STL is origin-shifted and slightly rescaled.

---------------------------------------------------------------------------------------------------
## 8. intriangulation(vertices, faces, testp) → {0, 1, −1}

Not a vote: a **fallback chain** of axis-parallel ray tests (heavytest = 0 → no random rotations):
1. Rays along **z** (through `(x,y)` of each point) → `in ∈ {0,1}` and undecided list `cl`.
2. For `cl` only: rays along **x** (coords permuted `[y,z,x]`); points found inside → 1; still-undecided → `cl`.
3. For remaining: rays along **y** (`[z,x,y]`); inside → 1.
4. Remaining undecided → **−1**. A point decided "outside" by the first decisive direction stays 0.

Per ray (VOXELISEinternal, ray direction = 3rd permuted coordinate "z"):
* candidate facets: `(ty - fymin)*(fymax - ty) > 0` **and** `(tx - fxmin)*(fxmax - tx) > 0` — **strict**: a
  point whose x or y equals a facet's bbox min/max never sees that facet; facets parallel to the ray
  (zero-width projection) are never candidates.
* in-triangle test: for each edge, compare the "predicted y" of the opposite vertex and of the ray on the
  edge line (`Y1p = y2 - (y2-y3)(x2-x1)/(x2-x3)`, `YRp = y2 - (y2-y3)(x2-tx)/(x2-x3)`); passes if both above,
  both below, or `(y2-y3)*(x2-tx) == 0` (horizontal edge or ray through vertex x → counted as crossing).
  IEEE ±Inf/NaN semantics matter for vertical edges (division by zero) — replicate with `np.errstate(all='ignore')`.
* crossing z from the facet plane; `|C|<1e-14 → C=0` (→ ±Inf/NaN); keep `zmin_mesh-1e-12 ≤ z ≤ zmax_mesh+1e-12`;
  `round(z*1e10)/1e10`; unique (sort + diff ≠ 0).
* no crossings → outside (decided). Even count `2m` → inside iff `z_{2p-1} < tz < z_{2p}` **strictly** for
  some p. Odd count → undecided (goes to `cl`).

How callers treat −1: `out`/`oute` (domain): `−1 → 0` (outside/passive). Supports, forces, keepdom:
`find(>0)` → −1 = outside. `domainstokeep`: accumulated then clamped (§1.6 quirk).
**Python**: treat −1 as outside everywhere (equivalent for every caller except the §1.6 accumulation).

Vectorisation hint (big win): grid points share rays. For the z-pass all `gp_z` nodes with the same `(i,j)`
share one ray → compute the sorted crossing list once per `(x_i, y_j)` (`gp_x·gp_y` rays), then classify all
z by `searchsorted` parity/strict comparison. Same for element centres and for the x/y passes on the residual
undecided points (few). Keep the exact strictness rules; the 1e-2 shrink + `diff/2` centring mean nodes rarely
sit on domain faces, but support/force STLs are user boxes and nodes **do** land on their faces (excluded by
the strict bbox test) — this is where a "robust" replacement would disagree with the dumps.

---------------------------------------------------------------------------------------------------
## 9. Quirks / bugs and recommendations

| # | Where | Quirk | Recommendation |
|---|---|---|---|
| 1 | FreeTO.m:73 | `all([Fmagx Fmagy Fmagz])` errors only when **every** magnitude is non-zero (intended `~any`). | **Fix**: error iff all magnitudes are zero. |
| 2 | forcevec | Scalar Fmag component written to column 0 for every load region. | Replicate (README examples rely on scalar 0 defaults). Document. |
| 3 | forcevec | `nf > n_forcefiles` → index error; `nf < n_forcefiles` → silently unused files. Empty force region → index error. | Validate in `cfg.validate()` with clear messages. |
| 4 | SIMP/SEMDOT | `U = zeros(ndof, nf)` but F may have fewer columns → MATLAB size error. | **Fix**: size U from `F.shape[1]`. |
| 5 | geomeshini | MC axis chosen by max |coordinate|, not extent. | Replicate; expose `mesh_axis="auto"` override optionally. |
| 6 | geomeshini | 1 % shrink only on the MC axis; other axes centred, may have nodes exactly on domain faces. | Replicate (needed for identical membership). |
| 7 | domainstokeep | −1 cancels earlier 1 during accumulation; keep-regions tested outside the domain → passive MusD elements. | OR of booleans (fix, documented); keep passive-MusD behaviour (affects `top`/`fvol`) faithfully, or optionally intersect with `ele` behind a flag (default off). |
| 8 | SIMP | `Emin = 0.001` absolute vs SEMDOT `0.001*E0`. | Replicate both. |
| 9 | SEMDOT | `dc` is `-E(x)·ce`, not `-(E0-Emin E0)·ce`. | Replicate. |
| 10 | loop | OC base point is `vxPhys` (smoothedge output), `change` uses `vx`; second smoothedge at termination may un-terminate. | Replicate exactly. |
| 11 | HHs3D/HnHns3D | Preallocation `^2`; trailing `(1,1,0)` entries. | No effect; build sparse directly. |
| 12 | smoothedge | Average over 125 inclusive points; Terr over all elements; floor 0.001; means over passive regions. | Replicate. |
| 13 | symmetry | Duplicated boundary slice (+h/4 slab); 'right' for `z-x` is the −y side. | Replicate array op; document physical placement (§7.2). |
| 14 | stlgen | Output at origin, scale `st_max/gmax` ≠ h; dead thresholding code; `.` split of filename. | Python writes in original coordinates with true `h` (documented deviation). |
| 15 | FreeTO.m | `figure`/`display` side effects; `S.gridden` pre-symmetry; `fvol` counts passive 0.001 floors. | Callback/log instead; keep definitions. |
| 16 | intriangulation | Strict bbox test, `==0` clauses, −1 fallback. | Replicate rules; treat −1 as outside. |
| 17 | beta print | "Parameter beta increased" printed even when capped at 2. | Replicate in `log`. |
| 18 | MMA recipe | Unbounded beta growth (`beta += 0.5` per iteration) when following README. | Replicate as the documented MMA default; expose `beta_max`. |
| 19 | FreeTO.m:65-66 | `symm` keeps only the supplied SymmetryN strings but `dir` always has 3 entries, so `Symmetry2` alone pairs with `direction1`. | Contract's list of `(plane, direction)` pairs sidesteps this; note in docs. |
| 20 | geomeshini | `inp_f` drops non-char entries: force paths given as MATLAB `string` objects would be silently ignored. | N/A in Python (validate paths exist). |

---------------------------------------------------------------------------------------------------
## 10. Unit-test invariants

* `lk_H8(0.3)`: symmetric (exactly, by construction); eigenvalues ≥ −1e-14; exactly 6 zero eigenvalues (rigid
  modes); `KE[0,0] = 0.23504273504273504`; `trace = 5.6410256410256405`; `sum(KE) ≈ 0`; `KE @ t = 0` for
  translations `t[c::3]=1`; with corner coords from §2 (`(0,0,0),(1,0,0),(1,1,0),(0,1,0),(0,0,1),…`),
  `u_x = x` gives `uᵀKEu = (1-ν)/((1+ν)(1-2ν)) = 1.3461538461538463` and rotation `u=(-y, x, 0)` is in the null space.
* HHs3D, rmin=1.5, interior element: 27 stored entries incl. 8 zeros (or 19 nonzeros), row sum
  `5.529437251522859`; `H` symmetric; `Hs > 0`; restriction: `Hs` of a boundary element in the restricted
  matrix ≤ full-domain row sum.
* HnHns3D: every nonzero weight equals `1-√0.75`; counts per node ∈ {1,2,4,8}; `Hn.shape == (N, nele)`;
  `Hn @ ones(nele) / Hns == 1`.
* smoothedge on a uniform field `v` (all elements active): `xn == v`, `xg == v`; after bisection
  `|mean(xgnew) - v| ≲ β·1e-5`; if `0.001 < v < 1` every element is grey → `tol == nele/nnele == 1`.
* Tiny box domain (hand-computable): box STL `[0,10]×[0,4]×[0,4]`, `MeshControl=11` →
  x-axis chosen (`stt=[0,10,0,4,0,4]`, max at 1b position 2 → x), `ssz_old=0.01`, `x = 0.01, 1.008, …, 9.99`, `h = 0.998`,
  `y = z = [0.004, 1.002, 2.0, 2.998, 3.996]` (5 pts; `y_old(end)=3.992`, `diff=0.008`),
  `x_cen = 0.509, 1.507, …, 9.491`, `y_cen = 0.503, 1.501, 2.499, 3.497`;
  `nelx,nely,nelz = 10,4,4`, `nele=160`, `N=275`, `ndof=825`, `KE` scale `h/1000 = 0.000998`, all elements
  active (`nnele=160`), `n_vec = all 825 DOFs`. Fixing the face `x=0` (support STL box `[-1,0.5]×…`) fixes
  nodes `i=0` → 25 nodes → 75 DOFs; a distributed `Fmagz=-1` on the face `x=10` (box `[9.5,11]×…`) spreads
  `-1/25` on 25 z-DOFs; then `FᵀU > 0`, compliance decreases monotonically over the first OC iterations,
  `sum(vxnew) ≈ vol·nnele` (bisection tolerance), `0 ≤ vxnew ≤ 1`, `|vxnew - vxPhys| ≤ move`.
* forcevec: `'point'` on ids `[5,9,2]` → node 5 (sorted `[2,5,9]`, index `(3-1)//2=1`); on `[2,5]` → 2.
* Frame conversion round-trip: `xyz = transpose(A[::-1], (1,0,2))` then back equals `A`; a support STL box at
  high y must produce fixed nodes with **row index 0** in the grid frame and `y ≈ y[-1]` physically.
* symmetry: shapes `(R,C,2P)` etc.; slice `P-1` equals slice `P` for 'right' along pages; `top` unchanged on the
  original block; physical mirror-plane positions per §7.2 (test: an asymmetric blob's mirrored centroid).
* STL: closed manifold (every edge shared by exactly 2 faces), vertices within `origin ± ...` box of the
  fine grid, volume by divergence theorem ≈ `h³·sum(top>0)/64` within a few %.

---------------------------------------------------------------------------------------------------
## 11. Octave reference dumps (developer-only, not shipped; see `tests/compare_ref.py`) — most valuable comparisons, in order

1. `x, y, z, ssz, a, gpx/gpy/gpz` from geomeshini (grid construction is the root of every downstream index).
2. `oute` (nely,nelx,nelz) and `out_p`; `sup_all, sup_x/y/z, Fn` (1b node ids) — membership parity with the
   ray test; then `outeM` / `MusD`.
3. `ele, ndn, n_vec, fixeddof, freedofs` (1b) and `F` (dense, ndof×ncols) — exact integer/float equality.
4. `edofMat1, edofMat` first few rows (1b) — column order.
5. `H` (nnele×nnele) nnz pattern & `Hs`; `Hn` shape, `Hns`.
6. Iteration 1: `Ee`, `c` (cc(1)), `dc` before and after filtering, `vxnew`, `vxPhys` after filter,
   `xn, xg, lss, tol, vxPhys(after smoothedge)`, `change`, `fvol(1)`, `beta`. Expect agreement to ~1e-10
   relative (solver-dependent) for `U`, `c`; exact for index sets; `lss` exact (17 identical bisection steps)
   unless a mean lands within solver noise of the target.
7. Iterations 2–5 of the same quantities (drift check), and the final `S` fields plus `top` after symmetry
   for the air-bracket and quadcopter settings (mirror side/placement).
8. `stlgen` vertices bounding box and face count (to confirm the origin/scale statement in §7.3).

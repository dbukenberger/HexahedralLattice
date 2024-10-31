<img src='https://bukenberger.net/papers/hexahedralLattice.png' align="right" width="220">

# Stress-Aligned Hexahedral Lattice Structures
[![PDF](https://img.shields.io/badge/PDF-green)](https://bukenberger.net/pdfs/2024_bukenberger_stress-aligned_hexahedral_lattice_structures.pdf)
[![CGF Paper](https://img.shields.io/badge/DOI-10.1111/cgf.15265-blue)](https://doi.org/10.1111/cgf.15265)

With this implementation you can generate stress-aligned hexahedral lattice structures as described in the paper.

## Dependencies

Used libraries can be easily installed with `pip install -r requirements.txt`.

**Required** are
* [scikit-fem](https://github.com/kinnala/scikit-fem) for all FEM related things (requires [meshio](https://github.com/nschloe/meshio))
* [NumPy](https://github.com/numpy/numpy) for vectorized arrays,
* [SciPy](https://github.com/scipy/scipy) for (sparse) matrix operations,
* [libigl](https://github.com/libigl/libigl-python-bindings) for fast winding numbers and
* [drbutil](https://github.com/dbukenberger/drbutil) for various utilities.

**Optional** and recommended are 
* [CuPy](https://github.com/cupy/cupy) for CUDA accelerated NumPy,
* [PyPardiso](https://github.com/haasad/PyPardiso) for a parallelized CPU solver,
* [embreex](https://github.com/trimesh/embreex) for fast ray intersections,
* [trimesh](https://github.com/mikedh/trimesh) as fallback option for other methods,
* [tqdm](https://github.com/tqdm/tqdm) for progress bars,
* [Mayavi](https://github.com/enthought/mayavi) and [Polyscope](https://github.com/nmwsharp/polyscope) for nice visuals.


## Run Examples
In the main directory you can run `python runExamples.py` to generate example results.
This script contains basic setups to recreate results (2D & 3D) from the paper and an overview of the required steps.

# Usage
To create structures of your own objects you'll need a triangular (2D planar) `.obj` file or a tetrahedral (3D volume) `.msh` file as input, respectively.
Suitable tet-inputs can be generated using [fTetWild](https://github.com/wildmeshing/fTetWild) using the `--no-binary` option.

## Generating a Stress Field
Boundary conditions, i.e., fixed vertices and applied forces, can be defined in a `.frc` file using the following structure:
* First, the input file `file` (`file.obj/msh`) and mesh type `type` (`tri/quad/tet/hex`) are specified.
* Keywords `fix` and `flx` at the start of a line specify if selected vertices are either fixed or flexible (where forces are applied).
* Vertex selection has two modes `d` (dimension) and `r` (radius).
Thereby, `d` is followed by two numbers, the first (int) indicating the dimension (`0=x`, `1=y`, `2=z`) and the second (float) to specify the position of the selection plane.
This will select all vertices on the positive side of the plane, i.e., above the threshold.
For the inverse, precede the dimension with a `-`.
The radius selection is defined by a point coordinate (two floats in 2D, three in 3D) and a radius.
* The forces acting on the flexible vertices are given with the `vec` keyword as simple 2D or 3D vectors, respectively.
Multiple selections can be given different force vectors and are simply assigned by their order in the force-file.

Lets examine the included `bar2D.frc` as an example:
```python
file bar2D.obj		# the input file
type tri		# input type
fix d -0 -1		# fixed vertex selection
flx r 1 0 0.01		# flex vertex selection A
flx r 0 -0.5 0.01	# flex vertex selection B
vec 1 0			# force acting on A
vec 0 -1		# force acting on B
```
* The `-0` in the fixed vertex selection indicates that we want to select all vertices below a threshold (`-`) in x-dimension (`0`), using a threshold value of `-1`.
* Vertex selection `A` includes all vertices around a point `(1, 0)` within a radius of `0.01`.
* Vertex selection `B` includes all vertices around a point `(0, -0.5)` within a radius of `0.01`.
* Force vectors applied on selected vertices `A` and `B` are `(1, 0)` and `(0, -1)`, respectively.

Fixed and flexible vertex selections include only vertices from the objects boundary hull.
One can specify multiple force-files for the same input object, i.e., as exemplified with the included buddha.
Lines can be commented-out using `#`.
Once the forces are applied by the `FemObject.py`, the FEM simulation data is stored in a `.stress` file.

## Generating a Hexahedral Lattice
Starting from the stress field just created, we can initialize a cubification object using various parameters, as exemplified in the `CubeObject.py` file.
By default, these settings are stored in `.cfg` files, thus can be reloaded again.
Parameters are self-explanatory and described in the paper.
Resulting quad- or hex-meshes are stored in `.ply` or `.mesh` files, respectively.

## Evaluation
Optimized results from the cubification are still solid objects.
To produce actual lightweight designs, they are again loaded as `FemObject.py` using the same initial force-file but setting the `loadResult = True` flag.
Then we can materialize the object using micro-structures and evaluate the compliance of the new optimized design by reapplying the specified boundary conditions.

## Citation
You can cite the paper with:
```
@Article{bukenberger2024stress,
  author    = {Bukenberger, Dennis R. and Wang, Junpeng and Wu, Jun and Westermann, Rüdiger},
  journal   = {Computer Graphics Forum},
  title     = {{Stress-Aligned Hexahedral Lattice Structures}},
  year      = {2024},
  issn      = {1467-8659},
  pages     = {e15265},
  volume    = {43.7},
  doi       = {10.1111/cgf.15265},
  publisher = {The Eurographics Association and John Wiley & Sons Ltd.},
}
```
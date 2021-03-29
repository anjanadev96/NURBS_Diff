
# NURBS_Diff
> NURBS-Diff : A Differentiable NURBS Layer for Machine Learning CAD Applications\
> Collaborators : Aditya Balu (baditya@iastate.edu), Harshil Shah (harshil@iastate.edu)

# Requirements and Install dependencies

### VS 2017 Command Prompt or Anaconda Command Prompt


To install main dependencies for Visual Studio:
1. Open Native x64 VS 2017 command prompt
2. `conda create -n 3dlearning`
3. `conda activate 3dlearning`
4. `conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch`
5. `set DISTUTILS_USE_SDK=1 && set PY_VCRUNTIME_REDIST=No thanks && set MSSdk=1`
6.  Download and unzip CUB (latest release) from https://github.com/NVIDIA/cub/releases
7.  `set CUB_HOME=path/to/CUB/folder/containing/cmakelists.txt`
8. In WSL, edit the following:
  * `sed -i.bak -e 's/CONSTEXPR_EXCEPT_WIN_CUDA/const/g' /c/tools/miniconda3/envs/test/lib/site-packages/torch/include/torch/csrc/jit/api/module.h`
  * `sed -i.bak -e 's/return \*(this->value)/return \*((type\*)this->value)/g' /c/tools/miniconda3/envs/test/lib/site-packages/torch/include/pybind11/cast.h`
  * `sed -i.bak '/static constexpr Symbol Kind/d' /c/tools/miniconda3/envs/test/lib/site-packages/torch/include/torch/csrc/jit/ir/ir.h`
9.  `pip install "git+https://github.com/facebookresearch/pytorch3d.git"`
10. `git clone https://github.com/anjanadev96/NURBS_Diff.git`
11. `cd NURBS_Diff`
12. `python setup.py develop`
	
	
* If not already installed via the environment file install NURBS-python by:
  * `pip install geomdl`


### Ubuntu Users

1. `conda create -n 3dlearning`
2. `conda activate 3dlearning`
3. `pip install -U fvcore`
4. `pip install -U iopath`
5. `conda install -c bottler nvidiacub`
6. `pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"`
7. `git clone https://github.com/anjanadev96/NURBS_Diff.git`
8. `python setup.py develop`
    

# Examples

* Curve Evaluation (curve_eval.py)
  1. The evaluation kernels for curve_eval.py are written under torch_nurbs_eval/csrc/curve_eval.cpp
  2. To run curve_eval.py, provide input control points, input point cloud and set the number of evaluation points under out_dim in CurveEval.
	3. To generate random distribution of control points, use data_generator.gen_control_points()
	4. Input Size parameters:
	    * control points : (No of curves, no of control points, [(x,y,weights) or (x,y,z,weights)] )
	    * point cloud : (No of point clouds, no of points in point cloud,3)
	    * Parameters to vary: degree, number of control points, number of evaluation points.
	5. To run the curve evaluation, cd into torch_nurbs_eval.
	6. To run `python curve_eval.py`

(Will add details for Surface Fitting soon)
>>>>>>> 39ea163a9b1a8098982ae17358ba1ffb0c3c07dc

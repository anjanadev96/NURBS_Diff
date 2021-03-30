
## NURBS_Diff : A Differentiable NURBS Layer for Machine Learning CAD Applications


> NURBS-diff is a differentiable layer that can be run as a standalone layer for CAD applications like curve fitting, surface fitting, surface offseting, and other applications that rely on Non-uniform rational B-splines (NURBS) for representation. NURBS are the current standard for representing CAD geometries, and this work seeks to bridge the gap that currently exists between Deep Learning and Computer-Aided design.\n
> The NURBS-diff layer can also be integrated with other DL frameworks for surface reconstruction to produce accurate rational B-spline surfaces as the output. 

![alt text](https://github.com/anjanadev96/NURBS_Diff/images/layer.PNG )


> Collaborators : Aditya Balu (baditya@iastate.edu), Harshil Shah (harshil@iastate.edu)

## Install dependencies

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


### Ubuntu

1. `conda create -n 3dlearning`
2. `conda activate 3dlearning`
3. `pip install -U fvcore`
4. `pip install -U iopath`
5. `conda install -c bottler nvidiacub`
6. `pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"`
7. `git clone https://github.com/anjanadev96/NURBS_Diff.git`
8. `python setup.py develop`
    

## Examples

>Each of the examples can be run using either the CPU version of the code, or the GPU version of the code (available as 'cuda' or 'tc'). \n
> To run each of the examples, first carry out the build using setup.py. 

* Curve Evaluation (curve_fitting_on_point_clouds.py)
  * The layer can be used to fit generic 2D and 3D curves, and point clouds obtained from images.
  * To run curve_fitting_on_point_clouds.py, provide a random initialization of input control points, input point cloud and set the number of evaluation points.
  * Parameters to vary: degree, number of control points, number of evaluation points.
	


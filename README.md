
## NURBS_Diff : A Differentiable NURBS Layer for Machine Learning CAD Applications


> NURBS-diff is a differentiable layer that can be run as a standalone layer for CAD applications like curve fitting, surface fitting, surface offseting, and other applications that rely on Non-uniform rational B-splines (NURBS) for representation. NURBS are the current standard for representing CAD geometries, and this work seeks to bridge the gap that currently exists between Deep Learning and Computer-Aided design.\
> The NURBS-diff layer can also be integrated with other DL frameworks for surface reconstruction to produce accurate rational B-spline surfaces as the output. 

![alt text](https://github.com/anjanadev96/NURBS_Diff/blob/main/images/layer.PNG "NURBS_Diff layer")

> Work done at Integrated Design and Engineering Analysis Lab, Iowa State University under Prof. Adarsh Krishnamurthy.
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

### Curve Fitting 
  * Code can be found under examples/curve_fitting_on_point_clouds.py
  * The layer can be used to fit generic 2D and 3D curves, and point clouds obtained from images.
  * To run curve_fitting_on_point_clouds.py, provide a random initialization of input control points, input point cloud and set the number of evaluation points.
  * Parameters to vary: degree, number of control points, number of evaluation points.
  * Dataset used : Pixel dataset provided under Skelneton challenge.
  <img src="https://github.com/anjanadev96/NURBS_Diff/blob/main/images/curve_fitting.gif" title="Curve fitting on point clouds" width="600" height="400">
  
### Surface Fitting 
  * Code can be found under examples/{surface_fitting.py, nurbs_surface_fitting.py}
  * The layer can fit rational and NURBS surfaces.
  * Provide input control point grid, number of evaluation points in u, v direction, degree.
  <img src="https://github.com/anjanadev96/NURBS_Diff/blob/main/images/nurbs_surface_fitting.gif" title="NURBS surface fitting on ducky model" width="500" height="250">
  
### Surface Offseting
   * Code found under examples for different cases.
   <img src="https://github.com/anjanadev96/NURBS_Diff/blob/main/images/nurbs_surface_offsets.gif" title="Surface offset with C1 continuity" width="600" height="400">
   
 ### Surface reconstruction using Deep Learning
   * Splinenet architecture and dataset borrowed from ParSeNet (https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123520256.pdf)
   * Trained on 2 NVIDIA Tesla V100s.
   * Added support for rational B-splines.
   #### Non-rational B-splines
   <img src="https://github.com/anjanadev96/NURBS_Diff/blob/main/images/reconstruction_big_legend.png" title="Surface reconstruction on non-rational B-splines" width="700" height="280">\n
   #### Rational B-splines
   <img src="https://github.com/anjanadev96/NURBS_Diff/blob/main/images/reconstruction_cd10.png" width="700" title = "Surface reconstruction on rational B-splines" height="280">
   
   
 ## Will be added soon:
 * Support for trimmed NURBS surfaces
 * Support for automatically learning number of control points
 * Dataset for NURBS and trimmed NURBS surfaces



  
  


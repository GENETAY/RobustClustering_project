# RobustClustering

This is the first stable version (version 1.0.1)
Please, send me a message or an email for any questions or remark: genetay.edouard@gmail.com
or on github https://github.com/GENETAY

# Content
It is a package of robust algorithms. One can find here some new python codes, doc or examples of code that are available in R of in python with some constrains that we got rid of:
- ROBIN (initialisation procedure, see https://rdrr.io/github/brodsa/wrsk/man/ROBIN.html)
- TClust (gaussian mixture model with trimming, EM algo, see https://cran.r-project.org/web/packages/tclust/index.html)
- K-PDTM (robust EM algo based on trimming that computes distances between measures instead of between points, see https://arxiv.org/abs/1801.10346)
- trimmed_Kmeans (gaussian mixture model with trimming, EM algo, https://rdrr.io/cran/lowmemtkmeans/man/tkmeans.html)
- SMM (student mixture model, EM algo, I edited his code: https://github.com/luiscarlosgph/t-Student-Mixture-Models/blob/master/src/smm/smm.py)
- K-bMOM (a mix between kmeans and the estimator MOM (median-of-means))

One can also find a modified version of kmeans++ (sklearn code) with slight modifications. The function kmedianspp that we coded is in the same time:
- kmeans++, non robust initialisation of clustering algorithm
- kmedians++, non robust initialisation of clustering algorithm but slightly more robust than kmeans++

# Install
First, download this repository.

second, make an executable (not a `.exe` but a `.tar.gz`) from the package folder you have just downloaded with the command `python setup.py sdist` executed inside the folder.

Third, the previous step has created a file named "dist" inside the folder, now you are able to install the package with the command `pip install dist/RobustClustering-x.x.x.tar.gz` executed inside the folder and where you have adapted the "x.x.x" to your own case (for your info: "x.x.x" is simply the current version of the package.).

# how to use it
If you installed the package as described above: python is now able to import it with "import Robustclustering"

If you want to download the files and use it without installing it as a package with "pip instal...", it is also possible. Download the repository, set youself inside the folder you have just downloaded and import functions or classes you need in you python script or notebook.
/!\ It is very likely that you get an error when importing RobustClustering. I solved this for me with the following lines of code executed before "import RobustClustering"

```python
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
   
import RobustClustering
```

Check out the numerous files in `RobustClustering_project/doc` and `RobustClustering_project/examples` to see how to use all functions and classes and what they do.

# Remark
We edited existing codes or created our own programs to be able to give initial centers in the procedures. For example, trimmed-kmeans (https://rdrr.io/cran/lowmemtkmeans/man/tkmeans.html) or SMM don't have such arguments. It did matter to be able to compare performances of algorithms.

We recoded TClust because we did not find any python package to do it when we started our work in 2019. Thing may have change by now.

You fill also find in `RobustClustering_project/RobustClustering/utils.py` a lot of auxiliary functions or metrics that can help you to compare the method. We can mention RMSE (or a more resilient version RMSE_ot), accuracy but also the functions that generate datasets like those related to the KbMOM or TClust  articles.

# Further info
These files were the programs used to do the experiments of the article about K-bMOM, submitted by Adrien SAUMARD, Camille SAUMARD and Edoaurd GENETAY in 2020 in a journal, see the final published article in `RobustClustering_project/doc`.

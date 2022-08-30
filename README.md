# RobustClustering

This is the first stable version (version 1.0.1)
Please, send me a message or an email for any questions or remark: genetay.edouard@gmail.com
or on github https://github.com/GENETAY

# Content
It is a package of robust algorithms.
One can find code and doc or examples for:
- ROBIN (initialisation procedure, see https://rdrr.io/github/brodsa/wrsk/man/ROBIN.html)
- TClust (gaussian mixture model with trimming, EM algo, see https://cran.r-project.org/web/packages/tclust/index.html)
- K-PDTM (robust EM algo based on trimming that computes distances between measures instead of between points, see https://arxiv.org/abs/1801.10346)
- trimKmeans (gaussian mixture model with trimming, EM algo, https://rdrr.io/cran/lowmemtkmeans/man/tkmeans.html)
- SMM (student mixture model, EM algo, I edited his code: https://github.com/luiscarlosgph/t-Student-Mixture-Models/blob/master/src/smm/smm.py)
- K-bMOM (a mix between kmeans and the estimator MOM (median-of-means))

One can also find a modified version of kmeans++ that is kmeans++ and kmedians++ at the same time.
- kmedianspp (the same as kmeans++ where remote data are less likely to be picked up than with kmeans++, no example nor doc, cf sklearn or code here for details)

# Install
First, download this repository.
second, make an executable (not a .exe but a .tar.gz) from the package folder you have just downloaded with the command "python setup.py sdist" executed inside the folder.
Third, the previous step has created a file named "dist" inside the folder, now you are able to install the package with the command "pip install dist/RobustClustering-x.x.x.tar.gz" executed inside the folder and where you adapt the "x.x.x" to your own case.

# how to use it
If you installed the package as described above: python is now able to import it with "import Robustclustering"

If you want to download the files and use it without installing it as a package with "pip instal...", it is also possible. Download the repository, set youself inside the folder you have just downloaded and import functions or classes you need in you python script or notebook.
/!\ It is very likely that you get an error when importing RobustClustering. I solved this for me with the following lines of code executed before "import RobustClustering"

"""python==3.6
import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
   
import RobustClustering
"""

# Remark
We edited existing codes or created our own programs to be able to give initial centers in the procedure. For example, trimmed-kmeans (https://rdrr.io/cran/lowmemtkmeans/man/tkmeans.html) or SMM don't have such arguments. It did matter to be able to compare performances of algorithms.

We recoded TClust because we did not find any python package to do it when we started our work in 2019. Thing may have change by now.

# Further info
These files were the programs used to do the experiments of the article about K-bMOM, submitted by Adrien SAUMARD, Camille SAUMARD and Edoaurd GENETAY in 2020 in a journal.
- Preprint here: https://arxiv.org/abs/2002.03899
- Submitted to JASA on august 2020, rejected after review in november 2020
- Submitted to CSDA on december 2020
- accepted at CSDA on february 2021

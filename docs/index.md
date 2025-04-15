# LocScale-SURFER<br><sup>(**S**egmentation of **U**nresolved **R**egions and **F**iltering for **E**nhanced **R**epresentation)</sup>

LocScale-SURFER is a ChimeraX bundle for enhancing representation of transmembrane regions of membrane proteins. It is trained to segment voxels corresponding to the micelle belt of an unsharpened cryo-EM reconstruction. The segmented map can then be used to remove micelle densities from the target map. 

![Overview of LocScale-SURFER](./src/images/figure_1-01.png)

Note, there are two ways to speed up the computation. One is to provide a mask which restricts the prediction of detergent micelle to the region of interest. The other is to use a GPU for computation. By default, the tool uses a GPU if available. 

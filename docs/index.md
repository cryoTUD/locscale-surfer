# SURFER<br><sup>**S**egmentation of **U**nstructured **R**egions and **F**iltering for **E**nhanced **R**epresentation</sup>

SURFER is a [ChimeraX](https://www.cgl.ucsf.edu/chimerax/) bundle for segmentation of membrane, micelle and nanodisc densities in cryo-EM maps. Originally developed for enhancing visualisation and interpretation of ```LocScale```-optimised maps containing contextual structure, its segmentation procedure operates on unfiltered half maps and the resulting segmentation masks can be applied to any (post-processed) map aligned to these half maps. It currently supports the detection and targeted segmentation/removal of micellar/membranous structures, but may be extended to other contextual structures in the future. 

### What does SURFER do?

```LocScale 2.0``` is a program for cryoEM map optimisation that attemps to preserve contextual structure in the optimised maps. For visualisation, it is often useful to toggle between views including and excluding this contextual structure. For example, for membrane proteins reconstituted in detergent micelles, liposomes or nanodisks it may be useful to represent the map with and without the micelle or membrane. ```LocScale-SURFER``` allows to infer these structures from the raw map to segment and subtract these densities from the optimised map. 

<div style="display: flex; flex-direction: column; align-items: left;">
  <div class="c-compare" style="--value:50%; position: relative; width: 400px; height: 333px; overflow: hidden;">
    <img class="c-compare__left"
         src="img/emd19995_SURFER_pre.png"
         alt="Raw map"
         style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain;" />

    <img class="c-compare__right"
         src="img/emd19995_SURFER_post.png"
         alt="Feature-enhanced map"
         style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: contain; clip-path: inset(0 0 0 var(--value));" />

    <input type="range" class="c-compare__range" min="0" max="100" value="50"
           oninput="this.parentNode.style.setProperty('--value', this.value + '%')"
           style="position: absolute; bottom: 10px; left: 10px; width: 90%; z-index: 10;" />
  </div>
</div>

### How do I use it?

`LocScale-SURFER` is available as a ChimeraX bundle. Just download and [install](install.md) it and follow the [instructions](tutorial.md). While LocScale-SURFER has been designed to work with `LocScale` feature-enhanced maps, it has been trained on unmodified maps and its segmentation and removal procedure should therefore also work for both unmodified maps or maps obtained with other post-processing software.

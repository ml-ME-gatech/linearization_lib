### Overview
There are there are three example jupyter notebooks explaining the notions of: 

(1) node pairing, to create the stress classification lines across a body between two surfaces
(2) volume interpolation, to interpolate the solution results to the integration points
(3) stress linearization, to compute the linearized stresses in the volumes

The powerpoint walks through some simple validation against both analytical solutions and against the APDL 
computation

### Dependencies
scipy,pandas, numpy, pathlib, and matplotlib are all you need outside the standard library. 

### Concepts
To know precisely what the code is doing, you must have a good understanding of the following concepts: 

- numpy broadcasting rules and array shaping
- stress linearization 
- graphs (computer science)
- linear sum assignment optimization
- multidimensional interpolation

you are welcome to simply use the classes at a high level in the early stages, as demonstrated in the example jupyter notebooks

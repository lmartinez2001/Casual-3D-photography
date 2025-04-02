<p align="center">
    <img src="images/logo.png" alt="logo" width="700">
</p>

This repo is an **unofficial implementation** of some key steps of the method introduced by Hedman et al. in **Casual 3D photography**. Their method reconstructs a 3D photo: i.e. a two-layered panoramic mesh with reconstructed surface color, depth, and normals, from casually captured smartphone or DSLR images, which can be rendered and manipulated with geometric awareness. In particular, we focus on depth maps reconstruction, stitching and two-layer fusion algorithms.

> [!WARNING]  
> The code of this repo results from our interpretation of the original article, and for academic purpose. Therefore, some parts might be inaccurate or wrong. Overall, this code isn't intended to be production-ready.

*Authors:*
- [Louis Martinez](https://github.com/lmartinez2001/) - ENS Paris Saclay (MVA)
- [Mohamed Ali Srir](https://github.com/metlouf/) - Dauphine PSL (IASD)

## How to run the code

### Downloading the dataset

We carried our experiments on the CreepyAttic dataset, provided with the supplementary material of the paper. We therefore recommend to use the same one to avoid redefining some hardcoded paths. To download it, run the following command

```bash
bash download_dataset.sh
```

This will download and unzip the dataset. Ultimately, a directory called `Volumes` is located at the root of the project.

### Executing the code

**Stitching**: Execute scripts in the following order  
- depth_map_to_pcd.py


## Technical details

Please refer to our [report](report.pdf) 
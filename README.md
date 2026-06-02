
# DS6050_G12_PROJECT

Repository for uvaMSDS 6015-006 Tree Canopy Capstone Project

Enter some project details here

## Running on UVA Rivanna

Download the environment setup script into your home directory

#```bash
#  curl -L https://raw.githubusercontent.com/d26clarke/DS6050_G12_PROJECT/main/scripts/setup_G12_proj_env.sh -o setup_G12_proj_env.sh
#  chmod 755 setup_G12_proj_env.sh
#```

#Execute the environment setup script e.g.: Usage: ./setup_G12_proj_env.sh <your selected environment: dev | sit | prod >

#```bash
#  ./setup_G12_proj_env.sh sit
#```

## Usage/Examples

Run single slurm to generate tiff files and centroids
```bash
  sbatch slurm/run_single.slurm 
```



## Screenshots

Per Class F1 Scores

![App Screenshot](https://github.com/d26clarke/DS6050_G12_PROJECT/blob/main/media/per-class%20f1%20score.png)


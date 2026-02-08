## <div align="center"> A High-Fidelity Synthetic GPS Trajectory Dataset via Road Network-Constrained Generative Modeling </div>

This is the official implementation of paper "A High-Fidelity Synthetic GPS Trajectory Dataset via Road Network-Constrained Generative Modeling".

If you find this repository helpful, please kindly leave a star here.

### Requirements

The required packages with Python environment is:
```
torch
torch_geometric
tqdm
PyYAML
numpy
pandas
scikit-learn
shapely
tensorboard
haversine
loguru
```

### Running

* Data Preprocessing

  Next, We use [KaHIP](https://github.com/KaHIP/KaHIP), a graph partitioning framework, to partition the road network. Install KaHIP by running the following commands in your terminal:

  ```console
  git clone --branch v3.17 https://github.com/KaHIP/KaHIP.git
  cd KaHIP
  mkdir build
  cd build 
  cmake ../ -DCMAKE_BUILD_TYPE=Release     
  make
  cd ../..
  ```

  Finally, run our script to preprocess the data:

  ```console
  cd data/preprocess
  python partition_road_network.py
  python get_zone_trans_mat.py
  cd ../..
  ```

* Model Training

  `python train.py`
  * `--dataset` sppython gene.py --dataset Tongzhou --seed 0 --cuda 3 --num_gene 100 --processes 2ecifies the dataset, such as `Tongzhou`
  * `--seed` specifies the random seed
  * `--cuda` specifies the GPU device number

* Trajectory Generation

  `python gene.py`
  * `--dataset` specifies the dataset, such as `Tongzhou`
  * `--seed` specifies the random seed
  * `--cuda` specifies the GPU device number
  * `--num_gene` specifies the number of trajectories to generate
  * `--processes` specifies the number of processes to use when generating trajectories in parallel

* Model Evaluation

  Please refer to `evaluation/main.ipynb`.


  


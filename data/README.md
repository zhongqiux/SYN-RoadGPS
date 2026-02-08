### Data Processing
The original dataset is a shared bike dataset sourced from China Mobile, consisting of GPS locations uploaded by shared bikes. Each location record includes `id`, `time`, `lat`, `lon`, along with other privacy-related information.
- First, anonymization is applied to the original dataset, retaining only the four columns mentioned above.
- Trajectories are then segmented based on time intervals and distances.
- The segmented trajectories are map-matched to the road network using an HMM method.

The resulting processed data includes `mm_id, entity_id, traj_id, rid_list, time_list, rate_list, gps_list`, where `rid_list` represents the road segment-level trajectory, `rate_list` indicates the proportion traveled on each road segment, and `gps_list` contains the original GPS trajectory. For details, please refer to task_data.csv.

Next, we use [KaHIP](https://github.com/KaHIP/KaHIP), a graph partitioning framework, to partition the road network. Install KaHIP by running the following commands in your terminal:

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

After completing the steps above, the processing of both the road network data and the raw trajectory data is finished. The structure of the required original road network data can be referenced from `roadmap.rel` and `roadmap.geo` in the `data/Tongzhou` directory.

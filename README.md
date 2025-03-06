
Run ClusterGAN on ROCK
To run ClusterGAN on the ROCK dataset, ensure the package is setup and then run:
python rock_clu.py -r test_run -w -n 210 -b 4 -s my_dataset


Generated Examples
To generate examples from randomly sampled latent space variables,
python gen-examples.py -r ./runs/my_dataset/20quan-bck400s-1e-4-210epoch_z64_wass_bs4_test_run -b 100

TSNE Figure
To produce a TSNE figure depicting the clustering of the latent space encoding of real images,
python t_sne_result_cluster.py -r ./runs/my_dataset/20quan-c600s-1e-4-210epoch_z64_wass_bs4_test_run -p 30 -n 1000

Result analysis
Python euler_function.py



In this study, we referred to the research work of Sudipto Mukherjee et al. and their provided code, which provided important insights and technical support for our research. We sincerely appreciate their contributions to this field.

References
<br>
Mukherjee S, Asnani H, Lin E, et al. ClusterGAN: Latent space clustering in generative adversarial networks[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2019, 33(01): 4610-4617.

Run ClusterGAN
<br>
To run ClusterGAN on the dataset, ensure the package is setup and then run:
<br>
python rock_clu.py -r test_run -w -n 210 -b 4 -s my_dataset


Generated Examples
<br>
To generate examples from randomly sampled latent space variables,
<br>
python gen-examples.py -r ./runs/my_dataset/20quan-bck400s-1e-4-210epoch_z64_wass_bs4_test_run -b 100

TSNE Figure
<br>
To produce a TSNE figure depicting the clustering of the latent space encoding of real images,
<br>
python t_sne_result_cluster.py -r ./runs/my_dataset/20quan-c600s-1e-4-210epoch_z64_wass_bs4_test_run -p 30 -n 1000

Result analysis
<br>
Python euler_function.py



# Keyphrase-Generation-GANs
Contains code for generating key phrases using GANs

## Instructions 
Download the following <a href="https://drive.google.com/drive/folders/1YIJOAAR8rK8oiAfPK-5aJwgwlmw0uie_?usp=sharing"> data folder </a> and <a href="https://drive.google.com/drive/folders/1jGLB30qPVh7q-ozbSL5ye_ZLtj5jfDiy?usp=sharing-"> Model checkpoints folder </a> and unzip it and insert it in this repo in your local system.The Data Folder contains around 2000 samples from the kp20k data folder . In order to train the Discriminator run hierarchal_attention_Discriminator_training.py . All the Discriminator checkpoints will be created in the folder Discriminator_checkpts . You can test the strength of the Dicriminator by running hierarchal_attention_Discriminator_training_f1score.py . Currently the program supports running on cpu.In order to run it on device of your choice change the devices variable in each python file


<ol>

<li> <b> Discriminator_individual.py : </b> A Discriminator which only identifies keyphrases as human readable or not human readable.</li>

<li> <b> Discriminator_training_individual.py: </b> Training a Discriminator which only identifies keyphrases as human readable or not human readable.</li>

<li> <b> Discriminator_training.py: </b> Training a Discriminator which assigns a score to keyphrase indicating whether it can be assigned to an abstract. </li>

<li> <b> Discriminator.py : </b> a Discriminator which assigns a score to keyphrase indicating whether it can be assigned to an abstract </li>

<li> <b>RLtraining(with 2 Discriminators).py : </b> The reward is calculated as adding the scores and then  from 2 discriminators we use the reward for reinforcement learning. </li>

<li> <b> RLtraining1_individual.py : </b> The reward is calculated as taking reward from the Discriminator_individual only </li>

<li> <b> RLtraining(with 2 Discriminators and f1 scores).py : </b> The reward is calculated by adding rewards from both the discriminators and also adding the f1 score of the generated keyphrase string. </li>
</ol>

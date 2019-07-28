# Keyphrase-Generation-GANs
Contains code for generating key phrases using GANs

<ol>

<li> <b> Discriminator_individual.py : </b> A Discriminator which only identifies keyphrases as human readable or not human readable.</li>

<li> <b> Discriminator_training_individual.py: </b> Training a Discriminator which only identifies keyphrases as human readable or not human readable.</li>

<li> <b> Discriminator_training.py: </b> Training a Discriminator which assigns a score to keyphrase indicating whether it can be assigned to an abstract. </li>

<li> <b> Discriminator.py : </b> a Discriminator which assigns a score to keyphrase indicating whether it can be assigned to an abstract </li>

<li> <b>RLtraining(with 2 Discriminators).py : </b> The reward is calculated as adding the scores and then  from 2 discriminators we use the reward for reinforcement learning. </li>

<li> <b> RLtraining1_individual.py : </b> The reward is calculated as taking reward from the Discriminator_individual only </li>

<li> <b> RLtraining(with 2 Discriminators and f1 scores).py : </b> The reward is calculated by adding rewards from both the discriminators and also adding the f1 score of the generated keyphrase string. </li>
</ol>

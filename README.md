# NeuralSB
NeuralSB is a generalizable framework for boundary testing of ML / DL models. 
Understanding a classifiers behavior in various situations is important in many domains such as automated driving and many more.
To test for boundaries we need a conditional StyleGAN pretrained on a dataset of similar domain. 
It is important that it is conditional and as such can generate images based on class information.

The framework consists of three distinct components:

1) The `Predictor`, which is the ml model to be tested.
2) The `Generator`, which is a conditional StyleGAN3 of specific domain.
3) The `Learner`, which is any optimization algorithm, to optimize for boundary discovery.

These components are modular, as such we are not restricted to images, we are also able to quickly adapt the optimization strategy based on individual needs.
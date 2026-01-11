# Pytorch-experimentation
A series of Jupyter notebook explaining the basics of Pytorch, with some sporadic mathematical context. Right now we have, in this order:
1. *Regression.ipynb* : Introduction to PyTorch by building a simple polynomial regression for synthetic data. We also look at how to do the same with sklearn. 
2. *MNISTclassifier.ipynb* : We construct a basic image classifier and introduce concepts like convolutional layers and cross-entropy loss.
3. *Time_Series_Introduction.ipynb* : We start looking at time series models. We construct a simple neural network and test it on synthetic data coming from a Gaussian process. We also provide a comparison with a classical AR(2) model.
4. *RNN.ipynb*: We cover the basics of recurrent neural networks, as well as miscellaneous skills like how to get real datasets, or how to manage data with pandas. We then go on to explain two popular extension of the RNN model: the LSTM and the GRU. This is the basis for more complex models like LTSNet. 
5. *papers-mini-rag* : This is not a notebook like the rest of the experiments. It's a mini-rag designed to get the latest papers in ML so that when you ask about an ML topic, it brings up recent, relevant papers. In practice it has a very small vector database because I have chunked whole papers. In the future it would be nice to just extract the introductions of many papers instead. It requires to have locally ollama.

Notebooks to come:
5. *Transformer_for_Time_Series.ipynb* : We start looking at models based on the transformer architecture.
6. *Introduction_to_NLP.ipynb* : Still in the air. We won't be training LLM from scratch but might look at "applicable" skills like LLM fine-tuning. 
7. *Image_generation.ipynb* : Still in the air. We wish to give an introduction to the three main image-generation models: VAEs, GANs and diffusion models. 
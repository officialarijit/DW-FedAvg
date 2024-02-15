# A Dynamic Weighted Federated Learning for Android Malware Classification

![Github Viweres](https://komarev.com/ghpvc/?username=officialarijit&label=Views&color=brightgreen&style=flat-square)


`ABSTRACT` 
Android malware attacks are increasing daily at a tremendous volume, making Android users more vulnerable to cyber-attacks. Researchers have developed many machine learning (ML)/ deep learning (DL) techniques to detect and mitigate android malware attacks. However, due to technological advancement, there is a rise in android mobile devices. Furthermore, the devices are geographically dispersed, resulting in distributed data. In such scenario, traditional ML/DL techniques are infeasible since all of these approaches require the data to be kept in a central system; this may provide a problem for user privacy because of the massive proliferation of Android mobile devices; putting the data in a central system creates an overhead. Also, the traditional ML/DL-based android malware classification techniques are not scalable. Researchers have proposed federated learning(FL) based android malware classification system to solve the privacy preservation and scalability with high classification performance. In traditional FL, Federated Averaging (FedAvg) is utilized to construct the global model at each round by merging all of the local models obtained from all of the customers that participated in the FL. However, the conventional FedAvg has a disadvantage: if one poor-performing local model is included in global model development for each round, it may result in an under-performing global model. Because FedAvg favors all local models equally when averaging. To address this issue, our main objective in this work is to design a dynamic weighted federated averaging (DW-FedAvg) strategy in which the weights for each local model are automatically updated based on their performance at the client. The DW-FedAvg is evaluated using two popular benchmark datasets, Melgenome and Darbin used in android malware classification research. The results show that our proposed approach is scalable, privacy preserved, and capable of outperforming the traditional FedAvg for android malware classification in terms of accuracy.


## NOTE*: Please feel free to use the code by giving proper citation and star to this repository.


## Installation: 


**DATA Rearrangement required**
```
- Drebin data --> 'data/drebin.csv'
- Malgenome data --> 'data/malgenome.csv'
- Kronodroid data --> 'data/kronodroid.csv'
- TUANDROMD data --> 'data/TUANDROMD.csv'
```

- Programming language
  - `Python 3.10`

- Operating system
  - `Ubuntu 18.04 (64 bit)` 

- Required packages
  - `Keras 2.10` 
  - `Tensorflow 2.10` &#8592; for developing the `neural network`.
  - `Scikit-Learn` &#8592; for model's performance matrics. 
  - `pandas`
  - `numpy`

  
- Installation steps:
  - Step 1: Install `Anaconda`. 
  - Step 2: Create a `virtual environment` in `Anaconnda` and install required packages from the given `requirements.txt` file.
  - Step 3: `Run the notebooks provided`. 
  - Step 4: Enjoy the results :wink:.



# Cite this work: 
```
	Chaudhuri, A., Nandi, A., Pradhan, B. (2023). 
	A Dynamic Weighted Federated Learning for Android Malware Classification. 
	In Soft Computing: Theories and Applications. 
	Lecture Notes in Networks and Systems, vol 627. Springer, Singapore. 
	https://doi-org./10.1007/978-981-19-9858-4_13
```
`OR`

`BIBTEX`

    @InProceedings{dw-fedavg,
		author="Chaudhuri, Ayushi
		and Nandi, Arijit
		and Pradhan, Buddhadeb",
		title="A Dynamic Weighted Federated Learning for Android Malware Classification",
		booktitle="Soft Computing: Theories and Applications",
		year="2023",
		publisher="Springer Nature Singapore",
		address="Singapore",
		pages="147--159",
		isbn="978-981-19-9858-4"
	}


# Published paper (Soft Computing: Theories and Applications):

https://link.springer.com/chapter/10.1007/978-981-19-9858-4_13

# Pre-print version on Arxiv:
https://arxiv.org/abs/2211.12874


## 📝 License

Copyright © [Arijit](https://github.com/officialarijit).
This project is MIT licensed.



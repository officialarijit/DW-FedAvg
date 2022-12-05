# DW-FedAvg: A Dynamic Weighted Federated Learning for Android Malware Classification

`ABSTRACT` 
Android malware attacks are increasing daily at a tremendous volume, making Android users more vulnerable to cyber-attacks. Researchers have developed many machine learning (ML)/ deep learning (DL) techniques to detect and mitigate android malware attacks. However, due to technological advancement, there is a rise in android mobile devices. Furthermore, the devices are geographically dispersed, resulting in distributed data. In such scenario, traditional ML/DL techniques are infeasible since all of these approaches require the data to be kept in a central system; this may provide a problem for user privacy because of the massive proliferation of Android mobile devices; putting the data in a central system creates an overhead. Also, the traditional ML/DL-based android malware classification techniques are not scalable. Researchers have proposed federated learning(FL) based android malware classification system to solve the privacy preservation and scalability with high classification performance. In traditional FL, Federated Averaging (FedAvg) is utilized to construct the global model at each round by merging all of the local models obtained from all of the customers that participated in the FL. However, the conventional FedAvg has a disadvantage: if one poor-performing local model is included in global model development for each round, it may result in an under-performing global model. Because FedAvg favors all local models equally when averaging. To address this issue, our main objective in this work is to design a dynamic weighted federated averaging (DW-FedAvg) strategy in which the weights for each local model are automatically updated based on their performance at the client. The DW-FedAvg is evaluated using two popular benchmark datasets, Melgenome and Darbin used in android malware classification research. The results show that our proposed approach is scalable, privacy preserved, and capable of outperforming the traditional FedAvg for android malware classification in terms of accuracy.

# Preprint ia available in ArXiv:
https://arxiv.org/abs/2211.12874

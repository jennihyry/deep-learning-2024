This is the final project for Deep Learning course in 2024. The idea was to train a pretrained ResNet18 using
miniImageNet data and then fine-tuning the model with EuroSAT dataset. The first dataset contains
natural images and the latter Copernicus satellite data. 

'train_resnet_miniimagenet.py' contains the script for training the ImageNet pretrained ResNet18 model.
It saves the trained model to path "./state_dict_model.pt". 'finetune_eurosat.py' contains the script for fine tuning. It uses state_dict_model.pt file. 
These should work on both CPU-only and CUDA-enabled PCs.

Paths to EuroSAT data and the pretrained model are defined in the beginning of script:

	# CHANGE THESE PATHS IF NECESSARY
	eurosat_local_path = './EuroSAT/2750'
	pretrained_model_path = './state_dict_model.pt'

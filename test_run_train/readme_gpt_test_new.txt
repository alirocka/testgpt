first we install the nvidia toolkit:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-4-local_12.4.0-550.54.14-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4

then the driver:
sudo apt-get install -y nvidia-driver-550-open
sudo apt-get install -y cuda-drivers-550

command for checking cuda version:
nvcc --version

then we install pytorch:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

command to check if both torch and cuda is available :
python -c "import torch; print(torch.cuda.is_available())"

if we get true it's all good and ready to go
if false torch is installed but cuda isnt available
if we get a traceback error torch has a problem
and transformers library:
pip install transformers

command to check if transformers is installed:
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"

and install scikit learn library:
pip3 install -U scikit-learn
and scipy library:
python -m pip install scipy

then we can run the code :)
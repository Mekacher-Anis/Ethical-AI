### This script sets up an conda env that can be used on the jupyterhub
module load Miniforge3
# create a new conda env
conda create --name eai-24 python=3.10
# activate your newly generated env
eval "$(conda shell.bash hook)"
conda activate eai-24
# install the jupyter kernel
conda install ipykernel
make-ipykernel --name eai24_env --display-name "eai-24"
# mv diretories that could get big to $BIGWORK
mv .conda $BIGWORK
mv .cache $BIGWORK
ln -s /bigwork/$USER/.cache/ .cache
ln -s /bigwork/$USER/.conda/ .conda
# add access to $BIGWORK
ln -s /bigwork/$USER/ bigwork
# automatically activate env when login to the cluster
echo "module load Miniforge3" >> .profile
echo "conda activate eai-24" >> .profile

conda create -n fitdit python=3.10
conda activate fitdit
pip install -r requirements.txt

yum install -y git-lfs
git lfs install

# clone models to models/
git clone https://huggingface.co/BoyuanJiang/FitDiT models/
conda env create -n environment.yml
conda init bash
source ~/.bashrc
conda activate environment.yml
pip install hnswlib
sudo apt --yes install bc git
git clone https://github.com/kunaldahiya/pyxclib.git
cd pyxclib
python setup.py install --user
echo "the pwd is : $PWD"
cd ..
bash run.sh /mnt/my_storage/DLTSEastUSBackup/GraphXML/data/$1/ $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14} ${15} | tee /mnt/my_storage/DLTSEastUSBackup/GraphXML/Logs/$1/`cat /dev/urandom | tr -cd 'a-f0-9' | head -c 32`.txt

# Instructions for running COVIDNet on Cori@NERSC

## Setup workflow enviroment

In the following instructions, ${PROJ_DIR} points to the directory of COVIDNet source code  

1. Setup Python environment

Create an Anaconda environment
```
cd ${PROJ_DIR}
source env/init.sh
conda create --name covidnet_env python=3.7
echo 'source activate covidnet_env' >> env/init.sh
echo 'export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/lib64:/usr/lib64:$HOME/.conda/envs/covidnet_env/lib"' >> env/init.sh
source env/init.sh
```

Install neccessary Python packages
```
conda install -c anaconda scikit-learn
conda install pytorch torchvision -c pytorch
conda install -c conda-forge optuna
...
```
Keep installing other corresponding packages if there are error message of not found somethings. 


2.1. To run distributed Optuna using a database, it is required to install MySQL-related Python packages. Otherwise, please skip this step
```
conda install mysqlclient pymysql
```

2.2. In order to run the workflow version using Decaf, it is mandatory to install Decaf. Otherwise, please skip this step

Install mpi4py 
```
wget https://bitbucket.org/mpi4py/mpi4py/downloads/mpi4py-3.0.3.tar.gz
tar zxvf mpi4py-3.0.3.tar.gz
cd mpi4py-3.0.3
CC=$(which cc) CXX=$(which CC) FC=$(which ftn) python setup.py build --mpicc=$(which cc)
CC=$(which cc) CXX=$(which CC) FC=$(which ftn) python setup.py install
```

Assume that Decaf will be installed to ${DECAF_PREFIX}. Please change ${DECAF_PREFIX} to your desired installation directory
```
git clone https://github.com/tpeterka/decaf.git 
cd decaf
mkdir build
CC=$(which cc) CXX=$(which CC) FC=$(which ftn) cmake .. -DCMAKE_INSTALL_PREFIX=${DECAF_PREFIX} -Dtransport_mpi=on
make -j 8
make install
```

Export Decaf environment variable to env/init.sh
```
cd ${PROJ_DIR}
echo 'export DECAF_PREFIX="${HOME}/software/install/decaf"' >> env/init.sh
echo 'export LD_LIBRARY_PATH="${DECAF_PREFIX}/lib:${LD_LIBRARY_PATH}" >> env/init.sh
```

Copy generated Python binding shared libraries to project directory
```
cp ${DECAF_PREFIX}/examples/python/*.so ${PROJ_DIR}
```

## Run the workflow for testing/debugging

1. Download the datasets
```
sh download.sh
unzip datasets.zip
```

2. It is recommended to create an interactive job required a node in debug queue to test the workflow 
```
salloc -q debug -C haswell -N 1 -t 00:30:00
```

3. Run the workflow having 2 workers, each worker runs on 32 cores (2 workers take a full Cori node) and explores 5 trials. 
```
chmod +x study
./study delete
./study create
sh cori.sh 2 32 5
```

## Submit jobs

- Directory structure  
        
```bash
├── datasets
├── db
│   ├── run
│   └── ...
├── decaf
│   ├── run
│   └── ...
└── main.sh
```


- How to submit

```
sbatch main.sh ${NWORKERS}
```

- References: [run](run), [main.sh](main.sh)

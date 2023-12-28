# README : Racing-CVGP #

## 0. Prequisites

### 0.1 Dependency packages

```bash
pip install deap
pip install Cython
pip install cryptography
```

### 0.2 Directory

- data: the generated dataset. Every file represent a ground-truth expression.
- racing-cvgp: implementation of the proposed method
- gp_cvgp: implementation of GP and CVGP. public code implementation from  https://github.com/jiangnanhugo/cvgp
- dso-classic: public code implementation from https://github.com/brendenpetersen/deep-symbolic-optimization.
- eureqa: impelemtation and result of Eureqa method
- scibench: the DataOracle 

- plots: the jupter notebook to generate our figures in the paper.
- result: contains all the output of all the programs, the training logs.



## 1. run Racing-CVGP
Run the **racing-CVGP** model on the **sincos** datasets.

```bash
./racing_cvgp/scripts/trigometric/run_onekey.sh

```

Run the **racing-CVGP** model on the **Livermore2** datasets.
```bash
./racing_cvgp/scripts/livermore2/run_onekey.sh

```


## 2. run GP or CVGP



Run the **CVGP, GP** model on the **feynman** datasets.

```bash
./gp_cvgp/scripts/feynman/run_gp_cvgp.bell.sh

```


## 3. Run DSR, PQT, VPG, GPMeld

### 3.1 prequisites

1. install python environment 3.6.13: `conda create -n py3613 python=3.6.13`.
2. use the enviorment `conda env py3613`.
3. install `dso`

```cmd
cd ./dso_classic
pip install --upgrade setuptools pip
export CFLAGS="-I $(python -c "import numpy; print(numpy.get_include())") $CFLAGS"
pip install -e ./dso
```
`
### 3.2 run

 run DSR, PQT, VPG, GPMeld models by
   If you want to run DSR, PQT, VPG, GPMeld

```bash
dso_classic/scripts/run_dso_series.bell.sh
dso_classic/scripts/feynman.vars5.bell.sh
```


## 4. Look at the summarized result

Just open the `result/plots` folder




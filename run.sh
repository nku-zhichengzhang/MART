expname="rVE8_L0.3_seq12_lr1_e100_pcce_MBT_b1_a36_e400_1e-4_Ve_pal_clsrec_test_sep_att_adaMratio_cross_lanAtt"
savedir="results/"
GPU_ID="0"
echo ${expname}
echo ${GPU_ID}

path=${savedir}${expname}"/"
if [ ! -d ${path} ];then
  mkdir ${path}
  echo 'create filepath'
  else
  echo "filepath already exits"
fi

main_py="main.py"
opt_py="opts.py"
train_py="train.py"
validation_py="validation.py"
mae_py="MART.py"

cp ${opt_py} ${path}${opt_py}
cp ${train_py} ${path}${train_py}
cp ${validation_py} ${path}${validation_py}
cp ${main_py} ${path}${main_py}
cp ${mae_py} ${path}${mae_py}

resultdir=${path}'result.txt'
if [ ! -f ${resultdir} ];then
  touch ${resultdir}
  echo "create result.txt"
  else
  echo "result.txt already exits"
fi
export CUDA_VISIBLE_DEVICES=${GPU_ID}
nohup python ${main_py} --exp_name ${path} > ${resultdir}
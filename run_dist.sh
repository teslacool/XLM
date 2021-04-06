DATA_DIR=$1
RESULT_DIR=$2

echo '***************here****************'
echo 'OMPI_COMM_WORLD_SIZE'
echo $OMPI_COMM_WORLD_SIZE
echo 'OMPI_COMM_LOCAL_WORLD_SIZE'
echo $OMPI_COMM_LOCAL_WORLD_SIZE
echo 'OMPI_COMM_WORLD_RANK'
echo $OMPI_COMM_WORLD_RANK
echo 'OMPI_COMM_WORLD_LOCAL_RANK'
echo $OMPI_COMM_WORLD_LOCAL_RANK
echo 'MASTER_ADDR'
echo $MASTER_ADDR
echo 'MASTER_PORT'
echo $MASTER_PORT
echo 'AZ_BATCHAI_MPI_MASTER_NODE'
echo $AZ_BATCHAI_MPI_MASTER_NODE
echo 'OMPI_MCA_orte_local_daemon_uri'
echo $OMPI_MCA_orte_local_daemon_uri
echo 'WORLD_SIZE'
echo $WORLD_SIZE
echo 'NGPU'
echo $NGPU
echo 'pwd'
pwd

echo 'show dist_url'
echo "tcp://"
echo $MASTER_ADDR
echo $MASTER_PORT

ompi_info --parsable --all | grep mpi_built_with_cuda_support:value

python -W ignore -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py \
	--exp_name 20210403-test-multinode \
	--dump_path $RESULT_DIR \
	--data_path $DATA_DIR \
	--lgs 'sw-ur-vi' \
	--clm_steps '' \
	--mlm_steps 'sw,ur,vi' \
	--emb_dim 1024 \
	--n_layers 12 \
	--n_heads 8 \
	--dropout 0.1 \
	--attention_dropout 0.1 \
	--gelu_activation true \
	--batch_size 16 \
	--bptt 256 \
	--lg_sampling_factor 0.7 \
	--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001,weight_decay=0 \
	--epoch_size 200000 \
	--save_periodic 1 \
	--max_vocab 95000 \
	--validation_metrics _valid_mlm_ppl \
	--stopping_criterion _valid_mlm_ppl,10 > $RESULT_DIR/nothing_log.txt
	# --accumulate_gradients 4 \
	# --split_data True
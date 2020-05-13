#!/bin/bash
##########temporal##############
##########################
#1make groundtruth for the first actor network in 1st iteration time 

##########generate features for different temporal sequences##########################
# python  0831test_4d_mpb.py --use_dense --gpu_ids 2 --name personlowrankvliddensnet   --crossname dqn100time20vilid \
    # --test_dir /home/songwenfeng/swf/pytorch/data/PRW/PRW_pytorch/  --which_epoch  199  --classnum 1600 \
    # --lowrankdir /home/songwenfeng/swf/pytorch/data/PRW/rgbmasktemporalcontext/frame_out/20/
# python  0831test_4d_mpb.py --use_dense --gpu_ids 2 --name personlowrankvliddensnet   --crossname dqnmarstime100vilid \
    # --test_dir /home/songwenfeng/swf/pytorch/data/PRW/PRW_pytorch/  --which_epoch  199  --classnum 1600 \
    # --lowrankdir /home/songwenfeng/swf/pytorch/data/PRW/rgbmasktemporalcontext/frame_out/100/
# python  0831test_4d_mpb.py --use_dense --gpu_ids 2 --name personlowrankvliddensnet   --crossname dqnmarstime1000vilid \
    # --test_dir /home/songwenfeng/swf/pytorch/data/PRW/PRW_pytorch/  --which_epoch  199  --classnum 1600 \
    # --lowrankdir /home/songwenfeng/swf/pytorch/data/PRW/rgbmasktemporalcontext/frame_out/1000/
# python  0831test_4d_mpb.py --use_dense --gpu_ids 2 --name personlowrankvliddensnet   --crossname dqn100time50vilid \
    # --test_dir /home/songwenfeng/swf/pytorch/data/PRW/PRW_pytorch/  --which_epoch  199  --classnum 1600 \
    # --lowrankdir /home/songwenfeng/swf/pytorch/data/PRW/rgbmasktemporalcontext/frame_out/50/
# python  0831test_4d_mpb.py --use_dense --gpu_ids 2 --name personlowrankvliddensnet   --crossname dqn100time20vilid \
    # --test_dir /home/songwenfeng/swf/pytorch/data/i-LIDS-VID/pytorch/  --which_epoch  199  --classnum 1600 \
    # --lowrankdir /home/songwenfeng/swf/pytorch/data/frameouts/ilids/frame_out/20
# python  0831test_4d_mpb.py --use_dense --gpu_ids 2 --name personlowrankvliddensnet   --crossname dqn100time20vilid \
    # --test_dir /home/songwenfeng/swf/pytorch/data/i-LIDS-VID/pytorch/  --which_epoch  199  --classnum 1600 \
    # --lowrankdir /home/songwenfeng/swf/pytorch/data/frameouts/ilids/frame_out/50
# python  0831test_4d_mpb.py --use_dense --gpu_ids 2 --name personlowrankvliddensnet   --crossname dqn100time20vilid \
    # --test_dir /home/songwenfeng/swf/pytorch/data/i-LIDS-VID/pytorch/  --which_epoch  199  --classnum 1600 \
    # --lowrankdir /home/songwenfeng/swf/pytorch/data/frameouts/ilids/frame_out/100
# python  0831test_4d_mpb.py --use_dense --gpu_ids 2 --name personlowrankvliddensnet   --crossname dqn100time20vilid \
    # --test_dir /home/songwenfeng/swf/pytorch/data/i-LIDS-VID/pytorch/  --which_epoch  199  --classnum 1600 \
    # --lowrankdir /home/songwenfeng/swf/pytorch/data/frameouts/ilids/frame_out/1000
# f0=dqntime50vilidpersonlowrankvliddensnet_pytorch_result.mat
# f1=dqntime50vilidpersonlowrankvliddensnet_pytorch_result.mat
# f2=dqnmarstime100vilidpersonlowrankvliddensnet_pytorch_result.mat
# f3=dqnmarstime1000vilidpersonlowrankvliddensnet_pytorch_result.mat
# datasetpath='timedqndatasetvlidstrue'
# echo $f0 $f1 $f2 $f3 $datasetpath
# # python ./dqn.py $f0 $f1 $f2 $f3 $datasetpath 
# #################listfile#######################################
# python ./dqnreward2actionclasses.py "/home/songwenfeng/swf/pytorch/code/0511ori/0528swf/"$datasetpath\
                            # "/home/songwenfeng/swf/pytorch/data/i-LIDS-VID/pytorch/gallery/"\
                            # "/home/songwenfeng/swf/pytorch/code/dqn/imagenet/actor-critic/"$datasetpath
# #################################actions########################################################
# python ./dqnreward2actionclasses.py "/home/songwenfeng/swf/pytorch/code/0511ori/0528swf/"$datasetpath\
                            # "/home/songwenfeng/swf/pytorch/data/frameouts/ilids/frame_out/50/"\
                            # "/home/songwenfeng/swf/pytorch/data/frameouts/ilids/frame_out/100/"\
                            # "/home/songwenfeng/swf/pytorch/data/frameouts/ilids/frame_out/1000/"\
                            # "/home/songwenfeng/swf/pytorch/data/ilids/newlowrankt/"

####################output groundtruth for the actor
######################train actor 
##############prepare dataset
# cd ./imagenet
# #############################make training dataset for actor############################################
# file='ilids'
# mkdir -p ./actor-critic/timeaction$file\2/
# cp -r ./train/* ./actor-critic/timeaction$file\2/
# cp -r --backup=existing --suffix=.orig ./actor-critic/timeaction$file/* ./actor-critic/timeaction$file\2/
# cd ./actor-critic/timeaction$file\2/0
# rename  's/\.jpg\.orig/_orig\.jpg/' *
# cd ../1
# rename  's/\.jpg\.orig/_orig\.jpg/' *
# cd ../2
# cd /home/songwenfeng/swf/pytorch/code/dqn/imagenet/
# rename  's/\.jpg\.orig/_orig\.jpg/' *
# mkdir -p ../actor-critic/actionmodel$file\2/time1/
# mkdir ./actor-critic/timeaction$file\2/train
# cd  ./actor-critic/timeaction$file\2/
# mv *  ./train/
# ln -sf train val
# #######################################################################################################################
# source activate pytorch0.4
# file=ilids
# mkdir -p ./actionmodel$file/time1/
# python main.py -a densenet121    --data ./actor-critic/timeaction$file\2/ --epochs 200  --savepath ./actionmodel$file/time1/ >densenet121$filetemporal.log
# ##############################################train######################################################################
# #######################generate actions###################################################################################
# python test.py -a densenet201  --resume /home/songwenfeng/swf/pytorch/code/dqn/imagenet/actionmodel/time1/model_best.pth.tar\
    # /home/songwenfeng/swf/pytorch/data/i-LIDS-VID/pytorch/train_all/
# python generatedir.py /home/songwenfeng/swf/pytorch/data/i-LIDS-VID/pytorch/train_all_vectorsdensenet121.mat\
              # /home/songwenfeng/swf/pytorch/data/i-LIDS-VID/pytorch/train_all/\
              # ./result/prw/densent121train_all
#########################################################################################################
# f0=dqnduketime50datasetduke0807predukecoco_dense121personmasklowrank_mars_densenet_mbp_pytorch_result.mat
# f1=dqnduketime20datasetduke0807predukecoco_dense121personmasklowrank_mars_densenet_mbp_pytorch_result.mat
# f2=dqnduketime100timedatasetduke0807predukecoco_dense121personmasklowrank_mars_densenet_mbp_pytorch_result.mat
# f3=dqnduketime1000datasetduke0807predukecoco_dense121personmasklowrank_mars_densenet_mbp_pytorch_result.mat
# datasetpath='timedqndatasetduketrue'
# echo $f0 $f1 $f2 $f3 $datasetpath
# python ./dqn.py $f0 $f1 $f2 $f3 $datasetpath 
# # #################listfile#######################################
# # # python ./dqnreward2actionclasses.py "/home/songwenfeng/swf/pytorch/code/0511ori/0528swf/"$datasetpath\
                            # # # "/home/songwenfeng/swf/pytorch/data/DukeMTMC-reID/pytorch/gallery/"\
                            # # # "/home/songwenfeng/swf/pytorch/code/dqn/imagenet/actor-critic/"$datasetpath
# # #################################actions########################################################
# python ./dqnreward2actionclasses.py "/home/songwenfeng/swf/pytorch/code/0511ori/0528swf/"$datasetpath\
                            # "/home/songwenfeng/swf/pytorch/data/frameouts/duke/frame_out/50/"\
                            # "/home/songwenfeng/swf/pytorch/data/frameouts/duke/frame_out/20/"\
                            # "/home/songwenfeng/swf/pytorch/data/frameouts/duke/frame_out/100/"\
                            # "/home/songwenfeng/swf/pytorch/data/duke/newlowrankt/"
################################################################################################
# f0=dqn50time399priddensnentpersonlowrankpridf2densnet_pytorch_result.mat
# f1=dqn20time399priddensnentpersonlowrankpridf2densnet_pytorch_result.mat
# f2=dqn100time399priddensnentpersonlowrankpridf2densnet_pytorch_result.mat
# f3=dqn1000time399priddensnentpersonlowrankpridf2densnet_pytorch_result.mat
# datasetpath='timedqndatasetpridtrue'
# echo $f0 $f1 $f2 $f3 $datasetpath
# python ./dqn.py $f0 $f1 $f2 $f3 $datasetpath 
#################listfile#######################################
# python ./dqnreward2actionclasses.py "/home/songwenfeng/swf/pytorch/code/0511ori/0528swf/"$datasetpath\
                            # "/home/songwenfeng/swf/pytorch/data/DukeMTMC-reID/pytorch/gallery/"\
                            # "/home/songwenfeng/swf/pytorch/code/dqn/imagenet/actor-critic/"$datasetpath
#################################actions########################################################
# python ./dqnreward2actionclasses.py "/home/songwenfeng/swf/pytorch/code/0511ori/0528swf/"$datasetpath\
                            # "/home/songwenfeng/swf/pytorch/data/prid/frame_out/50/"\
                            # "/home/songwenfeng/swf/pytorch/data/prid/frame_out/20/"\
                            # "/home/songwenfeng/swf/pytorch/data/prid/frame_out/100/"\
                            # "/home/songwenfeng/swf/pytorch/data/prid/newlowrankt/"
#########################################################################################
# f0=dqn50time399priddensnentpersonlowrankpridf2densnet_pytorch_result.mat
# f1=dqn20time399priddensnentpersonlowrankpridf2densnet_pytorch_result.mat
# f2=dqn100time399priddensnentpersonlowrankpridf2densnet_pytorch_result.mat
# f3=dqn1000time399priddensnentpersonlowrankpridf2densnet_pytorch_result.mat
# datasetpath='timedqndatasetpridtrue'
# echo $f0 $f1 $f2 $f3 $datasetpath
# python ./dqn.py $f0 $f1 $f2 $f3 $datasetpath 
# #################listfile#######################################
# # python ./dqnreward2actionclasses.py "/home/songwenfeng/swf/pytorch/code/0511ori/0528swf/"$datasetpath\
                            # # "/home/songwenfeng/swf/pytorch/data/DukeMTMC-reID/pytorch/gallery/"\
                            # # "/home/songwenfeng/swf/pytorch/code/dqn/imagenet/actor-critic/"$datasetpath
# #################################actions########################################################
# python ./dqnreward2actionclasses.py "/home/songwenfeng/swf/pytorch/code/0511ori/0528swf/"$datasetpath\
                            # "/home/songwenfeng/swf/pytorch/data/prid/frame_out/50/"\
                            # "/home/songwenfeng/swf/pytorch/data/prid/frame_out/20/"\
                            # "/home/songwenfeng/swf/pytorch/data/prid/frame_out/100/"\
                            # "/home/songwenfeng/swf/pytorch/data/duke/newlowrankt/"
# f0=./dqnmats/mars/dqnmarstime50precocomask_dense121_mars_densenet_mbp_pytorch_result.mat
# f1=./dqnmats/mars/dqnmarstime20precocomask_dense121_mars_densenet_mbp_pytorch_result.mat
# f2=./dqnmats/mars/dqnmarstime100precocomask_dense121_mars_densenet_mbp_pytorch_result.mat
# f3=./dqnmats/mars/dqnmarstime1000precocomask_dense121_mars_densenet_mbp_pytorch_result.mat
# f0=./dqntruemarstime50precocomask_dense121_mars_densenet_mbp_pytorch_result.mat 
# f1=./dqntruemarstime20precocomask_dense121_mars_densenet_mbp_pytorch_result.mat
# f2=./dqntruemarstime1000precocomask_dense121_mars_densenet_mbp_pytorch_result.mat 
# f3=./dqntruemarstime100precocomask_dense121_mars_densenet_mbp_pytorch_result.mat 
# # f1=./dqnmats/mars/dqnmarstime20precocomask_dense121_mars_densenet_mbp_pytorch_result.mat
# # f2=./dqnmats/mars/dqnmarstime100precocomask_dense121_mars_densenet_mbp_pytorch_result.mat
# # f3=./dqnmats/mars/dqnmarstime1000precocomask_dense121_mars_densenet_mbp_pytorch_result.mat
# datasetpath='timedqndatasetmarstrue'
# echo $f0 $f1 $f2 $f3 $datasetpath
# python ./dqn.py $f0 $f1 $f2 $f3 $datasetpath 
# # #################listfile#######################################
# python ./dqnreward2actionclasses.py "/home/songwenfeng/swf/pytorch/code/0511ori/0528swf/"$datasetpath\
    # /home/songwenfeng/swf/pytorch/data/mars/simpytorch/all/\
    # "/home/songwenfeng/swf/pytorch/code/dqn/imagenet/actor-critic/"$datasetpath
# # python ./dqnreward2actionclasses.py "/home/songwenfeng/swf/pytorch/code/0511ori/0528swf/"$datasetpath\
                            # # "/home/songwenfeng/swf/pytorch/data/DukeMTMC-reID/pytorch/gallery/"\
                            # # "/home/songwenfeng/swf/pytorch/code/dqn/imagenet/actor-critic/"$datasetpath
# #################################actions########################################################
# python ./dqnreward2actionclasses.py "/home/songwenfeng/swf/pytorch/code/0511ori/0528swf/"$datasetpath\
                            # "/home/songwenfeng/swf/pytorch/data/prid/frame_out/50/"\
                            # "/home/songwenfeng/swf/pytorch/data/prid/frame_out/20/"\
                            # "/home/songwenfeng/swf/pytorch/data/prid/frame_out/100/"\
                            # "/home/songwenfeng/swf/pytorch/data/duke/newlowrankt/"
###############################################market######################################################
f0=./dqnpridprw50densnent0805coco_personmasklowrank_market_densenet_mbp_pytorch_result.mat 
f1=./dqnpridprw20densnent0805coco_personmasklowrank_market_densenet_mbp_pytorch_result.mat
f2=./dqnpridprw1000densnent0805coco_personmasklowrank_market_densenet_mbp_pytorch_result.mat 
f3=./dqnpridprw100densnent0805coco_personmasklowrank_market_densenet_mbp_pytorch_result.mat 
datasetpath='timedqndatasetmarkettrue'
echo $f0 $f1 $f2 $f3 $datasetpath
python ./dqn.py $f0 $f1 $f2 $f3 $datasetpath 

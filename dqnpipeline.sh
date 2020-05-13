#!/bin/bash
####### 1 select the best temporal actions
######1 input: ori,images1,actor model densnet
#############dqn dataset ##############
####################prid################################################
####################prid 1218################################################
# python ./dqnreward2action.py "/home/songwenfeng/swf/pytorch/code/0511ori/0528swf/timedqndatasetprid/"\
                                # "/home/songwenfeng/swf/pytorch/data/prid/frame_out/50/"\
                                # "/home/songwenfeng/swf/pytorch/data/prid/frame_out/100/"\
                                # "/home/songwenfeng/swf/pytorch/data/prid/frame_out/1000/"\
                                # "/home/songwenfeng/swf/pytorch/data/prid/newlowrankt/"

# # # python mpb_dense_prw_train_4d.py --use_dense --gpu_ids 3 --name ac1temporaldense121rgbmask --train_all --batchsize 32  \
    # # # --data_dir /home/songwenfeng/swf/pytorch/data/PRW/multi-crop/action1files/  --erasing_p 0.5 >ac1temporaldense121rgbmaskprw.log.txt

# python   pridtrain4d.py  --use_dense --gpu_ids 3 --name ac11218temporaldense121prid --train_all --batchsize 32  \
    # --data_dir /home/songwenfeng/swf/pytorch/data/prid/pytorch/  --erasing_p 0.5 --pretrained n   \
    # --lowrankpath /home/songwenfeng/swf/pytorch/data/prid/newlowrankt/ > ac11218temporaldense121prid.log
########################################################################################################
# python topconcatepre_mpb_dense_market_train_4d.py --use_dense --gpu_ids 0 --name rgblowranktopcon0805_lowrank_market_densenet_mbp --train_all --batchsize 32  \
    # --data_dir /home/songwenfeng/swf/pytorch/data/Market/Market_data/pytorch/  --erasing_p 0.5  >rgblowranktopcon0805mpb_mask_marketlog.txt 
# python topconcatepre_mpb_dense_market_train_4d.py --use_dense --gpu_ids 3 --name normrgbdotlowranktopcon0805_lowrank_market_densenet_mbp --train_all --batchsize 32  \
    # --data_dir /home/songwenfeng/swf/pytorch/data/Market/Market_data/pytorch/  --erasing_p 0.5  >normrgbdotlowranktopcon0805mpb_mask_marketlog.txt 
# python mpb_dense_mars_train_4d.py --use_dense --gpu_ids 3 --name 0807predukecoco_dense121personmasklowrank_mars_densenet_mbp --train_all --batchsize 64  \
    # --data_dir /home/songwenfeng/swf/pytorch/data/mars/pytorch  --erasing_p 0.5 --pretrained 280   \
# python   pridtrain4d.py  --use_dense --gpu_ids 3 --name personlowrankpriddensnet --train_all --batchsize 32  \
    # --data_dir /home/songwenfeng/swf/pytorch/data/prid/pytorch/  --erasing_p 0.5 --pretrained n   \
# python  vlidtrain4d.py  --use_dense --gpu_ids 3 --name personlowrankvliddensnet --train_all --batchsize 32  \
    # --data_dir /home/songwenfeng/swf/pytorch/data/i-LIDS-VID/pytorch/  --erasing_p 0.5 --pretrained n   > vlidlog.txt\
    ############################market##################
# python ./dqnreward2action.py "/home/songwenfeng/swf/pytorch/code/0511ori/0528swf/timedqndatasetmarket/"\
                                # "/home/songwenfeng/swf/pytorch/data/frameouts/market/50/"\
                                # "/home/songwenfeng/swf/pytorch/data/frameouts/market/20/"\
                                # "/home/songwenfeng/swf/pytorch/data/frameouts/market/100/"\
                                # "/home/songwenfeng/swf/pytorch/data/market/newlowrank/"
# python mpb_dense_market_train_4d.py --use_dense --gpu_ids 0,1 --name ac1temporaldense121market --train_all --batchsize 32 \
    # --data_dir /home/songwenfeng/swf/pytorch/data/Market/Market_data/pytorch/  --erasing_p 0.5 \
    # --lowrankpath /home/songwenfeng/swf/pytorch/data/market/newlowrank/ >ac1temporaldense121market.log 
###############################################################market 1218########################################
# python ./dqnreward2action.py "/home/songwenfeng/swf/pytorch/code/0511ori/0528swf/timedqndatasetmarket"\
                                # "/home/songwenfeng/swf/pytorch/data/frameouts/market/50/"\
                                # "/home/songwenfeng/swf/pytorch/data/frameouts/market/20/"\
                                # "/home/songwenfeng/swf/pytorch/data/frameouts/market/100/"\
                                # "/home/songwenfeng/swf/pytorch/data/market/newlowrankt/"

# python mpb_dense_market_train_4d.py --use_dense --gpu_ids 0,1 --name ac11218temporaldense121market --train_all --batchsize 32 \
    # --data_dir /home/songwenfeng/swf/pytorch/data/Market/Market_data/pytorch/  --erasing_p 0.5 \
    # --lowrankpath /home/songwenfeng/swf/pytorch/data/market/newlowrankt/ >ac11218temporaldense121market.log 
########################################################mars##########################################################
# python ./dqnreward2action.py "/home/songwenfeng/swf/pytorch/code/0511ori/0528swf/timedqndatasetmars"\
                                # "/home/songwenfeng/swf/pytorch/data/frameouts/mars/50/"\
                                # "/home/songwenfeng/swf/pytorch/data/frameouts/mars/20/"\
                                # "/home/songwenfeng/swf/pytorch/data/frameouts/mars/100/"\
                                # "/home/songwenfeng/swf/pytorch/data/mars/newlowrank/"
# python mpb_dense_mars_train_4d.py --use_dense --gpu_ids 2 --name coco_dense121fcnmask_mars_densenet_mbp --train_all --batchsize 32  \
    # --data_dir /home/songwenfeng/swf/pytorch/data/mars/pytorch  --erasing_p 0.5  \
    # --pretrained n \
    # --lowrankpath /home/songwenfeng/swf/pytorch/data/mars/newlowrank/ > ac1temporaldense121mars.log
###############################duke#########################################################################################
# python ./dqnreward2action.py "/home/songwenfeng/swf/pytorch/code/0511ori/0528swf/timedqndatasetduke"\
                                # "/home/songwenfeng/swf/pytorch/data/frameouts/duke/frame_out/50"\
                                # "/home/songwenfeng/swf/pytorch/data/frameouts/duke/frame_out/20"\
                                # "/home/songwenfeng/swf/pytorch/data/frameouts/duke/frame_out/100"\
                                # "/home/songwenfeng/swf/pytorch/data/duke/newlowrankt/"
# python mpb_dense_dukevideo_train_4d.py  --use_dense --gpu_ids 2 --name ac11218temporaldense121duke --train_all --batchsize 32  \
    # --data_dir /home/songwenfeng/swf/pytorch/data/DukeMTMC-reID/pytorch  --erasing_p 0.5  \
    # --load_model  n \
    # --lowrankpath /home/songwenfeng/swf/pytorch/data/duke/newlowrankt/ > ac11218temporaldense121duke.log
    # # --lowrankpath /home/songwenfeng/swf/pytorch/data/DukeMTMC-reID/personlowrank/dukelabeledrgbmask/ > ac1orimasktemporaldense121duke.log
###############################ilids 1218#########################################################################################
# python ./dqnreward2action.py "/home/songwenfeng/swf/pytorch/code/0511ori/0528swf/timedqndatasetvlids/"\
                                # "/home/songwenfeng/swf/pytorch/data/frameouts/ilids/frame_out/50"\
                                # "/home/songwenfeng/swf/pytorch/data/frameouts/ilids/frame_out/20"\
                                # "/home/songwenfeng/swf/pytorch/data/frameouts/ilids/frame_out/100"\
                                # "/home/songwenfeng/swf/pytorch/data/ilids/newlowrankt/"
# python mpb_dense_dukevideo_train_4d.py  --use_dense --gpu_ids 1 --name ac11218temporaldense121vlids --train_all --batchsize 32  \
    # --data_dir /home/songwenfeng/swf/pytorch/data/i-LIDS-VID/pytorch/  --erasing_p 0.5  \
    # --lowrankpath /home/songwenfeng/swf/pytorch/data/ilids/newlowrankt/ > ac11218temporaldense121vlids.log
    # --lowrankpath /home/songwenfeng/swf/pytorch/data/i-LIDS-VID/personlowrank/ > ac1orimasktemporaldense121vlids.log

# --load_model /home/songwenfeng/swf/pytorch/code/0511ori/0528swf/model/nococo_dense121fcntrainmaskmask_duke_densenet_mbp/net_249.pth
# python mpb_dense_dukevideo_train_4d.py --use_dense --gpu_ids 0 --name nococo_dense121fcntrainmaskmask_duke_densenet_mbp --train_all --batchsize 64  --data_dir /home/songwenfeng/swf/pytorch/data/DukeMTMC-VideoReID/pytorch  --erasing_p 0.5 \
# --load_model /home/songwenfeng/swf/pytorch/code/0511ori/0528swf/model/nococo_dense121fcntrainmaskmask_duke_densenet_mbp/net_249.pth
# python ./dqnreward2action.py "/home/songwenfeng/swf/pytorch/code/0511ori/0528swf/timedqndatasetvlids/"\
                                # "/home/songwenfeng/swf/pytorch/data/frameouts/ilids/frame_out/50"\
                                # "/home/songwenfeng/swf/pytorch/data/frameouts/ilids/frame_out/20"\
                                # "/home/songwenfeng/swf/pytorch/data/frameouts/ilids/frame_out/100"\
                                # "/home/songwenfeng/swf/pytorch/data/ilids/newlowrankt/"

# python mpb_dense_market_train_4d.py --use_dense --gpu_ids 0,1 --name ac1temporaldense121market --train_all --batchsize 32 \
    # --data_dir /home/songwenfeng/swf/pytorch/data/Market/Market_data/pytorch/  --erasing_p 0.5 \
    # --lowrankpath /home/songwenfeng/swf/pytorch/data/market/newlowrankt/ >lowranktac1temporaldense121market.log 
######################################market############################################
# python ./dqnreward2action.py "/home/songwenfeng/swf/pytorch/code/0511ori/0528swf/timedqndatasetmarstrue/"\
                                # "/home/songwenfeng/swf/pytorch/data/frameouts/market/50/"\
                                # "/home/songwenfeng/swf/pytorch/data/frameouts/market/20/"\
                                # "/home/songwenfeng/swf/pytorch/data/frameouts/market/100"\
                                # "/home/songwenfeng/swf/pytorch/data/market/newlowrankt/"
python mpb_dense_dukevideo_train_4d.py  --use_dense --gpu_ids 2 --name ltac11218temporaldense121market --train_all --batchsize 32  \
    --data_dir /home/songwenfeng/swf/pytorch/data/Market/Market_data/pytorch/  --erasing_p 0.5  \
    --lowrankpath /home/songwenfeng/swf/pytorch/data/market/newlowrankt/ > ac11218temporaldense121vlids.log

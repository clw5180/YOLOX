#python -m yolox.tools.train -f exps/visdrone/yolox_s.py -d 1 -b 64 -c /home/caoliwei/base_utils/yolox_s.pth --fp16 -o  # --cache
#python -m yolox.tools.train -expn exp2_1504_baseline -f exps/visdrone/yolox_s.py -d 1 -b 8 -c /home/caoliwei/base_utils/yolox_s.pth --fp16 -o  # for img_sz 1504
#python -m yolox.tools.train -expn exp3_1504_p2_p4 -f exps/visdrone/yolox_s.py -d 1 -b 8 -c /home/caoliwei/base_utils/yolox_s.pth --fp16 -o 
#python -m yolox.tools.train -expn exp4_1504_p2_p5 -f exps/visdrone/yolox_s.py -d 1 -b 8 -c /home/caoliwei/base_utils/yolox_s_add_p2.pth --fp16 -o 
#python -m yolox.tools.train -expn exp5_640_p2_p5 -f exps/visdrone/yolox_s.py -d 1 -b 32 -c /home/caoliwei/base_utils/yolox_s_add_p2.pth --fp16 -o  # for img_sz 640
#python -m yolox.tools.train -expn exp6_640_p3_p5 -f exps/visdrone/yolox_s.py -d 1 -b 32 -c /home/caoliwei/base_utils/yolox_s.pth --fp16 -o
#python -m yolox.tools.train -expn exp7_640_p3_p5_fpn_loss_balance -f exps/visdrone/yolox_s.py -d 1 -b 32 -c /home/caoliwei/base_utils/yolox_s.pth --fp16 -o
#python -m yolox.tools.train -expn exp8_640_p3_p5_fpn_loss_balance_2_not_only_objloss -f exps/visdrone/yolox_s.py -d 1 -b 32 -c /home/caoliwei/base_utils/yolox_s.pth --fp16 -o
#python -m yolox.tools.train -expn exp9_640_focus_to_conv -f exps/visdrone/yolox_s.py -d 1 -b 32 -c /home/caoliwei/base_utils/yolox_s.pth --fp16 -o
#python -m yolox.tools.train -expn exp10_640_mobilenetv3 -f exps/visdrone/yolox_s.py -d 1 -b 32 -c /home/caoliwei/base_utils/mbv3_small_for_detection.pth --fp16 -o
python -m yolox.tools.train -expn exp11_only_backbone_weights -f exps/visdrone/yolox_s.py -d 1 -b 32 -c /home/caoliwei/base_utils/yolox_s_backbone.pth --fp16 -o

#########################################################################################
# -d: number of gpu devices
# -b: total batch size, the recommended number for -b is num-gpu * 8
# --fp16: mixed precision training
# --cache: caching imgs into RAM to accelarate training, which need large system RAM.

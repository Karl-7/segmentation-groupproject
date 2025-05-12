from fastai.vision.all import *
import matplotlib.pyplot as plt
from fastai.optimizer import SGD,Adam
from functools import partial
import random 
import torch
from torch.nn import CrossEntropyLoss
import re
from collections import Counter

# 添加数据增强（翻转、裁剪、缩放、亮度对比度扰动）
batch_tfms=aug_transforms(mult=1.0,do_flip=True,flip_vert=False,max_rotate=10.0,
                            max_zoom=1.1,max_lighting=0.2,max_warp=0.2,
                            p_affine=0.75,p_lighting=0.75,size=224)



def label_func(f): 
    return f[0].isupper()


pat=r'^(.*)_\d+.jpg'
    

def compute_class_weights(files, device='cuda'):
    "计算基于正则 `pat` 提取的多类分类任务中的权重向量"
    prog = re.compile(pat)
    labels = [prog.match(f.name).groups()[0] for f in files]
    label_counts = Counter(labels)

    # 保持类别顺序一致（按 sorted label name 排序）
    sorted_labels = sorted(label_counts.keys())
    label_to_idx = {label: idx for idx, label in enumerate(sorted_labels)}

    class_freq = [label_counts[label] for label in sorted_labels]
    total = sum(class_freq)
    weights = [total / c for c in class_freq]

    return torch.tensor(weights, dtype=torch.float, device=device), label_to_idx





if __name__ == '__main__':
    # path=untar_data(URLs.PETS)#
    path=Path('D:\A_KTH_Course\DL_DataScience\PROJECT-1\oxford-iiit-pet')
    print(path.ls())
    files=get_image_files(path/"images")
    #>////////////////////////////////////////////////////////////////////////////////////
    # Randomly delete 50% of images with uppercase first letter in their names
    uppercase_files=[f for f in files if label_func(f.name)]
    all_cats=len(uppercase_files)
    print("all_cats:",all_cats)
    to_delete=random.sample(uppercase_files,all_cats-all_cats//5)
    print("using cats:",all_cats-len(to_delete))
    print("example to delete:",to_delete[0:2])
    files=[f for f in files if f not in to_delete]
    #<//////////////////////////////////////////////////////////////////////////////////
    print("all files:",len(files))

    #>/////////////////////////////////////////////////////////////////////////////////
    # ✅ 1. 直接使用 默认data augmentation(没用）)
    # dls=ImageDataLoaders.from_name_re(path,files,pat,item_tfms=Resize(460),
                                    # batch_tfms=aug_transforms(size=224))
    #</////////////////////////////////////////////////////////////////////////////////
    #不使用数据增强
    # dls=ImageDataLoaders.from_name_re(path,files,pat,item_tfms=Resize(224),num_workers=0)
    #>//////////////////////////////////////////////////////////////////////////////////
    # ✅ 2. 使用自定义的 data augmentation
    
    dls=ImageDataLoaders.from_name_re(path,files,pat,
                                    item_tfms=Resize(460), # resize before augment
                                    batch_tfms=batch_tfms,
                                    num_workers=0)
    #<//////////////////////////////////////////////////////////////////////////////////

    #>///////////////////////////////////////////////////////////////////////////////////
    # NO L2 TERM
    # learn=vision_learner(dls,resnet18,metrics=error_rate)
    #<>//////////////////////////////////////////////////////////////////////////////////
    # WITH L2 TERM
    # learn=vision_learner(dls,resnet18,metrics=error_rate,wd=1e-4)#
    #<//////////////////////////////////////////////////////////////////////////////////
    #with weighted BCE:
    weights, label_to_idx = compute_class_weights(files)
    plt.figure(figsize=(12, 4))
    plt.bar(label_to_idx.keys(), weights.cpu().numpy())
    plt.xticks(rotation=90)
    plt.title("Class weights (inversely proportional to frequency)")
    plt.show()
    learn = vision_learner(dls, resnet18, metrics=[accuracy, error_rate],
                        loss_func=CrossEntropyLoss(weight=weights),
                        wd=1e-4)
    # learn.freeze()
    
    # learn.fit_one_cycle(1)

    learn.unfreeze()
    result={}
    for i in range(1,5):
        learn.freeze_to(-i)
        learn.opt_func=partial(SGD,mom=0.03)
        lr_min,lr_steep=learn.lr_find(suggest_funcs=(minimum,steep))
        
        print(f"📉 Suggested lr_min: {lr_min:.2e},lr_steep: {lr_steep:.2e}")
        print("freezing to layer:",-i)

        #////////////////////////////////////////////////////////////////////////////////////////////
        #✅ 手动执行一遍 forward pass 来更新 BatchNorm 的均值/方差（无需梯度）
        # 第一步：先把整个模型设为 eval 模式，防止不小心更新冻结层的 BatchNorm stats
        learn.model.eval()
        # 第二步：把解冻的那部分设为 train（只更新这部分的 BN stats）
        for name,module in learn.model.named_children():
            module.train()
        with torch.no_grad():
            for xb,yb in dls.train:
                xb,yb=xb.cuda(),yb.cuda()
                _=learn.model(xb)
                break  # 一次或几次就足够更新 BatchNorm stats
        #////////////////////////////////////////////////////////////////////////////////////////////

        learn.fit_one_cycle(1,lr_max=slice(lr_min/10,lr_min))
        acc=1.0-learn.validate()[1]
        result["unfreeze to "+str(-i)+"layer"]=acc
        print(f"Unfreeze Till Layer {-i} acc: {acc:.4f}")
    print("Results:",result)
    # learn.show_results()
    # plt.show()
    # learn.fine_tune(2,3e-3)p
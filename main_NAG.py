from fastai.vision.all import *
import matplotlib.pyplot as plt
from fastai.optimizer import SGD,Adam
from functools import partial

def label_func(f): 
    return f[0].isupper()

if __name__ == '__main__':
    path = untar_data(URLs.PETS)
    print(path.ls())
    files = get_image_files(path/"images")
    print(len(files))
    
    dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224),num_workers=0)

    learn = vision_learner(dls, resnet34, metrics=error_rate)
    learn.opt_func=SGD(partial,lr=0.0005,mom=0.2)
    # learn.freeze()
    # learn.fit_one_cycle(1)

    print("\n unfreezing:")
    learn.freeze_to(-1)
    
    #>////////////////////////////////////////////////////////////////////////////////////
    ###finding the learning rate
    # lr=learn.lr_find()
    # result: lr: SuggestedLRs(valley=0.0005754399462603033)█████████████████████████████████████----| 95.65% [88/92 02:45<00:07 1.8493]
    # learn.opt_func.lr=lr
    # print("lr:", learn.opt_func.lr)
    #<////////////////////////////////////////////////////////////////////////////////////
    
    # print("optimizer:", learn.opt_func)
    print("optim", type(learn.opt_func))
    print("lr:", learn.opt_func.param_groups[0]['lr'])
    print("momentum:", learn.opt_func.param_groups[0]['mom'])
    learn.fit_one_cycle(1)
    learn.show_results()
    plt.show()
    
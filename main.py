from fastai.vision.all import *
import matplotlib.pyplot as plt
import torch.optim as optim


def label_func(f): 
    return f[0].isupper()

if __name__ == '__main__':
    path = untar_data(URLs.PETS)
    print(path.ls())
    files = get_image_files(path/"images")
    print(len(files))
    
    dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224),num_workers=0)
    
    
    # # dls.show_batch()
    # # plt.show()
    learn = vision_learner(dls, resnet34, metrics=error_rate)
    
    print("lr:", learn.opt_func.lr)
    # learn.freeze()
    # learn.fit_one_cycle(1)
    print("\n unfreezing:")
    # learn.unfreeze()
    learn.freeze_to(-1)
    learn.fit_one_cycle(1)
    learn.fine_tune(2, 3e-3)
    learn.show_results()
    plt.show()
    
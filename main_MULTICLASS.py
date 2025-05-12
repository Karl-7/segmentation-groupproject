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
    
    
    pat = r'^(.*)_\d+.jpg'
    # dls = ImageDataLoaders.from_name_re(path, files, pat, item_tfms=Resize(460),
                                    # batch_tfms=aug_transforms(size=224))
    dls = ImageDataLoaders.from_name_re(path, files, pat, item_tfms=Resize(224),num_workers=0)
    # dls.show_batch()

    results={}
    for l in range(-5, 0):
        print(f"\n🔍 Testing freeze_to({l})")
        # 每次重新构建模型
        learn = vision_learner(dls, resnet18, metrics=error_rate)
        learn.freeze_to(l)
        learn.opt_func=partial(SGD,momentum=0.3)
        lr_min, lr_steep = learn.lr_find(suggest_funcs=(minimum, steep))
        print(f"📉 Suggested lr_min: {lr_min:.2e}, lr_steep: {lr_steep:.2e}")
        learn.fit_one_cycle(1, lr_max=slice(lr_min/10, lr_min))
        acc = learn.validate()[1]
        
        results["unfreeze to "+str(l)+"layer"] = 1-acc
        
        print(f"Unfreeze Till Layer {l} acc: {acc:.4f}")
    print("Results:", results)
    # learn.show_results()
    # plt.show()
    # learn.fine_tune(2, 3e-3)
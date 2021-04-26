import torch
import os
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm
import pickle
import pretrainedmodels
import pretrainedmodels.utils as utils

load_img = utils.LoadImage()

# transformations depending on the model
# rescale, center crop, normalize, and others (ex: ToBGR, ToRange255)

model_name = 'se_resnext101_32x4d' # could be fbresnet152 or inceptionresnetv2
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
model.eval()
tf_img = utils.TransformImage(model) 
cudnn.enabled = True
cudnn.benchmark = True
image_datasets = datasets.ImageFolder( '/data0/EEV/data-frames' , tf_img)

batchsize = 4096
dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=batchsize, shuffle=False, num_workers=16, pin_memory=True)
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    features = torch.FloatTensor()
    count = 0
    part = 0
    pool =  torch.nn.AdaptiveAvgPool2d((1,1))
    with tqdm(dataloaders, ascii=True) as tq:
        for data in tq:
            img, label = data
            n, c, h, w = img.size()
            count += n
            ff = torch.FloatTensor(n,2048).zero_().cuda() # we have six parts

            for i in range(1):
                if(i==1):
                    img = fliplr(img)
                input_img = Variable(img.cuda())
            #for scale in [1]:
            #    if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    #input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img) 
                #print(outputs.shape)
                #outputs = pool(outputs)
                ff += outputs.view(outputs.size(0), outputs.size(1))
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            features = torch.cat((features,ff.data.cpu()), 0)
            if count > batchsize*100 or n<batchsize: 
                count = 0
                snapshot_feature_pkl = '/data/EEV/'+model_name+'_%04d'%part+'.pkl'
                print('saving feature at %s'%snapshot_feature_pkl)
                part = part +1
                save_pkl(snapshot_feature_pkl, features)   
                features = torch.FloatTensor()
                
    return 

def save_pkl(name,data):
    f = open(name, "wb")
    pickle.dump(data,f)
    f.close()
def load_pkl(name):
    with open(name, "rb") as fp:   #Pickling
        mydict = pickle.load(fp) 
    return mydict


path = image_datasets.imgs
fdict = {}
key = []
print('Saving keys')
for i, p in enumerate(path):
    k = os.path.basename(os.path.dirname(p[0]))
    key.append(k)
save_pkl('/data/EEV/key_for_features_zzd.pkl', key)
print('Saved keys')
#    print(k)
# Extract feature
snapshot_feature_pkl = '/data/EEV/'+model_name+'_%04d'%0+'.pkl'
model.last_linear = torch.nn.Sequential()
if not os.path.isfile(snapshot_feature_pkl): 
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])
    model = model.cuda()
    with torch.no_grad():
        extract_feature(model,dataloaders)
else:
    features = load_pkl(snapshot_feature_pkl)


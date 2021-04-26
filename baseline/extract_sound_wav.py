import openl3
import soundfile as sf
import os
import pickle
from multiprocessing import Pool
import tensorflow as tf

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 

def save_pkl(name,data):
    f = open(name, "wb")
    pickle.dump(data,f)
    f.close()

audio_path = '/data/EEV/data-audio/'
save_path = '/data/EEV/audio-zzd-feature/'

model = openl3.models.load_audio_embedding_model(input_repr="linear", content_type="env",
                                                 embedding_size=512)

feature_dict = {} 
count = 0 
for root, dirs, files in os.walk(audio_path, topdown=True):
    for name in files:
        if not name[-3:]=='wav':
            continue
        src_path = audio_path + '/' + name
        if not os.path.isfile(save_path+name[:-4]+'.pkl'):
            audio, sr = sf.read(src_path)
            emb, ts = openl3.get_audio_embedding(audio, sr, model=model, hop_size=1/6)
            save_pkl(save_path+name[:-4]+'.pkl', emb)
            feature_dict[save_path+name[:-4]+'.pkl']= emb
            count =  count + 1 
        else:
            print('skipping %s'%name)

        #if count == 16:
        #    for key in feature_dict:
                #save_pkl(key, feature_dict[key])
        #    count = 0
        #    feature_dict = {}

import openl3
import soundfile as sf
import os
import pickle
from multiprocessing import Pool

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
audio_filepath_list = {}
for root, dirs, files in os.walk(audio_path, topdown=True):
    for name in files:
        if not name[-3:]=='wav':
            continue
        src_path = audio_path + '/' + name
        audio_filepath_list.append(src_path)

openl3.process_audio_file(audio_filepath_list, batch_size=32, model=model, hop_size=1/6, output_dir=save_path)

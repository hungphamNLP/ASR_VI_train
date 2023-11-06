from ASR.model import ASR_Wav2VecCTC
from ASR.audio import AudioFile
from pathlib import Path
path_root = Path(__file__).parents[0]
from utils import extract_audio,convert_audio,write_to_file,DEFAULT_TEMP_DIR,DEFAULT_FORMAT
import tqdm
import subprocess
from autosub import SubGenerator 



asr_path = str(path_root) + '/ASR/model_asr.pth'
vocab_json = str(path_root) + '/ASR/vocab.json'
vocab_path = str(path_root) + '/ASR/vocab_.txt'
model_language_path = str(path_root) + '/ASR/language_model/lm.binary'
model = ASR_Wav2VecCTC(asr_path,vocab_json,model_language_path,vocab_path)


from text_processing.inverse_normalize import InverseNormalizer
normalizer = InverseNormalizer('vi')


# post_progress= SubGenerator(model)
from gector.gec_model import GecBERTModel

gector = GecBERTModel(vocab_path="./data/output_vocabulary",
                      model_paths=["./gector/Model_GECTOR"],
                      device=None,
                      max_len=64, min_len=3,
                      iterations=3,
                      min_error_probability=0.2,
                      lowercase_tokens=False,
                      log=False,
                      confidence=0,
                      weights=None,
                      split_chunk=True,
                      chunk_size=48,
                      overlap_size=16,
                      min_words_cut=8)

last = 0
trans_dict = None
recognize_tokens = []
allow_tags = {"speech", "male", "female", "noisy_speech", "music_"}
segment_backend='vad'
classify_segment=False
show_progress=False
transcribe_music=False
auto_punc=False




def Segmen_VAD(audio_file,recognize_tokens):
    audio_file = AudioFile(audio_file)
    progress_bar = tqdm.tqdm(total=int(audio_file.audio_length * 1000))
    trans_dict = None
    for (start, end, audio, tag) in audio_file.split(backend='vad', classify=True):
        if tag not in allow_tags:
            continue
        if tag == "music" and not transcribe_music:
            if trans_dict is not None:
                # final_tokens = self.post_process(trans_dict['tokens'], auto_punc=auto_punc)
                # recognize_tokens.extend(final_tokens)
                trans_dict = None
            recognize_tokens.append(
                {
                    "text": "[âm nhạc]",
                    "start": start,
                    "end": end,
                }
            )
            continue
        if  tag == "background":
            if trans_dict is not None:
                # final_tokens = self.post_process(trans_dict['tokens'], auto_punc=auto_punc)
                # recognize_tokens.extend(final_tokens)
                trans_dict = None
            recognize_tokens.append(
                {
                    "text": "[no Speech]",
                    "start": start,
                    "end": end,
                }
            )
            continue
        yield start,end,audio


def format_text(tokens,auto_punc=True):
    final_tokens = normalizer.inverse_normalize_with_metadata(tokens,verbose=False)
    if auto_punc:
        final_batch,_ = gector.handle_batch_with_metadata([final_tokens],add_punc=auto_punc)
        final_tokens = final_batch[0]
    return final_tokens



def format_text_2(tokens,auto_punc=True):
    final_tokens = normalizer.inverse_normalize_with_metadata_text(tokens,verbose=False)
    if auto_punc:
        final_batch,_ = gector.handle_batch_with_metadata_text([[final_tokens]],add_punc=auto_punc)
        final_tokens = final_batch[0]

    return ' '.join([token['text'] for token in final_tokens])




def Speech2Text_stream(audio_file):
    transcribe = []
    for start,end,audio in Segmen_VAD(audio_file,recognize_tokens):
        z,tokens,y = model.transcribe_with_metadata(audio,start)[0]
        rs = ' '.join([token["text"] for token in tokens]) 
        
        transcribe.append(rs)

    return " ".join([token for token in transcribe])
        

import io

def Speech2Text_stream_io(stream):
    audio_file = io.BytesIO(stream)
    transcribe = []
    for start,end,audio in Segmen_VAD(audio_file,recognize_tokens):
        z,tokens,y = model.transcribe_with_metadata(audio,start)[0]
        rs = ' '.join([token["text"] for token in tokens]) 
        
        transcribe.append(rs)

    return " ".join([token for token in transcribe])


def instance_fileaudio(audio_file):
    trans_dict = None
    last = 0
    recognize_tokens = []
    for start,end,audio in Segmen_VAD(audio_file,recognize_tokens):
        z, tokens, y = model.transcribe_with_metadata(audio, start)[0]
        if trans_dict is not None:
            if (len(tokens)==0 or start -  trans_dict.get('end', 0)  > 200 or len(trans_dict['tokens']) > 32):
                final_tokens = format_text(trans_dict['tokens'])
                trans_dict = None
                # print(" ".join([token["text"] for token in final_tokens]))
                recognize_tokens.extend(final_tokens)
            else:
                trans_dict['tokens'].extend(tokens)
                trans_dict['split_times'].append(trans_dict['end'])
                trans_dict['end'] = end
        
        if trans_dict is None:
                trans_dict = {
                    'tokens': tokens,
                    'start': start,
                    'end': end,
                    'split_times': [],
                }
        # print(trans_dict)
    if trans_dict is not None:
        final_tokens = format_text(trans_dict['tokens'])
        recognize_tokens.extend(final_tokens)

    return recognize_tokens


def instance_fileaudio_stream(audio_file):
    trans_dict = None
    last = 0
    recognize_tokens = []
    for start,end,audio in Segmen_VAD(audio_file,recognize_tokens):
        z, tokens, y = model.transcribe_with_metadata(audio, start)[0]
        if trans_dict is not None:
            if (len(tokens)==0 or start -  trans_dict.get('end', 0)  > 200 or len(trans_dict['tokens']) > 32):
                final_tokens = format_text(trans_dict['tokens'])
                trans_dict = None
                yield " ".join([token["text"] for token in final_tokens])
                # recognize_tokens.extend(final_tokens)
            else:
                trans_dict['tokens'].extend(tokens)
                trans_dict['split_times'].append(trans_dict['end'])
                trans_dict['end'] = end
        
        if trans_dict is None:
                trans_dict = {
                    'tokens': tokens,
                    'start': start,
                    'end': end,
                    'split_times': [],
                }

        # print(trans_dict)
    if trans_dict is not None:
        final_tokens = format_text(trans_dict['tokens'])
        yield " ".join([token["text"] for token in final_tokens])

    # return recognize_tokens



# def Speech2Text_demo_3(audio_file):
#     audio_file = AudioFile(audio_file)
#     progress_bar = tqdm.tqdm(total=int(audio_file.audio_length * 1000))
#     result = []
    
#     for (start, end, audio, tag) in audio_file.split(backend='vad', classify=True):
#         trans_dict = None
#         if tag not in allow_tags:
#             continue
#         if tag == "music" and not transcribe_music:
#             continue
#         if tag == "background":
#             continue

#         namespeaker = get_label_db(audio)
#         z, tokens, y = model.transcribe_with_metadata(audio, start)[0]
#         final_tokens =normalizer.inverse_normalize_with_metadata(tokens, verbose=False)
#         final_batch, _ = gector.handle_batch_with_metadata([final_tokens], add_punc=True)
#         final = final_batch[0]
#         infer_text = " ".join([token["text"] for token in final])
#         # print(infer_text)  
#         if trans_dict is None:
#             trans_dict = {
#                 'name_speaker': namespeaker,
#                 'transcript': infer_text
#             }
        
#         yield trans_dict


# def Speech2Text_demo(audio_file):
#     audio_file = AudioFile(audio_file)
#     progress_bar = tqdm.tqdm(total=int(audio_file.audio_length * 1000))
#     result = []
#     trans_dict = None
#     for (start, end, audio, tag) in audio_file.split(backend='vad', classify=True):
#         if tag not in allow_tags:
#             continue
#         if tag == "music" and not transcribe_music:
#             continue
#         if tag == "background":
#             continue

#         namespeaker = get_label_db(audio)
#         z, tokens, y = model.transcribe_with_metadata(audio, start)[0]
#         final_tokens =normalizer.inverse_normalize_with_metadata(tokens, verbose=False)
#         final_batch, _ = gector.handle_batch_with_metadata([final_tokens], add_punc=True)
#         final = final_batch[0]
#         infer_text = " ".join([token["text"] for token in final])
#         # print(infer_text)  
#         if trans_dict is None:
#             trans_dict = {
#                 'name_speaker': namespeaker,
#                 'transcript': infer_text
#             }
#         if trans_dict is not None:
#             result.append(trans_dict)
#         # print(trans_dict)
#             trans_dict = None

#     return result

if __name__ == '__main__':
    # for stream in instance_fileaudio_stream('bogia.wav'):
    #     print(stream)# send socket fe
    # Speech2Text_stream('bogia.wav')
    out=format_text_2('xin chào chúc bạn một ngày tốt lành cùng với nhiều hạnh phúc')
    print(out)

    # out=Speech2Text_stream('bogia.wav')
    # print(out)
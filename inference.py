from build_config import processor,config
from model_handling import Wav2Vec2ForCTC
import torchaudio
import torch

input_speech,_ = torchaudio.load(config['wav_test'])
input=processor.feature_extractor(input_speech[0],return_tensors='pt')


model = Wav2Vec2ForCTC.from_pretrained(config['output_dir']+'/checkpoint-100')
out = model(**input)
rs=processor.tokenizer.batch_decode(torch.argmax(out.logits,dim=-1))
print(rs)

from transformers import Wav2Vec2CTCTokenizer,Wav2Vec2FeatureExtractor,Wav2Vec2Processor
import yaml 
from model_handling import Wav2Vec2ForCTC

with open('./config.yaml','r') as f:
    config = yaml.load(f,Loader=yaml.FullLoader)


feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(config['tokenize'])
model = Wav2Vec2ForCTC.from_pretrained(config['model_pretrained'], ctc_loss_reduction="mean",pad_token_id=tokenizer.pad_token_id)
#model.save_pretrained('Model_pretrained')

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

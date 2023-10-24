# ASR_VI_train

## Mô tả định dạng dữ liệu
- file .json {"file":"dataset_sample/quangnam.wav","text":"cơ bản tình hình tất niên còn nhiều vấn đề diễn biến phức tạp","duration":4.98}
- sample rate : 16000 Hz
- file âm thanh : .wav
- key "file" : đường dẫn đến file âm thanh
- key "text" : transcript file âm thanh

## File cấu hình config.yaml 
- Thay đổi các tham số cho phù hợp
- bacth
- proc_num: tùy vào lượng RAM và cpu khuyến khích dùng 4 worker
- num_epoch: só vòng lặp khi train
- eval_steps: tiến hành validation sau khi train n lần
- wav_test: file test tiến inference xem kết quả
- model_prertrained: mô hình pretrained
- output_dir : lưu lại mô hình đã train lại folder
- max_input_length_in_sec: độ dài thời gian âm thanh khuyến nghị để max (20 s)
- dataset_train: đường dẫn file .json chứa dữ liệu train
- dataset_test: đường dẫn file .json chứa dữ liệu test
## Các bước train mô hình với docker
#### 1. Tải docker image tiến hành chạy container này trong máy 
- docker pull nvcr.io/nvidia/tensorrt:22.05-py3
- docker run -it --gpus=all ${PWD}:/home/ nvcr.io/nvidia/tensorrt:22.05-py3
- cd /home/
#### 2. Cài đặt môi trường để train
- pip install requirements.txt
- python run_train.py
#### 3. Test kết quả train được
- python inference.py
  
  

python3 encode.py --input_path=resources/test_images/peppers.png --output_path=resources/encoded/B=10/peppers.encoded --encoder_path=models/model_B=10/encoder.model --quantize_mode=soft
python3 encode.py --input_path=resources/test_images/lena.png --output_path=resources/encoded/B=10/lena.encoded --encoder_path=models/model_B=10/encoder.model --quantize_mode=soft
python3 encode.py --input_path=resources/test_images/baboon.png --output_path=resources/encoded/B=10/baboon.encoded --encoder_path=models/model_B=10/encoder.model  --quantize_mode=soft

python3 encode.py --input_path=resources/test_images/peppers.png --output_path=resources/encoded/B=2/peppers.encoded --encoder_path=models/model_B=2/encoder.model --quantize_mode=hard
python3 encode.py --input_path=resources/test_images/lena.png --output_path=resources/encoded/B=2/lena.encoded --encoder_path=models/model_B=2/encoder.model --quantize_mode=hard
python3 encode.py --input_path=resources/test_images/baboon.png --output_path=resources/encoded/B=2/baboon.encoded --encoder_path=models/model_B=2/encoder.model  --quantize_mode=hard

python3 decode.py --output_path=resources/results/B=10/peppers_reconstructed.png --input_path=resources/encoded/B=10/peppers.encoded --decoder_path=models/model_B=10/decoder.model --quantize_mode=soft
python3 decode.py --output_path=resources/results/B=10/lena_reconstructed.png --input_path=resources/encoded/B=10/lena.encoded --decoder_path=models/model_B=10/decoder.model --quantize_mode=soft
python3 decode.py --output_path=resources/results/B=10/baboon_reconstructed.png --input_path=resources/encoded/B=10/baboon.encoded --decoder_path=models/model_B=10/decoder.model --quantize_mode=soft

python3 decode.py --output_path=resources/results/B=2/peppers_reconstructed.png --input_path=resources/encoded/B=2/peppers.encoded --decoder_path=models/model_B=2/decoder.model --quantize_mode=hard
python3 decode.py --output_path=resources/results/B=2/lena_reconstructed.png --input_path=resources/encoded/B=2/lena.encoded --decoder_path=models/model_B=2/decoder.model --quantize_mode=hard
python3 decode.py --output_path=resources/results/B=2/baboon_reconstructed.png --input_path=resources/encoded/B=2/baboon.encoded --decoder_path=models/model_B=2/decoder.model --quantize_mode=hard
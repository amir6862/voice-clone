# model/
# Pre-trained model weights are stored here when loaded.
# The zero-shot voice conversion pipeline does not require
# pre-downloaded weights — it runs entirely from spectral features.
#
# To extend with a neural vocoder (e.g. HiFi-GAN):
#   1. Download HiFi-GAN weights from https://github.com/jik876/hifi-gan
#   2. Place generator_v1 checkpoint here
#   3. Update voice_converter.py to use HifiganVocoder class

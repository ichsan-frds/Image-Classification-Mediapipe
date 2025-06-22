# Image Classification Mediapipe
![image](https://github.com/user-attachments/assets/ac123b7f-df8b-4b0f-8f09-10c7125ae4a7)

## Demo
### Static Image
![image](https://github.com/user-attachments/assets/5304cf42-f057-4e90-8396-255af44eacd1)

### Realtime
![image](https://github.com/user-attachments/assets/854ca29c-7deb-4a57-8ac4-f8af31aa1795)

## Experiment
### Comparing the trade-off between model size and latency, accuracy, and confidence.
Model | Model Size (MB) | Average Latency (ms) | Top-1 Accuracy (%) | Top-3 Accuracy (%) | Average Confidence |
--- | --- | --- | --- |--- |---
EfficientNet-Lite0 (int)  | 5.3 | 11.82 | 84.00 | 94.00 | 0.62 |
EfficientNet-Lite0 (float)  | 18.1 | 19.12 | 88.00 | 94.00 | 0.66 |
EfficientNet-Lite0 (int)  | 6.9 | 22.35 | 80.00 | 92.00 | 0.49 |
EfficientNet-Lite2 (float)  | 23.7 | 40.76 | 84.00 | 94.00 | 0.52 |

### Conclusion
Based on the test results, the quantized (int8) model is characterized by its small size and low latency, with an accuracy and confidence trade-off that is not significantly worse than the float32 model. The float32 model, on the other hand, generally produces higher-quality inference results.

Although Lite2 (float) provides the best accuracy, its inference time is twice as slow compared to Lite0 (int). Therefore, model selection should consider the context of use. For real-time applications or resource-constrained devices, Lite0 (int) may be more suitable; whereas for high-accuracy requirements on devices with sufficient resources, Lite2 (float) is the ideal choice.

## Source
- Program Reference  : [[Code](https://ai.google.dev/edge/mediapipe/solutions/vision/image_classifier/python)]
- Pre-trained Models : [[Models](https://ai.google.dev/edge/mediapipe/solutions/vision/image_classifier#models)]

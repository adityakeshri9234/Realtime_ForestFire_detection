
# Realtime Forest Fire Detection

![Project Status](https://img.shields.io/badge/Status-Active-green)
![License](https://img.shields.io/badge/License-MIT-blue)

## Overview

Wildfires are among the most destructive natural disasters, causing extensive ecological, economic, and social damage. This project aims to provide a robust, real-time forest fire detection and prediction system using a hybrid deep learning framework. By integrating Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs), the system predicts wildfire spread with a focus on efficiency and accuracy.

## Key Features

- **Hybrid Model Architecture**: Combines CNN (MobileNetV2) for spatial feature extraction and RNN (LSTM) for temporal analysis.
- **High Accuracy**: Achieves an accuracy of 97.2% and precision of 82.1% for fire region detection.
- **IoT Integration**: Designed for real-time monitoring using IoT-enabled sensors.
- **Data Augmentation**: Includes techniques like random cropping and center cropping for robust training.
- **Scalability**: Lightweight and adaptable for diverse geographical locations and environmental conditions.

---

   
## Project Workflow

1. **Data Preprocessing**: 
   - Input features include environmental parameters like temperature, humidity, wind speed, and vegetation indices.
   - Normalization and clipping ensure consistency and efficiency.
   ![image](https://github.com/user-attachments/assets/0a864bcf-2073-45b7-a871-a7cbadbf5d0f)

2. **Model Architecture**:
   - MobileNetV2 for efficient feature extraction.
   - LSTM layers for capturing temporal dependencies.
   - Upsampling with skip connections for precise segmentation.
   ![Screenshot 2024-11-24 185126](https://github.com/user-attachments/assets/ec0aab6c-a600-455b-a5f8-12f2fa6c91f9)

3. **Training and Evaluation**:
   - BCE-Dice loss for optimizing segmentation.
   - Metrics such as IoU, precision, recall, and F1-score for performance evaluation.
   - Early stopping and model checkpointing for optimal results.
     ![image](https://github.com/user-attachments/assets/f8de7857-a8e9-4e5c-beb4-93b45cf4f619)


4. **Deployment**:
   - Potential for integration with IoT devices and real-time systems.

---

## Results

| Metric        | Value   |
|---------------|---------|
| Accuracy      | 97.2%   |
| IoU           | 21.3%   |
| Precision     | 82.1%   |
| Recall        | 24.0%   |
| F1-Score      | 24.6%   |
| False Alarm Rate (FAR) | 0.179 |

### Visualization

- **Training vs. Validation Loss**: Plots indicate stable convergence and minimal overfitting.
- **Inference Results**: Comparison of true vs. predicted fire regions for selected samples.

---

## Libraries and Tools

- TensorFlow
- NumPy
- Matplotlib
- Pandas
- OpenCV
- Scikit-Learn
- TQDM

---

## Future Improvements

- **Enhanced Recall**: Incorporate focal loss and address class imbalance.
- **Attention Mechanisms**: Improve model focus on critical regions.
- **Multi-Modal Data**: Leverage additional data sources like weather forecasts and historical fire data.
- **Real-Time Implementation**: Deploy with edge computing and IoT networks for proactive disaster management.

---

## Installation

### Clone the Repository
```bash
git clone https://github.com/adityakeshri9234/Realtime_ForestFire_detection.git
cd Realtime_ForestFire_detection
```



---

## Usage

1. Prepare your dataset in the required format.
   ```bash
   python datapreprocessing.py
   ```
2. ```bash
   !pip install git+https://github.com/tensorflow/examples.git
   ```
3. Initialize the model:
   ```bash
   python model.py
   ```
4. Train and evaluate results:
   ```bash
   python train_evaluate.py
   ```
5. Deploy for real-time detection.

---

## References

- **Next Day Wildfire Spread Dataset**: [Kaggle](https://www.kaggle.com/)
- TensorFlow Documentation: [TensorFlow](https://www.tensorflow.org/)
- Related Literature:
  - Shi et al., "Change Detection in Remote Sensing Using AI".
  - Dampage et al., "Forest Fire Detection Using WSN and ML".
  - Grari et al., "IoT and ML for Forest Fire Monitoring".

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

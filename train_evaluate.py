import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tqdm import tqdm
from typing import Callable, Tuple

# Intersection over Union (IoU) Metric
def IoU_metric(real_mask: tf.Tensor, predicted_mask: tf.Tensor) -> float:
    """
    Calculates the Intersection over Union (IoU) metric.
    """
    real_mask = tf.where(real_mask < 0, 0, real_mask)
    intersection = np.logical_and(real_mask, predicted_mask)
    union = np.logical_or(real_mask, predicted_mask)
    
    if np.sum(union) == 0:
        return 1.0  # Perfect match if both masks are empty
    return np.sum(intersection) / np.sum(union)

# Recall Metric
def recall_metric(real_mask: tf.Tensor, predicted_mask: tf.Tensor) -> float:
    """
    Calculates the recall (sensitivity) metric.
    """
    real_mask = tf.where(real_mask < 0, 0, real_mask)
    true_positives = np.sum(np.logical_and(real_mask, predicted_mask))
    actual_positives = np.sum(real_mask)
    if actual_positives == 0:
        return 1.0
    return true_positives / actual_positives

# Precision Metric
def precision_metric(real_mask: tf.Tensor, predicted_mask: tf.Tensor) -> float:
    """
    Calculates the precision metric.
    """
    real_mask = tf.where(real_mask < 0, 0, real_mask)
    true_positives = np.sum(np.logical_and(real_mask, predicted_mask))
    predicted_positives = np.sum(predicted_mask)
    if predicted_positives == 0:
        return 1.0
    return true_positives / predicted_positives

# F1-Score

def calculate_f1_score(real_mask: tf.Tensor, predicted_mask: tf.Tensor) -> float:
    """
    Calculates the F1-Score.
    """
    precision = precision_metric(real_mask, predicted_mask)
    recall = recall_metric(real_mask, predicted_mask)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)



# False Alarm Rate (FAR)
def false_alarm_rate(real_mask: tf.Tensor, predicted_mask: tf.Tensor) -> float:
    """
    Calculates the False Alarm Rate (FAR).
    """
    real_mask = tf.where(real_mask < 0, 0, real_mask)
    false_positives = np.sum(np.logical_and(predicted_mask, np.logical_not(real_mask)))
    predicted_positives = np.sum(predicted_mask)
    if predicted_positives == 0:
        return 0.0
    return false_positives / predicted_positives

# Dice Coefficient
def dice_coef(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Dice coefficient for similarity measurement.
    """
    smooth = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

# Weighted Binary Crossentropy
def weighted_bincrossentropy(true: tf.Tensor, pred: tf.Tensor, weight_zero: float = 0.01, weight_one: float = 1.0) -> tf.Tensor:
    """
    Calculates weighted binary crossentropy.
    """
    bin_crossentropy = K.binary_crossentropy(true, pred)
    weights = true * weight_one + (1. - true) * weight_zero
    return K.mean(weights * bin_crossentropy)

# BCE + Dice Loss
def bce_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Combines Binary Crossentropy and Dice Loss.
    """
    return weighted_bincrossentropy(y_true, y_pred) + (1 - dice_coef(y_true, y_pred))

# Accuracy Metric
# Accuracy Metric
def accuracy_metric(real_mask: tf.Tensor, predicted_mask: tf.Tensor) -> float:
    """
    Calculates the accuracy metric.
    """
    real_mask = tf.where(real_mask < 0, 0, real_mask)  # Ensure no negative values
    real_mask_np = real_mask.numpy()  # Convert to NumPy array for operations
    predicted_mask_np = predicted_mask.numpy()  # Convert to NumPy array for operations
    
    total_pixels = real_mask_np.size  # Use NumPy's size attribute
    correct_predictions = np.sum(real_mask_np == predicted_mask_np)
    return correct_predictions / total_pixels


# Updated Evaluation Pipeline
def evaluate_model(prediction_function: Callable[[tf.Tensor], tf.Tensor], 
                   eval_dataset: tf.data.Dataset) -> Tuple[float, float, float, float, float, float, float]:
    """
    Evaluates the model using various metrics including Accuracy.
    """
    IoU_measures = []
    recall_measures = []
    precision_measures = []
    f1_scores = []
    far_scores = []
    accuracy_measures = []
    losses = []
    
    for inputs, labels in tqdm(eval_dataset):
        predictions = prediction_function(inputs)
        for i in range(inputs.shape[0]):
            IoU_measures.append(IoU_metric(labels[i, :, :, 0], predictions[i, :, :]))
            recall_measures.append(recall_metric(labels[i, :, :, 0], predictions[i, :, :]))
            precision_measures.append(precision_metric(labels[i, :, :, 0], predictions[i, :, :]))
            computed_f1_score = calculate_f1_score(labels[i, :, :, 0], predictions[i, :, :])
            f1_scores.append(computed_f1_score)
            #f1_scores.append(f1_score(labels[i, :, :, 0], predictions[i, :, :]))
            far_scores.append(false_alarm_rate(labels[i, :, :, 0], predictions[i, :, :]))
            accuracy_measures.append(accuracy_metric(labels[i, :, :, 0], predictions[i, :, :]))
        labels_cleared = tf.where(labels < 0, 0, labels)
        losses.append(bce_dice_loss(labels_cleared, tf.expand_dims(tf.cast(predictions, tf.float32), axis=-1)))
    
    return (
        np.mean(IoU_measures),
        np.mean(recall_measures),
        np.mean(precision_measures),
        np.mean(f1_scores),
        np.mean(far_scores),
        np.mean(accuracy_measures),
        np.mean(losses)
    )
def train_model(model: Model, train_dataset: tf.data.Dataset, epochs:int=10) -> Tuple[List[float], List[float]]:
    """
    Trains a model using train dataset. (Save weights of model with best IoU)
    
    Args:
        model (Model): Model to train.
        train_dataset (Dataset): Training dataset.
        epochs (int): Number of epochs
    Returns:
        Tuple[List[float], List[float]]: Train losses and Validation losses
    """
    loss_fn = bce_dice_loss
    optimizer = tf.keras.optimizers.Adam()
    batch_losses = []
    val_losses = []
    best_IoU = 0.0
    
    for epoch in range(epochs):
        losses = []
        print(f'Epoch {epoch+1}/{epochs}')
        # Iterate through the dataset
        progress = tqdm(train_dataset)
        for images, masks in progress:
            with tf.GradientTape() as tape:
                # Forward pass
                predictions = model(images, training=True)
                label = tf.where(masks < 0, 0, masks)
                # Compute the loss
                loss = loss_fn(label, predictions)
                losses.append(loss.numpy())
                progress.set_postfix({'batch_loss': loss.numpy()})
            # Compute gradients
            gradients = tape.gradient(loss, model.trainable_variables)
            # Update the model's weights
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        # Evaluate model
        print("Evaluation...")
        IoU, recall, precision, f1_score, far, accuracy, val_loss = evaluate_model(
    lambda x: tf.where(model.predict(x) > 0.5, 1, 0)[:, :, :, 0], validation_dataset
)

        print("Validation set metrics:")
        print(f"Mean IoU: {IoU}\nAccuracy : {accuracy}\nMean precision: {precision}\nMean recall: {recall}\nF1 score:{f1_score}\nFAR :{far}\nValidation loss: {val_loss}\n")
        # Save best model
        if IoU > best_IoU:
            best_IoU = IoU
            model.save_weights("best.weights.h5")
        
        # Print the loss for monitoring
        print(f'Epoch: {epoch}, Train loss: {np.mean(losses)}')
        batch_losses.append(np.mean(losses))
        val_losses.append(val_loss)
    
    print(f"Best model IoU: {best_IoU}")
    return batch_losses, val_losses

# Set reproducability
tf.random.set_seed(1337)

segmentation_model = build_CNN_RNN_AE_model()
train_losses, val_losses = train_model(segmentation_model, train_dataset, epochs=5)
def plot_train_and_val_losses(train_losses, val_losses):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].plot(train_losses)
    axs[0].set_title("train loss")
    
    axs[1].plot(val_losses)
    axs[1].set_title("validation loss")
    
    plt.show()


plot_train_and_val_losses(train_losses, val_losses)
segmentation_model = build_CNN_RNN_AE_model()
segmentation_model.load_weights("/kaggle/input/model_ffd/tensorflow2/default/1/best.weights.h5")
print("Evaluation...")
print("Test set metrics:")
IoU, recall, precision, f1_score, far, accuracy, val_loss = evaluate_model(
    lambda x: tf.where(segmentation_model.predict(x) > 0.5, 1, 0)[:, :, :, 0], test_dataset
)
print(f"Mean IoU: {IoU}\nAccuracy : {accuracy}\nMean precision: {precision}\nMean recall: {recall}\nF1 score:{f1_score}\nFAR :{far}\nTest loss: {val_loss}\n")
def show_inference(n_rows: int, features: tf.Tensor, label: tf.Tensor, prediction_function: Callable[[tf.Tensor], tf.Tensor]) -> None:
    """
    Show model inference through images.
    
    Args:
        n_rows (int): Number of rows for subplots.
        features (tf.Tensor): Input features.
        label (tf.Tensor): True labels.
        prediction_function (Callable[[tf.Tensor], tf.Tensor]): Function for model prediction.
    """
    
    # Variables for controllong the color map for the fire masks
    CMAP = colors.ListedColormap(['black', 'silver', 'orangered'])
    BOUNDS = [-1, -0.1, 0.001, 1]
    NORM = colors.BoundaryNorm(BOUNDS, CMAP.N)
    
    fig = plt.figure(figsize=(15,n_rows*4))
    
    prediction = prediction_function(features)
    for i in range(n_rows):
        plt.subplot(n_rows, 3, i*3 + 1)
        plt.title("Previous day fire")
        plt.imshow(features[i, :, :, -1], cmap=CMAP, norm=NORM)
        plt.axis('off')
        plt.subplot(n_rows, 3, i*3 + 2)
        plt.title("True next day fire")
        plt.imshow(label[i, :, :, 0], cmap=CMAP, norm=NORM)
        plt.axis('off')
        plt.subplot(n_rows, 3, i*3 + 3)
        plt.title("Predicted next day fire")
        plt.imshow(prediction[i, :, :])
        plt.axis('off')    
    plt.tight_layout()
features, labels = next(iter(test_dataset))
show_inference(5, features, labels, lambda x: tf.where(segmentation_model.predict(x) > 0.5, 1, 0)[:,:,:,0])
